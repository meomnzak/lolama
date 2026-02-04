// metal_ext.cpp — Obj-C++ Metal extension for fused W8A16 dequant+matmul
// Compiled as Obj-C++ via extra_cflags=['-ObjC++']
// Linked against Metal + Foundation frameworks
//
// Optimized kernel using:
//   - simdgroup_matrix hardware 8×8 matmul (Apple Silicon AMX/GPU matrix units)
//   - 32×32 output tiles (4× larger than naive → 2× arithmetic intensity)
//   - 4 simdgroups per threadgroup, each computing 16×16 via 2×2 grid of 8×8
//   - K-tile of 32, inner loop of 4 simdgroup multiplies per tile load
//   - Float accumulators for precision, fp16 output

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static const char* METAL_SHADER_SOURCE = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Optimized W8A16 dequant GEMM ──────────────────────────────────────────
// output[M,N] = x[M,K] @ W_int8[N,K]^T   (per-channel dequant)
//
// Threadgroup : 128 threads = 4 simdgroups (sg0–sg3)
// Output tile : 32×32
// K tile      : 32  (4 inner simdgroup_matrix iterations of 8)
// Shared mem  : x_sm[32][32] + w_sm[32][32] + out_buf[32][32]
//             = 2 KB + 2 KB + 4 KB = 8 KB
//
// Simdgroup layout in the 32×32 output tile:
//   sg0 → [0:16,  0:16]    sg1 → [0:16,  16:32]
//   sg2 → [16:32, 0:16]    sg3 → [16:32, 16:32]
// Each simdgroup accumulates 2×2 = 4 blocks of 8×8 via simdgroup_matrix.

kernel void dequant_matmul_kernel(
    device const half*   x       [[buffer(0)]],
    device const char*   w_int8  [[buffer(1)]],
    device const half*   scales  [[buffer(2)]],
    device half*         output  [[buffer(3)]],
    constant uint&       M       [[buffer(4)]],
    constant uint&       N       [[buffer(5)]],
    constant uint&       K       [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint  simd_id  [[simdgroup_index_in_threadgroup]],
    uint  tid      [[thread_index_in_threadgroup]]
) {
    const uint TILE  = 32;
    const uint KTILE = 32;

    uint base_m = group_id.y * TILE;
    uint base_n = group_id.x * TILE;

    // Which 16×16 subtile this simdgroup owns
    uint sg_m = (simd_id / 2) * 16;   // 0 or 16
    uint sg_n = (simd_id % 2) * 16;   // 0 or 16

    // 2×2 grid of 8×8 float accumulators per simdgroup
    simdgroup_matrix<float, 8, 8> acc00(0), acc01(0), acc10(0), acc11(0);

    // Shared memory tiles (reused every K iteration)
    threadgroup half x_sm[TILE][KTILE];    // 32×32 fp16 = 2 KB
    threadgroup half w_sm[TILE][KTILE];    // 32×32 fp16 = 2 KB (dequantized)

    for (uint k_base = 0; k_base < K; k_base += KTILE) {

        // ── Cooperative load: 128 threads, 1024 elems each tile, 8 per thread ──
        for (uint i = 0; i < 8; i++) {
            uint elem = tid + i * 128;          // strided for coalescing
            uint row  = elem / KTILE;
            uint col  = elem % KTILE;
            uint gm   = base_m + row;
            uint gk   = k_base + col;
            x_sm[row][col] = (gm < M && gk < K) ? x[gm * K + gk] : half(0);
        }

        for (uint i = 0; i < 8; i++) {
            uint elem = tid + i * 128;
            uint row  = elem / KTILE;
            uint col  = elem % KTILE;
            uint gn   = base_n + row;
            uint gk   = k_base + col;
            if (gn < N && gk < K) {
                w_sm[row][col] = half(float(w_int8[gn * K + gk]) * float(scales[gn]));
            } else {
                w_sm[row][col] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Inner loop: 4 simdgroup_matrix multiplies (KTILE / 8 = 4) ──
        for (uint ki = 0; ki < KTILE; ki += 8) {
            simdgroup_matrix<half, 8, 8> a0, a1, b0, b1;

            // A tiles from x_sm: this simdgroup's M rows
            simdgroup_load(a0, &x_sm[sg_m    ][ki], KTILE);
            simdgroup_load(a1, &x_sm[sg_m + 8][ki], KTILE);

            // B tiles from w_sm: transposed (w is [N,K], we need w^T for matmul)
            simdgroup_load(b0, &w_sm[sg_n    ][ki], KTILE, ulong2(0), true);
            simdgroup_load(b1, &w_sm[sg_n + 8][ki], KTILE, ulong2(0), true);

            // 2×2 multiply-accumulate (half inputs → float accumulator)
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Store: float accumulators → threadgroup buffer → half global ──
    threadgroup float out_buf[TILE][TILE];  // 32×32 float = 4 KB

    simdgroup_store(acc00, &out_buf[sg_m    ][sg_n    ], TILE);
    simdgroup_store(acc01, &out_buf[sg_m    ][sg_n + 8], TILE);
    simdgroup_store(acc10, &out_buf[sg_m + 8][sg_n    ], TILE);
    simdgroup_store(acc11, &out_buf[sg_m + 8][sg_n + 8], TILE);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 128 threads write 1024 elements (8 each, strided for coalescing)
    for (uint i = 0; i < 8; i++) {
        uint elem = tid + i * 128;
        uint row  = elem / TILE;
        uint col  = elem % TILE;
        uint gm   = base_m + row;
        uint gn   = base_n + col;
        if (gm < M && gn < N) {
            output[gm * N + gn] = half(out_buf[row][col]);
        }
    }
}
)METAL";


// ─── C++ Dispatch ───────────────────────────────────────────────────────────

static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static bool g_initialized = false;
static bool g_init_failed = false;

static bool ensure_metal_initialized(id<MTLDevice> device) {
    if (g_initialized) return !g_init_failed;
    g_initialized = true;

    @autoreleasepool {
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:METAL_SHADER_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;
        options.languageVersion = MTLLanguageVersion3_0;

        g_library = [device newLibraryWithSource:source options:options error:&error];
        if (!g_library) {
            NSLog(@"Metal shader compile error: %@", error);
            g_init_failed = true;
            return false;
        }

        id<MTLFunction> function = [g_library newFunctionWithName:@"dequant_matmul_kernel"];
        if (!function) {
            NSLog(@"Metal function 'dequant_matmul_kernel' not found");
            g_init_failed = true;
            return false;
        }

        g_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!g_pipeline) {
            NSLog(@"Metal pipeline error: %@", error);
            g_init_failed = true;
            return false;
        }
    }

    return true;
}

static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor dequant_matmul(
    torch::Tensor x,         // [M, K] fp16 on MPS
    torch::Tensor w_int8,    // [N, K] int8 on MPS
    torch::Tensor scales     // [N]    fp16 on MPS
) {
    TORCH_CHECK(x.is_mps(), "x must be on MPS device");
    TORCH_CHECK(w_int8.is_mps(), "w_int8 must be on MPS device");
    TORCH_CHECK(scales.is_mps(), "scales must be on MPS device");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(w_int8.dtype() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [N, K]");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D [N]");

    int64_t M = x.size(0);
    int64_t K = x.size(1);
    int64_t N = w_int8.size(0);

    TORCH_CHECK(w_int8.size(1) == K, "K dimension mismatch");
    TORCH_CHECK(scales.size(0) == N, "scales length must match N");

    x = x.contiguous();
    w_int8 = w_int8.contiguous();
    scales = scales.contiguous();

    at::mps::MPSStream* mpsStream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = mpsStream->device();

    TORCH_CHECK(ensure_metal_initialized(device), "Metal initialization failed");

    auto output = torch::empty({M, N}, x.options());

    id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();
    [encoder setComputePipelineState:g_pipeline];

    [encoder setBuffer:getMTLBufferStorage(x)       offset:x.storage_offset() * x.element_size()           atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(w_int8)   offset:w_int8.storage_offset() * w_int8.element_size() atIndex:1];
    [encoder setBuffer:getMTLBufferStorage(scales)   offset:scales.storage_offset() * scales.element_size() atIndex:2];
    [encoder setBuffer:getMTLBufferStorage(output)   offset:0                                               atIndex:3];

    uint32_t m = static_cast<uint32_t>(M);
    uint32_t n = static_cast<uint32_t>(N);
    uint32_t k = static_cast<uint32_t>(K);
    [encoder setBytes:&m length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&k length:sizeof(uint32_t) atIndex:6];

    // 128 threads = 4 simdgroups of 32; grid tiles are 32×32
    MTLSize threadgroupSize = MTLSizeMake(128, 1, 1);
    MTLSize gridSize = MTLSizeMake(
        (N + 31) / 32,
        (M + 31) / 32,
        1
    );

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_matmul", &dequant_matmul,
          "Fused W8A16 dequant+matmul on Metal (MPS) — simdgroup_matrix optimized");
}
