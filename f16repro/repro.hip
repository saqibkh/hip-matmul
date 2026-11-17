// Standalone reproducer: copy of MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8,8>::run
// plus a trivial marker kernel, on grids of different shapes. A = (m_outer+1), B = (n_outer+1), so every
// element of block (m,n) must equal (m+1)*(n+1)*32*K_outer.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <map>
#define CK(x) do { hipError_t e = (x); if (e) { printf("HIP error %d (%s) at line %d\n", (int)e, hipGetErrorString(e), __LINE__); return 1; } } while (0)
constexpr int MS = 8, NS = 8;
using float16x4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using float16x8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__global__ __launch_bounds__(256) void kern(const void *A_data, const void *B_data, void *C_data, int M_outer, int N_outer, int K_outer) {
  floatx4_t acc[MS][NS / 4] = {{0}};
  int m_outer = blockIdx.x, n_outer = blockIdx.y, tid = threadIdx.x;
  constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
  constexpr int B_tile_size_in_vec8 = NS * 16 * 4;
  const float16x8_t *A_global = static_cast<const float16x8_t *>(A_data) + m_outer * K_outer * A_tile_size_in_vec8;
  const float16x8_t *B_global = static_cast<const float16x8_t *>(B_data) + n_outer * K_outer * B_tile_size_in_vec8;
  const float16x8_t *A_global_ptr = A_global + tid;
  const float16x8_t *B_global_ptr = B_global + tid;
  __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
  __shared__ float16x8_t B_shared[B_tile_size_in_vec8];
  for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
    for (int i = 0; i < A_tile_size_in_vec8; i += 256) A_shared[i + tid] = A_global_ptr[i];
    for (int j = 0; j < B_tile_size_in_vec8; j += 256) B_shared[j + tid] = B_global_ptr[j];
    A_global_ptr += A_tile_size_in_vec8;
    B_global_ptr += B_tile_size_in_vec8;
    __syncthreads();
    for (int i = 0; i < MS; ++i)
      for (int j = 0; j < NS / 4; ++j) {
        float16x8_t a = A_shared[64 * i + (tid % 64)];
        float16x8_t b = B_shared[256 * j + tid];
        float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
        float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
        float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
        float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
        acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc[i][j], 0, 0, 0);
        acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc[i][j], 0, 0, 0);
      }
    __syncthreads();
  }
  floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) + MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
  for (int i = 0; i < MS; ++i)
    for (int j = 0; j < NS / 4; ++j) C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
}

// Trivial kernel with the same addressing and same shared-memory footprint, no MFMA: writes block id to C.
__global__ __launch_bounds__(256) void marker(void *C_data, int M_outer, int N_outer) {
  __shared__ float16x8_t s1[512], s2[512];
  s1[threadIdx.x] = (float16x8_t){0}; s2[threadIdx.x] = (float16x8_t){0};
  __syncthreads();
  floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) + MS * NS * 16 * 4 * (N_outer * blockIdx.x + blockIdx.y);
  float v = 1.0f + blockIdx.x * 1000 + blockIdx.y;
  for (int q = 0; q < 16; ++q) C_ptr[256 * q + threadIdx.x] = (floatx4_t){v, v, v, v};
}

int run(int M, int N, int K, bool real) {
  size_t A_n = (size_t)M * K * 512 * 8, B_n = (size_t)N * K * 512 * 8, C_n = (size_t)M * N * 16384;
  std::vector<_Float16> A(A_n), B(B_n);
  for (int m = 0; m < M; ++m) for (size_t i = 0; i < (size_t)K * 4096; ++i) A[(size_t)m * K * 4096 + i] = (_Float16)(m + 1);
  for (int n = 0; n < N; ++n) for (size_t i = 0; i < (size_t)K * 4096; ++i) B[(size_t)n * K * 4096 + i] = (_Float16)(n + 1);
  std::vector<float> C(C_n, -1.f);
  void *dA, *dB, *dC;
  CK(hipMalloc(&dA, A_n * 2)); CK(hipMalloc(&dB, B_n * 2)); CK(hipMalloc(&dC, C_n * 4));
  CK(hipMemcpy(dA, A.data(), A_n * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dB, B.data(), B_n * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dC, C.data(), C_n * 4, hipMemcpyHostToDevice));
  if (real) kern<<<dim3(M, N), 256>>>(dA, dB, dC, M, N, K);
  else marker<<<dim3(M, N), 256>>>(dC, M, N);
  CK(hipGetLastError()); CK(hipDeviceSynchronize());
  CK(hipMemcpy(C.data(), dC, C_n * 4, hipMemcpyDeviceToHost));
  int badblocks = 0; long badelems = 0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float exp = real ? (float)(m + 1) * (n + 1) * 32 * K : 1.0f + m * 1000 + n;
    long bad = 0; std::map<float, long> h;
    for (int e = 0; e < 16384; ++e) { float v = C[(size_t)(N * m + n) * 16384 + e]; if (v != exp) { ++bad; ++h[v]; } }
    if (bad) {
      ++badblocks; badelems += bad;
      if (badblocks <= 5) { printf("   block (%d,%d): %ld/16384 wrong, expected %g; first values:", m, n, bad, exp); int c = 0; for (auto &p : h) { if (c++ >= 4) break; printf(" %g x%ld", p.first, p.second); } printf("\n"); }
    }
  }
  printf("%s grid %dx%d K_outer=%d: %d/%d blocks wrong, %ld elems wrong\n", real ? "mfma  " : "marker", M, N, K, badblocks, M * N, badelems);
  hipFree(dA); hipFree(dB); hipFree(dC);
  return 0;
}
int main() {
  int shapes[][3] = {{1,1,1},{1,1,2},{1,1,20},{2,1,3},{4,4,8},{19,1,2},{20,1,2},{1,20,2},{20,20,1},{20,20,2},{20,20,20}};
  for (bool real : {false, true}) for (auto &s : shapes) { if (run(s[0], s[1], s[2], real)) return 1; fflush(stdout); }
}
