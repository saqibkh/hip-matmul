// Sequential CPU emulation of the control flow of MmtKernel_StreamKTree_..._Kx2::run (matmul.hip).
// Each workgroup is run to completion, one after another, in a given order. Workgroups never wait on each
// other (wait-free), so every such order is a legal GPU schedule. Accumulator = 1 scalar per tile.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <bit>
#include <cassert>
using namespace std;
static int ceil_div(int a, int b) { return (a + b - 1) / b; }
struct Emu {
  int M, N, K; int MaxWG = 304;
  int wgc, tiles, per_tile; vector<uint32_t> ctr; vector<double> loc, C; vector<double> part; // part[tile*K+k]
  bool oob = false;
  void init(int M_, int N_, int K_, mt19937 &r) {
    M = M_; N = N_; K = K_; tiles = M * N;
    int64_t ti = (int64_t)M * N * K; wgc = min<int64_t>(MaxWG, ti);
    per_tile = 1 + ceil_div(wgc, tiles);
    ctr.assign(round_up(wgc * 2) + 64, 0); loc.assign((size_t)per_tile * tiles, 0); C.assign(tiles, -12345.0);
    part.resize((size_t)tiles * K); for (auto &p : part) p = (int)(r() % 11) - 5;
    oob = false;
  }
  static int round_up(int x) { return (x + 31) / 32 * 32; }
  double &L(int idx) { if (idx < 0 || idx >= (int)loc.size()) { oob = true; static double d; return d; } return loc[idx]; }
  uint32_t &Ctr(int idx) { if (idx < 0 || idx >= wgc * 2) { oob = true; static uint32_t d; return d; } return ctr[idx]; }
  void run_wg(int bid) {
    int total_iters = tiles * K, workgroup = bid;
    if (wgc == MaxWG) workgroup = (workgroup % 8) * (wgc / 8) + (workgroup / 8);
    int iter_start = (int)((int64_t)workgroup * total_iters / wgc), iter_end = (int)((int64_t)(workgroup + 1) * total_iters / wgc);
    int iter = iter_start;
    while (iter < iter_end) {
      int tile_idx = iter / K, tile_iter_start = tile_idx * K, tile_iter_end = tile_iter_start + K;
      int local_iter = iter - tile_iter_start, local_iter_end = min(tile_iter_end, iter_end) - tile_iter_start;
      if (local_iter == local_iter_end) break;
      double acc = 0; for (int k = local_iter; k < local_iter_end; ++k) acc += part[(size_t)tile_idx * K + k];
      int cs = ceil_div((tile_iter_start + 1) * wgc, total_iters) - 1;
      int ce = ceil_div(tile_iter_end * wgc, total_iters);
      assert(workgroup >= cs && workgroup < ce);
      int cc = ce - cs; bool store_final = true;
      if (cc != 1) {
        int widx = workgroup - cs;
        int depth = 32 - countl_zero((uint32_t)(cc - 1));
        for (int lvl = 0; lvl < depth; ++lvl) {
          int bit = 1 << lvl;
          int lai = (widx & ~(bit - 1)) + tile_idx * per_tile;
          int slai = ((widx & ~(bit - 1)) ^ bit) + tile_idx * per_tile;
          int parent = widx & ~((bit << 1) - 1);
          if ((parent | bit) >= cc) continue;
          L(lai) = acc;
          int nudge = tile_idx & 1;
          uint32_t &c = Ctr(cs * 2 + 2 * parent + nudge);
          uint32_t old = c; c |= bit;
          if (!(old & bit)) { store_final = false; break; }
          acc += L(slai);
        }
      }
      if (store_final) {
        C[tile_idx] = acc;
        for (int i = 0; i < cc; ++i) Ctr(cs * 2 + 2 * i + (tile_idx & 1)) = 0;
      }
      iter = tile_iter_end;
    }
  }
  int check() { int bad = 0; for (int t = 0; t < tiles; ++t) { double e = 0; for (int k = 0; k < K; ++k) e += part[(size_t)t * K + k]; if (C[t] != e) ++bad; } return bad; }
};
int main() {
  mt19937 r(1);
  int shapes[][3] = {{20,20,20},{305,1,1},{5,2,3},{4,4,8},{2,3,4},{1,1,10}};
  for (auto &s : shapes) {
    Emu e; int nbadmax_asc = 0, nbad_desc = 0, nbad_rand = 0, oobs = 0;
    for (int mode = 0; mode < 3; ++mode) for (int rep = 0; rep < (mode == 2 ? 200 : 1); ++rep) {
      e.init(s[0], s[1], s[2], r);
      vector<int> ord(e.wgc); for (int i = 0; i < e.wgc; ++i) ord[i] = i;
      if (mode == 1) reverse(ord.begin(), ord.end()); if (mode == 2) shuffle(ord.begin(), ord.end(), r);
      for (int w : ord) e.run_wg(w);
      int b = e.check(); oobs += e.oob;
      (mode == 0 ? nbadmax_asc : mode == 1 ? nbad_desc : nbad_rand) = max(mode == 0 ? nbadmax_asc : mode == 1 ? nbad_desc : nbad_rand, b);
    }
    printf("shape %dx%dx%d wgc=%d per_tile=%d: wrong tiles asc=%d desc=%d random(max of 200)=%d, out-of-bounds in %d runs\n", s[0], s[1], s[2], e.wgc, e.per_tile, nbadmax_asc, nbad_desc, nbad_rand, oobs);
  }
}
