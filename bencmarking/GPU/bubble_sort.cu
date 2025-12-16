

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

static int         RUNS_DEFAULT     = 30;
static int         ITERS_DEFAULT    = 7;
static std::string MEASURE_DEFAULT  = "kernel";
static std::string INPUT_DEFAULT    = "C:/Users/user/Desktop/CPU/data/nearly_sorted_10m.csv";
static std::string SORTED_DEFAULT   = "C:/Users/user/Desktop/CPU/data/outputs/output_bubble_gpu.csv";
static std::string BENCH_DEFAULT    = "C:/Users/user/Desktop/CPU/data/benchmark_results/benchmark_bubble_gpu.csv";

static int TILE_DEFAULT            = 1024;  
static int LOCAL_ITERS_DEFAULT     = 1024;  
static int BOUNDARY_SWEEPS_BOOST   = 10;    
static int POLISH_PASSES_DEFAULT   = 64;    
static int EXTRA_REDUNDANT_PASSES  = 32;

#define CUDA_CHECK(stmt) do {                                   \
  cudaError_t err__ = (stmt);                                   \
  if (err__ != cudaSuccess) {                                   \
    std::cerr << "CUDA error: " << cudaGetErrorString(err__)    \
              << " @ " << __FILE__ << ":" << __LINE__ << "\n";  \
    std::exit(1);                                               \
  }                                                             \
} while(0)

struct GPUTimer {
  cudaEvent_t st{}, ed{};
  GPUTimer() { CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&ed)); }
  ~GPUTimer(){ cudaEventDestroy(st); cudaEventDestroy(ed); }
  void start(){ CUDA_CHECK(cudaEventRecord(st)); }
  float stop(){ CUDA_CHECK(cudaEventRecord(ed)); CUDA_CHECK(cudaEventSynchronize(ed));
                float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed)); return ms; }
};

struct WallTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  void start(){ t0 = clock::now(); }
  double stop_ms() const {
    auto t1 = clock::now();
    return std::chrono::duration<double,std::milli>(t1 - t0).count();
  }
};

static std::vector<uint32_t> load_csv_u32(const std::string& path){
  std::ifstream in(path);
  if (!in) { std::cerr << "Failed to open: " << path << "\n"; std::exit(1); }
  std::vector<uint32_t> v; v.reserve(1<<20);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) v.push_back(static_cast<uint32_t>(std::stoll(line)));
  }
  return v;
}
static void write_csv_u32(const std::string& path, const std::vector<uint32_t>& v){
  std::ofstream out(path);
  for (auto x : v) out << x << "\n";
}
static bool check_sorted(const std::vector<uint32_t>& v){
  for (size_t i=1;i<v.size();++i) if (v[i-1] > v[i]) return false;
  return true;
}
static void write_bench_csv(const std::string& path,
                            const std::vector<double>& dur_ms,
                            const std::vector<uint64_t>& mem_bytes){
  std::ofstream out(path);
  out << "Run,Duration(ms),Memory(B)\n";
  out << std::fixed << std::setprecision(3);
  for (size_t i=0;i<dur_ms.size();++i)
    out << (i+1) << "," << dur_ms[i] << "," << mem_bytes[i] << "\n";
}
static double avg(const std::vector<double>& v){
  if (v.empty()) return 0.0;
  long double s=0; for (auto x:v) s+=x; return (double)(s/v.size());
}
static double stdev(const std::vector<double>& v, double mean){
  if (v.empty()) return 0.0;
  long double s=0; for (auto x:v){ long double d=x-mean; s+=d*d; }
  return (double)std::sqrt(s/v.size());
}


namespace bubble {

template <typename T>
struct Less {
  __device__ __host__ inline bool operator()(const T& a, const T& b) const { return a < b; }
};
template <typename T>
struct Greater {
  __device__ __host__ inline bool operator()(const T& a, const T& b) const { return a > b; }
};

template <typename T>
__device__ inline void dswap(T& a, T& b) {
  T tmp = a; a = b; b = tmp;
}

template <typename T, typename Cmp, int TILE>
__global__ void odd_even_tile_kernel(T* __restrict__ data, int n, int local_iters, Cmp cmp) {
  static_assert(TILE % 2 == 0, "TILE must be even");
  __shared__ T s[TILE];

  const int tile_base = blockIdx.x * TILE;
  if (tile_base >= n) return;

  for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
    int gi = tile_base + i;
    s[i] = (gi < n) ? data[gi] : data[n-1];
  }
  __syncthreads();

  for (int it = 0; it < local_iters; ++it) {
    for (int t = threadIdx.x; t < TILE/2; t += blockDim.x) {
      int idx = 2*t;
      T a = s[idx], b = s[idx+1];
      if (!cmp(a, b)) { dswap(a, b); s[idx] = a; s[idx+1] = b; }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < TILE/2; t += blockDim.x) {
      int idx = 2*t + 1;
      if (idx + 1 < TILE) {
        T a = s[idx], b = s[idx+1];
        if (!cmp(a, b)) { dswap(a, b); s[idx] = a; s[idx+1] = b; }
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
    int gi = tile_base + i;
    if (gi < n) data[gi] = s[i];
  }
}

template <typename T, typename Cmp, int TILE>
__global__ void boundary_kernel(T* __restrict__ data, int n, Cmp cmp, int parity) {
  int pairIdx = blockIdx.x * 2 + parity;
  int leftTileBase = pairIdx * TILE;
  int rightTileBase = leftTileBase + TILE;
  if (rightTileBase >= n) return;

  int leftIdx  = leftTileBase + (TILE - 1);
  int rightIdx = rightTileBase;

  if (threadIdx.x == 0) {
    T a = data[leftIdx];
    T b = data[rightIdx];
    if (!cmp(a, b)) {
      data[leftIdx]  = b;
      data[rightIdx] = a;
    }
  }
}

template <typename T, typename Cmp>
__global__ void odd_even_global_kernel(T* __restrict__ data, int n, Cmp cmp, int phase) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + phase;
  if (i + 1 < n) {
    T a = data[i], b = data[i+1];
    if (!cmp(a, b)) { data[i] = b; data[i+1] = a; }
  }
}

template <typename T, typename Cmp>
static void sort_tiled_cfg_impl(T* d_data, int n,
                                int tile,
                                int local_iters,
                                int sweeps_boost,
                                int polish_passes,
                                int extra_redundant_passes,
                                Cmp cmp) {
  if (n <= 1) return;

  if (tile % 2 != 0) tile += 1;   
  tile = std::max(32, tile);
  local_iters = std::max(1, local_iters);

  int tiles = (n + tile - 1) / tile;

  {
    dim3 block(std::min(tile/2, 512));
    if (tile == 512) {
      dim3 gridTiles((n + 512 - 1) / 512);
      odd_even_tile_kernel<T, Cmp, 512><<<gridTiles, block>>>(d_data, n, local_iters, cmp);
    } else {
      dim3 gridTiles((n + 1024 - 1) / 1024);
      dim3 block1024(std::min(1024/2, 512));
      odd_even_tile_kernel<T, Cmp, 1024><<<gridTiles, block1024>>>(d_data, n, local_iters, cmp);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  {
    int neighborPairs = std::max(0, tiles - 1);
    int sweeps = std::max(2, sweeps_boost * tiles);
    if (neighborPairs > 0) {
      for (int s = 0; s < sweeps; ++s) {
        int parity = s & 1;
        int pairsThis = neighborPairs - parity;
        if (pairsThis <= 0) continue;
        dim3 gridBound((pairsThis + 1) / 2);
        if (tile == 512) {
          boundary_kernel<T, Cmp, 512><<<gridBound, 32>>>(d_data, n, cmp, parity);
        } else {
          boundary_kernel<T, Cmp, 1024><<<gridBound, 32>>>(d_data, n, cmp, parity);
        }
        CUDA_CHECK(cudaGetLastError());
      }
    }
  }

  auto global_pass = [&](int passes){
    if (passes <= 0) return;
    int threads = 512;
    int pairs   = n / 2;
    dim3 grid((pairs + threads - 1) / threads);
    for (int p = 0; p < passes; ++p) {
      odd_even_global_kernel<T, Cmp><<<grid, threads>>>(d_data, n, cmp, 0);
      CUDA_CHECK(cudaGetLastError());
      odd_even_global_kernel<T, Cmp><<<grid, threads>>>(d_data, n, cmp, 1);
      CUDA_CHECK(cudaGetLastError());
    }
  };

  global_pass(polish_passes);

  global_pass(extra_redundant_passes);
}

template <typename T>
void sort_cfg(T* d_data, int n, bool descending,
              int tile, int local_iters, int sweeps_boost, int polish_passes,
              int extra_redundant_passes) {
  if (descending) {
    using Cmp = Greater<T>;
    sort_tiled_cfg_impl<T, Cmp>(d_data, n, tile, local_iters, sweeps_boost,
                                polish_passes, extra_redundant_passes, Cmp{});
  } else {
    using Cmp = Less<T>;
    sort_tiled_cfg_impl<T, Cmp>(d_data, n, tile, local_iters, sweeps_boost,
                                polish_passes, extra_redundant_passes, Cmp{});
  }
}

} 

static int g_tile = TILE_DEFAULT;
static int g_local_iters = LOCAL_ITERS_DEFAULT;
static int g_sweeps_boost = BOUNDARY_SWEEPS_BOOST;
static int g_polish_passes = POLISH_PASSES_DEFAULT;
static int g_extra_redundant = EXTRA_REDUNDANT_PASSES;

static float bubble_kernel_once(uint32_t* d_inout, int n){
  GPUTimer T; CUDA_CHECK(cudaDeviceSynchronize()); T.start();

  bubble::sort_cfg<uint32_t>(d_inout, n, false,
                             g_tile, g_local_iters, g_sweeps_boost, g_polish_passes,
                             g_extra_redundant);

  float ms = T.stop(); CUDA_CHECK(cudaDeviceSynchronize());
  return ms;
}

static float bubble_kernel_median(uint32_t* d_buf, int n, int iters, const std::vector<uint32_t>& h_src){
  std::vector<float> times((size_t)iters);
  for (int it=0; it<iters; ++it) {
    CUDA_CHECK(cudaMemcpy(d_buf, h_src.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));
    times[it] = bubble_kernel_once(d_buf, n);
  }
  std::sort(times.begin(), times.end());
  return times[iters/2];
}

int main(int argc, char** argv){
  std::string input_path  = INPUT_DEFAULT;
  std::string sorted_path = SORTED_DEFAULT;
  std::string bench_path  = BENCH_DEFAULT;
  std::string measure     = MEASURE_DEFAULT;
  int runs  = RUNS_DEFAULT;
  int iters = ITERS_DEFAULT;
  bool verify = false;

  if (argc >= 2 && argv[1][0] != '-') input_path = argv[1];
  for (int i=1; i<argc; ++i) {
    std::string s = argv[i];
    if (s == "--verify") verify = true;
    else if (s.rfind("--measure=",0)==0)      measure = s.substr(10);
    else if (s.rfind("--runs=",0)==0)         runs = std::max(1, std::stoi(s.substr(7)));
    else if (s.rfind("--iters=",0)==0)        iters = std::max(1, std::stoi(s.substr(8)));
    else if (s.rfind("--input=",0)==0)        input_path = s.substr(8);
    else if (s.rfind("--sorted=",0)==0)       sorted_path = s.substr(9);
    else if (s.rfind("--bench=",0)==0)        bench_path  = s.substr(8);
    else if (s.rfind("--tile=",0)==0)         g_tile = std::max(32, std::stoi(s.substr(7)));
    else if (s.rfind("--localIters=",0)==0)   g_local_iters = std::max(1, std::stoi(s.substr(13)));
    else if (s.rfind("--sweepsBoost=",0)==0)  g_sweeps_boost = std::max(1, std::stoi(s.substr(14)));
    else if (s.rfind("--polish=",0)==0)       g_polish_passes = std::max(0, std::stoi(s.substr(9)));
  }

  auto h_in = load_csv_u32(input_path);
  const int n = (int)h_in.size();
  std::cout << "Loaded n=" << n << " from " << input_path << "\n";

  int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "CUDA devices: " << deviceCount << "\n";
  CUDA_CHECK(cudaFree(0)); 

  uint32_t *d_a=nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(uint32_t)*n));

  uint64_t mem_est = static_cast<uint64_t>(sizeof(uint32_t)) * static_cast<uint64_t>(n);

  std::vector<double>   durations_ms(runs);
  std::vector<uint64_t> mem_bytes(runs, mem_est);
  std::vector<uint32_t> h_out(n);

  if (measure == "kernel") {
    for (int run=0; run<runs; ++run) {
      float ms = bubble_kernel_median(d_a, n, iters, h_in);
      durations_ms[run] = (double)ms;
      std::cout << "Run " << std::setw(2) << (run+1) << ": "
                << std::fixed << std::setprecision(3) << durations_ms[run]
                << " ms | est. peak mem (bytes): " << mem_est << "\n";
    }
    CUDA_CHECK(cudaMemcpy(d_a, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));
    bubble::sort_cfg<uint32_t>(d_a, n, false,
                               g_tile, g_local_iters, g_sweeps_boost, g_polish_passes,
                               g_extra_redundant);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
  }
  else if (measure == "device") {
    for (int run=0; run<runs; ++run) {
      double sum = 0.0;
      for (int it=0; it<iters; ++it) {
        WallTimer T; T.start();
        CUDA_CHECK(cudaMemcpy(d_a, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));

        bubble::sort_cfg<uint32_t>(d_a, n, false,
                                   g_tile, g_local_iters, g_sweeps_boost, g_polish_passes,
                                   g_extra_redundant);

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        sum += T.stop_ms();
      }
      durations_ms[run] = sum / iters;
      std::cout << "Run " << std::setw(2) << (run+1) << ": "
                << std::fixed << std::setprecision(3) << durations_ms[run]
                << " ms | est. peak mem (bytes): " << mem_est << "\n";
    }
  }
  else if (measure == "full") {
    for (int run=0; run<runs; ++run) {
      double sum = 0.0;
      for (int it=0; it<iters; ++it) {
        WallTimer T; T.start();
        auto h2 = load_csv_u32(input_path); 
        CUDA_CHECK(cudaMemcpy(d_a, h2.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));

        bubble::sort_cfg<uint32_t>(d_a, n, false,
                                   g_tile, g_local_iters, g_sweeps_boost, g_polish_passes,
                                   g_extra_redundant);

        CUDA_CHECK(cudaMemcpy(h2.data(), d_a, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        sum += T.stop_ms();
        if (it == iters-1) h_out = std::move(h2);
      }
      durations_ms[run] = sum / iters;
      std::cout << "Run " << std::setw(2) << (run+1) << ": "
                << std::fixed << std::setprecision(3) << durations_ms[run]
                << " ms | est. peak mem (bytes): " << mem_est << "\n";
    }
  }
  else {
    std::cerr << "Unknown --measure=" << measure << " (use kernel|device|full)\n";
    CUDA_CHECK(cudaFree(d_a));
    return 2;
  }



  double aT   = avg(durations_ms);
  double sT   = stdev(durations_ms, aT);
  double sPct = (aT == 0.0 ? 0.0 : (sT / aT) * 100.0);
  double mnT  = *std::min_element(durations_ms.begin(), durations_ms.end());
  double mxT  = *std::max_element(durations_ms.begin(), durations_ms.end());

  uint64_t aM  = mem_est;
  uint64_t mnM = aM, mxM = aM;

  std::string interp = (sPct < 5.0) ? "Very stable performance"
                     : (sPct <= 15.0) ? "Fairly stable performance"
                                      : "Some variation in performance";

  std::cout << "\nBenchmark Summary (" << runs << " runs, mode=" << measure << ")\n";
  std::cout << "  Average time: " << std::fixed << std::setprecision(3) << aT << " ms\n";
  std::cout << "  Std deviation: " << sT << " ms (" << sPct << "%)\n";
  std::cout << "  Min: " << mnT << " ms | Max: " << mxT << " ms\n";
  std::cout << "  Peak memory (bytes, est.): " << aM
            << " | min: " << mnM << " | max: " << mxM << "\n";
  std::cout << "  Interpretation: " << interp << "\n";

  write_bench_csv(bench_path, durations_ms, mem_bytes);
  std::cout << "Benchmark report saved: " << bench_path << "\n";

  write_csv_u32(sorted_path, h_out);
  std::cout << "Sorted file saved: " << sorted_path << "\n";

  CUDA_CHECK(cudaFree(d_a));
  return 0;
}
