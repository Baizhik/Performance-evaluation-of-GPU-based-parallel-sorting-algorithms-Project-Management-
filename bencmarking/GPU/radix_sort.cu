

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
#include <cub/cub.cuh>


static int         RUNS_DEFAULT     = 30;
static int         ITERS_DEFAULT    = 7;     
static std::string MEASURE_DEFAULT  = "kernel";  
static std::string INPUT_DEFAULT    = "C:/Users/user/Desktop/CPU/data/nearly_sorted_10m.csv";
static std::string SORTED_DEFAULT   = "C:/Users/user/Desktop/CPU/data/outputs/output_radix_gpu.csv";
static std::string BENCH_DEFAULT    = "C:/Users/user/Desktop/CPU/data/benchmark_results/benchmark_radix_gpu.csv";

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

static float cub_kernel_once(uint32_t* d_in, uint32_t* d_out, int n,
                             void* d_temp, size_t temp_bytes){
  GPUTimer T; T.start();
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n));
  return T.stop(); 
}

static float cub_kernel_median(uint32_t* d_inout, int n, int iters){
  uint32_t* d_aux = nullptr;
  CUDA_CHECK(cudaMalloc(&d_aux, sizeof(uint32_t)*n));

  void* d_temp = nullptr; size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_inout, d_aux, n));
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_inout, d_aux, n));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> times((size_t)iters);
  uint32_t* src = d_inout;
  uint32_t* dst = d_aux;

  for (int it=0; it<iters; ++it) {
    GPUTimer T; CUDA_CHECK(cudaDeviceSynchronize()); T.start();
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, src, dst, n));
    float ms = T.stop(); CUDA_CHECK(cudaDeviceSynchronize());
    times[it] = ms;
    std::swap(src, dst);
  }
  if (dst != d_inout) {
    CUDA_CHECK(cudaMemcpy(d_inout, dst, sizeof(uint32_t)*n, cudaMemcpyDeviceToDevice));
  }
  std::sort(times.begin(), times.end());
  float med = times[iters/2];

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_aux));
  return med;
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
    else if (s.rfind("--measure=",0)==0) measure = s.substr(10);
    else if (s.rfind("--runs=",0)==0)    runs = std::max(1, std::stoi(s.substr(7)));
    else if (s.rfind("--iters=",0)==0)   iters = std::max(1, std::stoi(s.substr(8)));
    else if (s.rfind("--input=",0)==0)   input_path = s.substr(8);
    else if (s.rfind("--sorted=",0)==0)  sorted_path = s.substr(9);
    else if (s.rfind("--bench=",0)==0)   bench_path  = s.substr(8);
  }

  auto h_in = load_csv_u32(input_path);
  const int n = (int)h_in.size();
  std::cout << "Loaded n=" << n << " from " << input_path << "\n";

  int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "CUDA devices: " << deviceCount << "\n";
  CUDA_CHECK(cudaFree(0)); 

  uint32_t *d_a=nullptr, *d_b=nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(uint32_t)*n));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(uint32_t)*n));

  void* d_temp = nullptr; size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_a, d_b, n));
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  uint64_t mem_est = static_cast<uint64_t>(sizeof(uint32_t)) * static_cast<uint64_t>(n) 
                   + static_cast<uint64_t>(sizeof(uint32_t)) * static_cast<uint64_t>(n) 
                   + static_cast<uint64_t>(temp_bytes);                                   

  std::vector<double>  durations_ms(runs);     
  std::vector<uint64_t> mem_bytes(runs, mem_est);

  std::vector<uint32_t> h_out(n);

  if (measure == "kernel") {
    for (int run=0; run<runs; ++run) {
      CUDA_CHECK(cudaMemcpy(d_a, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));
      float ms = cub_kernel_median(d_a, n, iters);
      durations_ms[run] = (double)ms; 
      std::cout << "Run " << std::setw(2) << (run+1) << ": "
                << std::fixed << std::setprecision(3) << durations_ms[run]
                << " ms | est. peak mem (bytes): " << mem_est << "\n";
    }
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
  }
  else if (measure == "device") {
    for (int run=0; run<runs; ++run) {
      double sum = 0.0;
      for (int it=0; it<iters; ++it) {
        WallTimer T; T.start();
        CUDA_CHECK(cudaMemcpy(d_a, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_a, d_b, n));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_b, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
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
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_a, d_b, n));
        CUDA_CHECK(cudaMemcpy(h2.data(), d_b, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaFree(d_temp)); CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    return 2;
  }

  if (verify) {
    std::cout << "verify: " << (check_sorted(h_out) ? "OK" : "FAIL") << "\n";
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

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  return 0;
}
