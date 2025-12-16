
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(stmt) do { \
  cudaError_t err__ = (stmt); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
              << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

struct GPUTimer {
  cudaEvent_t st{}, ed{};
  GPUTimer(){ CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&ed)); }
  ~GPUTimer(){ cudaEventDestroy(st); cudaEventDestroy(ed); }
  void start(){ CUDA_CHECK(cudaEventRecord(st)); }
  float stop(){ CUDA_CHECK(cudaEventRecord(ed)); CUDA_CHECK(cudaEventSynchronize(ed));
                float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed)); return ms; }
};

namespace memtrack {
  static std::atomic<long long> live{0};
  static std::atomic<long long> peak{0};
  static std::atomic<long long> allocs{0};
  static std::atomic<long long> frees{0};
  static std::mutex map_mu;
  static std::unordered_map<void*, size_t> sizes;

  inline void reset(){
    live.store(0); peak.store(0); allocs.store(0); frees.store(0);
    std::lock_guard<std::mutex> g(map_mu); sizes.clear();
  }
  inline void on_alloc(void* p, size_t sz){
    if(!p) return;
    { std::lock_guard<std::mutex> g(map_mu); sizes[p] = sz; }
    long long now = (live += (long long)sz);
    long long prev = peak.load(std::memory_order_relaxed);
    while (now > prev && !peak.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {}
    allocs.fetch_add(1, std::memory_order_relaxed);
  }
  inline void on_free(void* p){
    if(!p) return;
    size_t sz=0;
    { std::lock_guard<std::mutex> g(map_mu);
      auto it=sizes.find(p); if(it!=sizes.end()){ sz=it->second; sizes.erase(it); } }
    if(sz) live -= (long long)sz;
    frees.fetch_add(1, std::memory_order_relaxed);
  }
}
template<typename T>
static inline cudaError_t cudaMallocTracked(T** p, size_t sz){
  void* vp=nullptr; cudaError_t e=cudaMalloc(&vp, sz);
  if(e==cudaSuccess){ *p=static_cast<T*>(vp); memtrack::on_alloc(vp, sz); }
  return e;
}
static inline cudaError_t cudaFreeTracked(void* p){ memtrack::on_free(p); return cudaFree(p); }
#define cudaMalloc cudaMallocTracked
#define cudaFree   cudaFreeTracked

static std::vector<uint32_t> load_csv_u32(const std::string& path){
  std::ifstream in(path);
  if(!in){ std::cerr<<"Failed to open: "<<path<<"\n"; std::exit(1); }
  std::vector<uint32_t> v; v.reserve(1<<20);
  std::string line;
  while(std::getline(in,line)){ if(!line.empty()) v.push_back((uint32_t)std::stoll(line)); }
  return v;
}
static void write_csv_u32(const std::string& path, const std::vector<uint32_t>& v){
  std::filesystem::path p(path);
  if (!p.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec); // ok if exists
  }
  std::ofstream out(path);
  if (!out) { std::cerr << "Failed to open for write: " << path << "\n"; std::exit(1); }
  out << std::fixed << std::setprecision(10);
  for (auto x : v) out << (double)x << "\n";
  std::cout << "sorted file written: " << std::filesystem::absolute(p).string() << "\n";
}
 

static void write_bench_csv(const std::string& path,
                            const std::vector<int>& run_ids,
                            const std::vector<int>& iter_ids,
                            const std::vector<double>& times_ms,
                            const std::vector<double>& peak_mb){
  std::ofstream out(path);
  out<<"Run,Iter,Kernel(ms),PeakDev(MB)\n";
  out<<std::fixed<<std::setprecision(6);
  for(size_t i=0;i<times_ms.size();++i)
    out<<run_ids[i]<<","<<iter_ids[i]<<","<<times_ms[i]<<","<<peak_mb[i]<<"\n";
}
static bool check_sorted(const std::vector<uint32_t>& v){
  for(size_t i=1;i<v.size();++i) if(v[i-1]>v[i]) return false;
  return true;
}
static double avg(const std::vector<double>& v){
  if(v.empty()) return 0.0; long double s=0; for(double x: v) s+=x; return (double)(s/v.size());
}
static double stdev(const std::vector<double>& v, double mean){
  if(v.empty()) return 0.0; long double s=0; for(double x: v){ long double d=x-mean; s+=d*d; }
  return (double)std::sqrt(s/v.size());
}

__device__ __forceinline__ uint32_t neg_inf_u32(){ return 0u; }
__device__ __forceinline__ uint32_t pos_inf_u32(){ return 0xFFFFFFFFu; }

__device__ int merge_path_cut(const uint32_t* A, int a0, int aLen,
                              const uint32_t* B, int b0, int bLen,
                              int diag){
  int lo=max(0, diag-bLen);
  int hi=min(diag, aLen);
  while(lo<hi){
    int i=(lo+hi)>>1;
    int j=diag-i;
    uint32_t Ai_1=(i==0)?neg_inf_u32():A[a0+i-1];
    uint32_t Bj  =(j==bLen)?pos_inf_u32():B[b0+j];
    if(Ai_1>Bj) hi=i; else lo=i+1;
  }
  return lo;
}

template<int TILE>
__global__ void small_bitonic(uint32_t* d, int n){
  __shared__ uint32_t s[TILE];
  const int base = blockIdx.x * TILE;
  const int tid  = threadIdx.x;

  if (base + tid < n) s[tid] = d[base + tid];
  else                s[tid] = 0xFFFFFFFFu;
  __syncthreads();

  for (int k=2; k<=TILE; k<<=1){
    for (int j=k>>1; j>0; j>>=1){
      int ixj = tid ^ j;
      if (ixj > tid){
        bool up = ((tid & k) == 0);
        uint32_t a = s[tid], b = s[ixj];
        bool sw = up ? (a > b) : (a < b);
        if (sw){ s[tid] = b; s[ixj] = a; }
      }
      __syncthreads();
    }
  }

  if (base + tid < n) d[base + tid] = s[tid];
}

template<int UNUSED_TILE>
__global__ void merge_pass_simple(const uint32_t* __restrict__ in,
                                  uint32_t* __restrict__ out,
                                  int n, int w){
  const int pair = blockIdx.x;
  const int s = pair*(2*w);
  if (s >= n) return;

  const int m = min(s+w, n);
  const int e = min(s+2*w, n);

  const int a0=s, aLen=m-s;
  const int b0=m, bLen=e-m;
  const int outLen=aLen+bLen;

  const int TPB = blockDim.x;
  const int t   = threadIdx.x;
  const int seg = (outLen + TPB - 1)/TPB; 
  const int diagL = t*seg;
  const int diagH = min(outLen, (t+1)*seg);
  if (diagL >= diagH) return;

  int i = merge_path_cut(in, a0, aLen, in, b0, bLen, diagL);
  int j = diagL - i;
  const int iEnd = merge_path_cut(in, a0, aLen, in, b0, bLen, diagH);
  const int jEnd = diagH - iEnd;

  int outPos = s + diagL;
  while (i<iEnd && j<jEnd){
    uint32_t va=in[a0+i], vb=in[b0+j];
    if (va<=vb){ out[outPos++]=va; ++i; } else { out[outPos++]=vb; ++j; }
  }
  while (i<iEnd) { out[outPos++]=in[a0+(i++)]; }
  while (j<jEnd) { out[outPos++]=in[b0+(j++)]; }
}

static inline void sort_tiled(uint32_t* d, uint32_t* tmp, int n, int tpb, int& tile){
  constexpr int CHUNK = 1;
  {
    int grid = (n + CHUNK - 1) / CHUNK;
    small_bitonic<CHUNK><<<grid, CHUNK>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
  }

  uint32_t* src = d;
  uint32_t* dst = tmp;
  bool flipped = false;

  for (int w=CHUNK; w<n; w<<=1){
    const int pairs = (n + 2*w - 1) / (2*w);
    dim3 grid(pairs), block(tpb);

    switch (tile){
      case 16: merge_pass_simple<16><<<grid, block>>>(src, dst, n, w); break;
      case 32: merge_pass_simple<32><<<grid, block>>>(src, dst, n, w); break;
      case 64: merge_pass_simple<64><<<grid, block>>>(src, dst, n, w); break;
      default: merge_pass_simple<32><<<grid, block>>>(src, dst, n, w); break;
    }
    CUDA_CHECK(cudaGetLastError());

    std::swap(src, dst);
    flipped = !flipped;
  }

  if (flipped){
    CUDA_CHECK(cudaMemcpy(d, src, sizeof(uint32_t)*n, cudaMemcpyDeviceToDevice));
  }
}

static inline float run_kernel(uint32_t* d, uint32_t* tmp, int n, int tpb, int& tile){
  GPUTimer T; CUDA_CHECK(cudaDeviceSynchronize()); T.start();
  sort_tiled(d, tmp, n, tpb, tile);
  return T.stop();
}

int main(int argc, char** argv){
  if (argc < 2){
    std::cerr<<"Usage: merge_sort.exe <dataset.csv> [--verify] "
               "[--iters=N] [--tpb=256] [--tile=32] "
               "[--sorted=sorted.csv] [--bench=bench.csv]\n";
    return 1;
  }

  std::string path = argv[1];
  bool verify = false;
  int iters = 30;
  int tpb   = 256;
  int tile  = 32;
  std::string sorted_out = "sorted_out.csv";
  std::string bench_out  = "merge_bench.csv";

  for (int i=2;i<argc;++i){
    std::string s=argv[i];
    if (s=="--verify") verify=true;
    else if (s.rfind("--iters=",0)==0) iters=std::max(1, std::stoi(s.substr(8)));
    else if (s.rfind("--tpb=",0)==0)   tpb  =std::max(32, std::stoi(s.substr(6)));
    else if (s.rfind("--tile=",0)==0)  tile =std::max(8,  std::stoi(s.substr(7)));
    else if (s.rfind("--sorted=",0)==0) sorted_out=s.substr(9);
    else if (s.rfind("--bench=",0)==0)  bench_out =s.substr(8);
  }

  auto h_in = load_csv_u32(path);
  const int n = (int)h_in.size();
  std::vector<uint32_t> h_out(n);
  memtrack::reset();

  uint32_t *d_work=nullptr, *d_orig=nullptr, *d_tmp=nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(uint32_t)*n));
  CUDA_CHECK(cudaMalloc(&d_orig, sizeof(uint32_t)*n));
  CUDA_CHECK(cudaMalloc(&d_tmp,  sizeof(uint32_t)*n));

  CUDA_CHECK(cudaMemcpy(d_work, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_orig, h_in.data(), sizeof(uint32_t)*n, cudaMemcpyHostToDevice));

  std::vector<int> run_ids, iter_ids;
  std::vector<double> times_ms, peaks_mb;
  std::cout<<std::fixed<<std::setprecision(3);

  int tile_rt = tile;

  for (int it=0; it<iters; ++it){
    memtrack::peak.store(memtrack::live.load());

    CUDA_CHECK(cudaMemcpy(d_work, d_orig, sizeof(uint32_t)*n, cudaMemcpyDeviceToDevice));
    float ms = run_kernel(d_work, d_tmp, n, tpb, tile_rt);
    double peak_mb = memtrack::peak.load() / (1024.0*1024.0);

    run_ids.push_back(1);
    iter_ids.push_back(it+1);
    times_ms.push_back(ms);
    peaks_mb.push_back(peak_mb);

    std::cout<<"Iter "<<(it+1)<<": "<<ms<<" ms (peak_dev="<<peak_mb<<" MB)\n";
  }

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_work, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost));

  if (verify) {
    bool ok = check_sorted(h_out);
    std::cout<<"verify: "<<(ok?"OK":"FAIL")<<"\n";
  }

  write_csv_u32(sorted_out, h_out);
  std::cout<<"sorted file written: "<<sorted_out<<"\n";

  write_bench_csv(bench_out, run_ids, iter_ids, times_ms, peaks_mb);
  std::cout<<"bench file written: "<<bench_out<<"\n";

  const double mean = avg(times_ms), sd = stdev(times_ms, mean);
  std::cout<<"\n[Kernel Timing]\n  iters="<<iters<<"\n  mean_ms="<<mean<<"\n  stdev_ms="<<sd<<"\n";

  std::cout<<"\n[Memory Tracking]\n"
           <<"  device_live_bytes="<<memtrack::live.load()<<"\n"
           <<"  device_peak_bytes="<<memtrack::peak.load()<<"\n"
           <<"  allocs="<<memtrack::allocs.load()<<", frees="<<memtrack::frees.load()<<"\n";

  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFree(d_orig));
  CUDA_CHECK(cudaFree(d_tmp));
  return 0;
}
