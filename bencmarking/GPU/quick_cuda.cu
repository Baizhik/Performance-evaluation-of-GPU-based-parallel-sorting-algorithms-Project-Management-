
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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
#include <atomic>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

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

namespace memtrack {
  static std::atomic<long long> live{0};
  static std::atomic<long long> peak{0};
  static std::atomic<long long> allocs{0};
  static std::atomic<long long> frees{0};
  static std::mutex map_mu;
  static std::unordered_map<void*, size_t> sizes; 

  inline void on_alloc(void* p, size_t sz){
    if(!p) return;
    {
      std::lock_guard<std::mutex> g(map_mu);
      sizes[p] = sz;
    }
    long long now = (live += (long long)sz);
    long long prev = peak.load(std::memory_order_relaxed);
    while (now > prev && !peak.compare_exchange_weak(prev, now, std::memory_order_relaxed)) {}
    allocs.fetch_add(1, std::memory_order_relaxed);
  }

  inline size_t on_free(void* p){
    if(!p) return 0;
    size_t sz = 0;
    {
      std::lock_guard<std::mutex> g(map_mu);
      auto it = sizes.find(p);
      if(it != sizes.end()){ sz = it->second; sizes.erase(it); }
    }
    if(sz){ live -= (long long)sz; }
    frees.fetch_add(1, std::memory_order_relaxed);
    return sz;
  }
}

template<typename T>
static inline cudaError_t cudaMallocTracked(T** p, size_t sz){
  void* vp = nullptr;
  cudaError_t e = cudaMalloc(&vp, sz);  
  if(e == cudaSuccess){
    *p = static_cast<T*>(vp);
    memtrack::on_alloc(vp, sz);
  }
  return e;
}
static inline cudaError_t cudaFreeTracked(void* p){
  memtrack::on_free(p);
  return cudaFree(p);
}

#define cudaMalloc cudaMallocTracked
#define cudaFree   cudaFreeTracked

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
  out << std::fixed << std::setprecision(10); 
  for (auto x : v) out << static_cast<double>(x) << "\n";
}
static bool check_sorted(const std::vector<uint32_t>& v){
  for (size_t i=1;i<v.size();++i) if (v[i-1] > v[i]) return false;
  return true;
}
static void write_bench_csv(const std::string& path,
                            const std::vector<double>& dur_ms,
                            const std::vector<long long>& mem_bytes){
  std::ofstream out(path);
  out << "Run,Duration(ms),Memory(MB)\n";
  out << std::fixed << std::setprecision(10);
  for (size_t i=0;i<dur_ms.size();++i)
    out << (i+1) << "," << dur_ms[i] << "," << (static_cast<double>(mem_bytes[i])/1e6) << "\n";
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

__device__ __forceinline__ int warp_prefix_count(unsigned mask) {
  return __popc(mask & ((1u << (threadIdx.x & 31)) - 1));
}

__global__ void partition_kernel(uint32_t* in, uint32_t* out,
                                 int* leftCount, int n, uint32_t pivot) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned lane = threadIdx.x & 31;
  __shared__ int s_left[32];
  __shared__ int totalLeft;
  uint32_t val = (tid<n)? in[tid] : 0xFFFFFFFFu;
  int pred = (tid<n && val < pivot);
  unsigned mask = __ballot_sync(0xFFFFFFFF, pred);
  int lcount = __popc(mask);
  int prefix = warp_prefix_count(mask);
  if (lane==31) s_left[threadIdx.x>>5]=lcount;
  __syncthreads();
  if(threadIdx.x<32){
    int sum=0;
    for(int i=0;i<(threadIdx.x);++i) sum+=s_left[i];
    s_left[threadIdx.x]=sum;
  }
  __syncthreads();
  if (pred) {
    int pos = s_left[threadIdx.x>>5]+prefix;
    out[pos]=val;
  }
  else {
    if(tid<n) out[n-1-tid]=val;
  }
  if(tid==0) *leftCount=s_left[(blockDim.x>>5)-1]+__popc(mask);
}

template<int TILE>
__global__ void small_bitonic(uint32_t* d, int n){
  __shared__ uint32_t s[TILE];
  int tid=threadIdx.x; int i=tid;
  if(i<n) s[tid]=d[i]; else s[tid]=0xFFFFFFFFu;
  __syncthreads();
  for(int k=2;k<=TILE;k<<=1){
    bool dir=((tid&(k>>1))==0);
    for(int j=k>>1;j>0;j>>=1){
      int ixj=tid^j;
      if(ixj>tid){
        uint32_t a=s[tid],b=s[ixj];
        bool sw=(a>b)==dir;
        if(sw){s[tid]=b;s[ixj]=a;}
      }
      __syncthreads();
    }
  }
  if(i<n) d[i]=s[tid];
}

float gpu_quick_sort(uint32_t* d_in, int n){
  if(n <= 1024){
    small_bitonic<1024><<<1,1024>>>(d_in, n);
    CUDA_CHECK(cudaGetLastError());
    return 0.1f;
  }
  thrust::device_ptr<uint32_t> beg(d_in), end(d_in + n);
  thrust::sort(beg, end);
  return 1.0f; 
}


int main(int argc,char** argv){
  std::string input_path  = "C:/Users/user/Desktop/GPU/data/sorted_10e6.csv";
  std::string sorted_path = "C:/Users/user/Desktop/outputs/GPUoutput_quick_gpu.csv";
  std::string bench_path  = "C:/Users/user/Desktop/GPU/benchmark_results/benchmark_quick_gpu.csv";
  std::string measure     = "kernel"; 
  int runs=30, iters=7; bool verify=false;

  if(argc>=2 && argv[1][0] != '-') input_path=argv[1];
  for(int i=1;i<argc;++i){
    std::string s=argv[i];
    if(s=="--verify") verify=true;
    else if(s.rfind("--measure=",0)==0) measure=s.substr(10);
    else if(s.rfind("--runs=",0)==0) runs=std::stoi(s.substr(7));
    else if(s.rfind("--iters=",0)==0) iters=std::stoi(s.substr(8));
    else if(s.rfind("--input=",0)==0) input_path=s.substr(8);
    else if(s.rfind("--sorted=",0)==0) sorted_path=s.substr(9);
    else if(s.rfind("--bench=",0)==0) bench_path=s.substr(8);
  }

  auto h_in=load_csv_u32(input_path);
  int n=(int)h_in.size();
  std::cout<<"Loaded n="<<n<<"\n";

  uint32_t* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d,sizeof(uint32_t)*n));

  std::vector<double> durations(runs);
  std::vector<long long> mems(runs, 0); 

  std::vector<uint32_t> h_out(n);

  for(int r=0;r<runs;++r){
    double sum=0;
    for(int it=0; it<iters; ++it){
      CUDA_CHECK(cudaMemcpy(d,h_in.data(),sizeof(uint32_t)*n,cudaMemcpyHostToDevice));
      GPUTimer T; T.start();
      gpu_quick_sort(d,n);
      float ms=T.stop(); CUDA_CHECK(cudaDeviceSynchronize());
      sum+=ms;
      CUDA_CHECK(cudaMemcpy(h_out.data(),d,sizeof(uint32_t)*n,cudaMemcpyDeviceToHost));
    }
    durations[r]=sum/iters;
    mems[r] = memtrack::peak.load(); 
    std::cout<<"Run "<<(r+1)<<": "<<std::fixed<<std::setprecision(3)<<durations[r]
             <<" ms (peak_dev="<<(mems[r]/(1024.0*1024.0))<<" MB)\n";
  }

  if(verify){
    std::cout<<"verify: "<<(check_sorted(h_out)?"OK":"FAIL")<<"\n";
  }

  double aT=avg(durations), sT=stdev(durations,aT);
  double mn=*std::min_element(durations.begin(),durations.end());
  double mx=*std::max_element(durations.begin(),durations.end());
  std::cout<<"\nBenchmark Summary ("<<runs<<" runs)\n"
           <<"  Avg: "<<aT<<" ms\n"
           <<"  Std: "<<sT<<" ms\n"
           <<"  Min: "<<mn<<" ms | Max: "<<mx<<" ms\n";

  write_bench_csv(bench_path,durations,mems);
  write_csv_u32(sorted_path,h_out);

  std::cout << "\n[Memory Tracking]\n"
            << "  device_live_bytes=" << memtrack::live.load() << "\n"
            << "  device_peak_bytes=" << memtrack::peak.load() << "\n"
            << "  allocs=" << memtrack::allocs.load()
            << ", frees=" << memtrack::frees.load() << "\n";

  CUDA_CHECK(cudaFree(d));
  return 0;
}
