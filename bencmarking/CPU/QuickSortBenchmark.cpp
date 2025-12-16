

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

static atomic<long long> g_live{0};
static atomic<long long> g_peak{0};

namespace {
struct AllocHeader { size_t sz; };

inline void* alloc_and_track(size_t sz) {
    void* raw = std::malloc(sizeof(AllocHeader) + sz);
    if (!raw) throw std::bad_alloc();
    static_cast<AllocHeader*>(raw)->sz = sz;
    void* p = static_cast<char*>(raw) + sizeof(AllocHeader);

    long long now = (g_live += (long long)sz);
    long long prev = g_peak.load(std::memory_order_relaxed);
    while (now > prev && !g_peak.compare_exchange_weak(
               prev, now, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    return p;
}
inline void free_and_track(void* p) noexcept {
    if (!p) return;
    void* raw = static_cast<char*>(p) - sizeof(AllocHeader);
    size_t sz = static_cast<AllocHeader*>(raw)->sz;
    g_live -= (long long)sz;
    std::free(raw);
}
} 

void* operator new(std::size_t sz)             { return alloc_and_track(sz); }
void* operator new[](std::size_t sz)           { return alloc_and_track(sz); }
void  operator delete(void* p) noexcept        { free_and_track(p); }
void  operator delete[](void* p) noexcept      { free_and_track(p); }
void  operator delete(void* p, std::size_t) noexcept   { free_and_track(p); }
void  operator delete[](void* p, std::size_t) noexcept { free_and_track(p); }

static inline void reset_peak()         { g_live.store(0); g_peak.store(0); }
static inline long long peak_bytes()    { return g_peak.load(); }


static const int RUNS = 30;

const string INPUT_FILE     = "C:/Users/user/Desktop/CPU/data/sorted_10e6.csv";
const string SORTED_FILE    = "C:/Users/user/Desktop/CPU/data/outputs/output_quicksort_sorted.csv";
const string BENCHMARK_FILE = "C:/Users/user/Desktop/CPU/data/benchmark_results/benchmark_quicksort.csv";


static vector<int> readCSV(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Failed to open input file: " + path);
    vector<int> a; a.reserve(1<<20);
    string line;
    while (getline(in, line)) {
        if (!line.empty()) a.push_back(stoi(line));
    }
    return a;
}
static void writeCSV(const string& path, const vector<int>& v) {
    ofstream out(path);
    for (int x : v) out << x << "\n";
}
static void writeBenchmarkCSV(const string& path,
                              const vector<long long>& dur_ms,
                              const vector<long long>& mem_bytes) {
    ofstream out(path);
    out << "Run,Duration(ms),Memory(MB)\n";
    out << fixed << setprecision(2);
    for (size_t i = 0; i < dur_ms.size(); ++i)
        out << (i+1) << "," << dur_ms[i] << "," << (double(mem_bytes[i]) / 1e6) << "\n";
}
static double avg(const vector<long long>& v){
    if (v.empty()) return 0.0;
    long double s=0; for (auto x:v) s+=x;
    return (double)(s/v.size());
}
static double stdev(const vector<long long>& v, double mean){
    if (v.empty()) return 0.0;
    long double s=0; for (auto x:v){ long double d=x-mean; s+=d*d; }
    return (double)std::sqrt(s/v.size());
}


int partitionLomuto(vector<int>& a, int low, int high) {
    int pivotIndex = low + rand() % (high - low + 1);
    swap(a[pivotIndex], a[high]);

    int pivot = a[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (a[j] <= pivot) { ++i; swap(a[i], a[j]); }
    }
    swap(a[i + 1], a[high]);   
    return i + 1;
}


void quickSortNaive(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partitionLomuto(arr, low, high);
        quickSortNaive(arr, low, pi - 1);
        quickSortNaive(arr, pi + 1, high);
    }
}

void quickSortNaive(vector<int>& arr) {
    if (!arr.empty())
        quickSortNaive(arr, 0, (int)arr.size() - 1);
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> original = readCSV(INPUT_FILE);
    cout << " Loaded " << original.size() << " integers from " << INPUT_FILE << "\n";

    vector<long long> durations_ms(RUNS);
    vector<long long> peak_mem(RUNS);

    for (int run = 0; run < RUNS; ++run) {
        reset_peak();
        vector<int> arr = original;

        auto t0 = high_resolution_clock::now();
        quickSortNaive(arr);
        auto t1 = high_resolution_clock::now();

        long long ms = duration_cast<milliseconds>(t1 - t0).count();
        durations_ms[run] = ms;
        peak_mem[run] = peak_bytes();

        cout << "Run " << setw(2) << (run+1) << ": " << ms
             << " ms | Peak memory: " << fixed << setprecision(2)
             << (double(peak_mem[run]) / 1e6) << " MB\n";
    }

    double aT   = avg(durations_ms);
    double sT   = stdev(durations_ms, aT);
    double sPct = (aT == 0.0 ? 0.0 : (sT / aT) * 100.0);
    long long mnT = *min_element(durations_ms.begin(), durations_ms.end());
    long long mxT = *max_element(durations_ms.begin(), durations_ms.end());

    double aM  = avg(peak_mem) / 1e6;
    double mnM = (*min_element(peak_mem.begin(), peak_mem.end())) / 1e6;
    double mxM = (*max_element(peak_mem.begin(), peak_mem.end())) / 1e6;

    string interp = (sPct < 5.0) ? " Very stable performance"
                    : (sPct <= 15.0) ? " Fairly stable performance"
                    : " Some variation in performance";

    cout << "\n Benchmark Summary (" << RUNS << " runs)\n";
    cout << "  Average time: " << fixed << setprecision(2) << aT << " ms\n";
    cout << "  Std deviation: " << sT << " ms (" << sPct << "%)\n";
    cout << "  Min: " << mnT << " ms | Max: " << mxT << " ms\n";
    cout << "  Interpretation: " << interp << "\n";
    cout << " Peak memory (avg): " << aM
         << " MB | min: " << mnM << " MB | max: " << mxM << " MB\n";

    writeBenchmarkCSV(BENCHMARK_FILE, durations_ms, peak_mem);
    cout << " Benchmark report saved: " << BENCHMARK_FILE << "\n";

    vector<int> sorted = original;
    quickSortNaive(sorted);
    writeCSV(SORTED_FILE, sorted);
    cout << " Sorted file saved: " << SORTED_FILE << "\n";

    return 0;
}
