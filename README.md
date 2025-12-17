# Performance-evaluation-of-GPU-based-parallel-sorting-algorithms (Project Management) 

### Overview

### •	Short product overview 

This project accompanies an academic paper that evaluates the performance of sequential CPU based sorting algorithms and their CUDA based parallel GPU implementations. The study focuses on comparing four commonly used sorting algorithms namely Merge Sort, Quick Sort, Bubble Sort, and Radix Sort under different data distributions including sorted, nearly sorted, random, and reverse sorted inputs.
The implementation provides a benchmarking framework that executes both CPU and GPU versions of each algorithm on large scale datasets, measures execution time, and analyzes performance differences and speedups achieved through GPU parallelization.

### •	The main problem and the proposed solution 

As dataset sizes continue to grow, traditional sequential sorting algorithms executed on CPUs often become performance bottlenecks due to limited parallelism and memory bandwidth constraints. While GPUs offer massive parallel processing capabilities, not all sorting algorithms benefit equally from GPU acceleration. The lack of a structured and reproducible comparison makes it difficult to evaluate when and why GPU based parallel sorting outperforms sequential CPU approaches. 

To address this problem, the paper employs a unified benchmarking framework to evaluate both sequential CPU implementations and CUDA based parallel GPU implementations of several sorting algorithms under identical experimental conditions. Each algorithm is executed on large datasets with different input characteristics, and execution time and speedup are measured to enable a fair and systematic comparison between CPU and GPU performance.

The use of CUDA makes it possible to exploit the parallel processing capabilities of GPUs and their memory architecture, while a consistent measurement methodology ensures that the results are reliable and comparable. This evaluation highlights the performance trade offs of each algorithm and demonstrates the scenarios in which GPU acceleration provides clear advantages over sequential execution. 

### •	Target users

This project is primarily intended for students, researchers, and practitioners who are interested in high-performance computing, parallel algorithms, and GPU programming with CUDA. It is particularly useful for:

Computer science and engineering students studying parallel computing, algorithms, or GPU programming, who want a concrete example of how classical algorithms behave when ported from CPU to GPU.

Researchers and academic users who require a reproducible baseline for evaluating and comparing sorting algorithms under different data distributions and execution models.

Performance engineers and developers seeking practical insights into the advantages and limitations of GPU acceleration for data-intensive workloads.
Overall, the project serves as both an educational reference and an experimental benchmark for understanding how algorithm design and hardware architecture influence performance. 


### Tech Stack

#### Programming Languages
- C++ for sequential CPU implementations
- CUDA C++ for parallel GPU implementations

#### GPU Computing
- NVIDIA CUDA Toolkit with the nvcc compiler
- NVIDIA GPU architecture for parallel execution

#### Development Tools
- Visual Studio Community 2022
- MSVC v143 C++ compiler for x64
- CUDA integration for Visual Studio

#### Benchmarking and Measurement
- CPU execution timing using standard C++ timing utilities
- GPU execution timing using CUDA event based measurements



### •	Steps to run the project locally  
 Step 1: Install Visual Studio and required components

First, install Visual Studio Community 2022 (purple icon). This IDE provides the MSVC C++ compiler that is required by the CUDA compiler on Windows. CUDA uses MSVC as the host compiler, so Visual Studio is mandatory even if code editing is done in another editor.

During installation, enable Desktop development with C++ and ensure that MSVC v143 and the Windows SDK are selected. 

Step 2: Open the x64 Developer Command Prompt

After installation, open x64 Native Tools Command Prompt for Visual Studio 2022.
This command prompt initializes the correct 64 bit compilation environment required by CUDA and MSVC. 
<img width="1103" height="644" alt="image" src="https://github.com/user-attachments/assets/5a62b862-f01e-4bb9-b22f-e16d749ca3f2" />


You should see a message indicating that the environment is initialized for x64. 

Step 3: Verify the environment

Inside the x64 Native Tools Command Prompt, verify that all required tools are available.

• **Verify the C++ Compiler**  
  `cl`

• **Verify NVIDIA GPU and Driver**  
  `nvidia-smi`

• **Verify CUDA Toolkit Version**  
  `nvcc --version`
  
<img width="657" height="128" alt="image" src="https://github.com/user-attachments/assets/23f4ebf9-6be3-42d5-87ad-25f7e9d49541" /> 

<img width="824" height="250" alt="image" src="https://github.com/user-attachments/assets/cceec59f-3aad-4243-b704-28ff5c251b7d" />


These commands confirm that the compiler, GPU, and CUDA Toolkit are correctly installed and ready for execution. 

Step 4: Navigate to the Project Directory

After opening the x64 Native Tools Command Prompt, move to the directory where the project is located.

• Navigate to the project folder  
  `cd C:\Users\ your path`

 <img width="1099" height="640" alt="image" src="https://github.com/user-attachments/assets/c72d371d-5930-4dad-8d41-e715c81ba60d" />

The directory should contain the CUDA source files, compiled executables, and data folders required for running the benchmarks. 

Step 5: Open the Project in VS Code

From the same command prompt, open the project folder in Visual Studio Code.

• Open the project in VS Code

`code .`

VS Code is used as a lightweight editor to inspect and modify source files. Compilation and execution are handled through the x64 Native Tools Command Prompt. 

Step 6: Configure Input and Output Paths

Each sorting algorithm defines default paths for input data, sorted output, and benchmark results within the source code.

For example, in the Radix Sort implementation:

• Default configuration values

`MEASURE_DEFAULT = "kernel"
INPUT_DEFAULT = "C:/Users/user/Desktop/CPU/data/nearly_sorted_10m.csv"
SORTED_DEFAULT = "C:/Users/user/Desktop/CPU/data/outputs/output_radix_gpu.csv"
BENCH_DEFAULT = "C:/Users/user/Desktop/CPU/data/benchmark_results/benchmark_radix_gpu.csv"`

Update these paths if necessary to match your local directory structure. The same configuration approach is used in the Merge Sort, Quick Sort, and Bubble Sort implementations. 

Step 7: Run the Algorithm

From the x64 Native Tools Command Prompt, execute the compiled binary.

• Example Radix Sort execution
`.\radix_sort.exe`

<img width="790" height="565" alt="image" src="https://github.com/user-attachments/assets/a96a174a-4acc-481f-9e5c-0f218c1bb85c" />


The program loads the input dataset, executes the GPU kernel multiple times, measures execution time, and reports memory usage and performance statistics. The execution process is identical for all algorithms, with only the executable name changing. 

Step 8: Check Results and Outputs

After execution completes, the following outputs are generated.

• The sorted output file is saved to the output path specified in the source code

• The benchmark summary is saved as a CSV file in the benchmark results directory

<img width="817" height="185" alt="image" src="https://github.com/user-attachments/assets/f7e69237-bd14-4c7c-ad85-7107fb1da305" />

To verify correctness, navigate to the output directory and inspect the generated CSV file. The file should contain correctly sorted values. 

Step 9: Apply the Same Process to All Algorithms

All sorting algorithms in this project follow the same execution workflow.

 - Open the x64 Native Tools Command Prompt
-  Verify the environment
-  Navigate to the project directory
 - Open the project in VS Code
 - Adjust input and output paths if needed
 -  Run the executable
  - Validate the sorted output and benchmark results 


### •	project structure with folder descriptions
The project consists of two main components: the benchmarking code repository and the Overleaf paper files.

Benchmarking Repository

The benchmarking repository contains the implementation of sequential CPU and parallel GPU sorting algorithms, along with input data and generated results.

• benchmarking/
Root directory of the benchmarking framework

• benchmarking/CPU/
Sequential CPU based sorting algorithm implementations

• benchmarking/GPU/
CUDA based parallel GPU sorting algorithm implementations

Overleaf Paper Files

The Overleaf directory contains the academic paper and supporting materials. The structure is straightforward and reflected directly in the file tree.

• Figures/
Figures and plots used in the paper

• CUDA.tex
Main LaTeX source file of the paper

• Ref.bib
Bibliography file

• Results.xlsx
Experimental results and aggregated performance data

• Coverletter.docx
Journal cover letter

• Response to Reviewers.docx
Reviewer response documents

• plos2015.bst
The design requirements of the paper from the journal

<img width="445" height="416" alt="image" src="https://github.com/user-attachments/assets/18a6d957-eb2e-4da1-a9b3-27e417a41e8b" />
