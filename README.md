#HPC

# High-Performance Computing (HPC) - Complete MCQ Preparation Guide

## Table of Contents
1. [MCQ Questions by Topic](#mcq-questions-by-topic)
2. [Answer Key with Explanations](#answer-key-with-explanations)
3. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## MCQ Questions by Topic

### **Section 1: Significance and Application of HPC**

**Q1.** Which of the following is NOT a primary application of High-Performance Computing?
- A) Climate modeling and weather forecasting
- B) Drug discovery and molecular simulations
- C) Word processing and spreadsheet calculations
- D) Nuclear physics simulations

**Q2.** What is the primary reason HPC systems are essential for modern scientific research?
- A) They consume less power than traditional computers
- B) They can solve complex modeling problems that exceed the capacity of standard computers
- C) They are cheaper to operate than desktop systems
- D) They require less programming expertise

**Q3.** In the context of HPC, what does "scalability" primarily refer to?
- A) The physical size of the computer system
- B) The ability to efficiently utilize increasing hardware resources
- C) The cost reduction when purchasing multiple systems
- D) The number of programming languages supported

**Q4.** Which industry benefits from HPC by performing real-time financial market analysis and risk assessment?
- A) Healthcare
- B) Agriculture
- C) Financial services
- D) Retail

**Q5.** The Frontier supercomputer achieves approximately how many floating-point operations per second?
- A) 1 teraflop
- B) 1 petaflop
- C) 1 exaflop
- D) 1 gigaflop

---

### **Section 2: Multicore CPU vs GPU**

**Q6.** What is the fundamental architectural difference between CPUs and GPUs?
- A) CPUs have more cores than GPUs
- B) CPUs are optimized for sequential processing; GPUs for parallel processing
- C) GPUs cannot perform general-purpose computing
- D) CPUs use more power than GPUs

**Q7.** Which of the following best describes GPU cores compared to CPU cores?
- A) GPU cores are more powerful but fewer in number
- B) GPU cores are less powerful individually but vastly more numerous
- C) GPU cores and CPU cores have identical capabilities
- D) GPU cores can only handle graphics operations

**Q8.** What type of memory hierarchy characteristic distinguishes CPUs from GPUs?
- A) CPUs have no cache memory
- B) CPUs rely heavily on large cache layers; GPUs tolerate higher memory latency
- C) GPUs have larger L1 cache than CPUs
- D) Both have identical memory hierarchies

**Q9.** For which type of workload are GPUs MOST efficient?
- A) Sequential database transactions
- B) Operating system kernel operations
- C) Massively parallel matrix operations
- D) Single-threaded conditional logic

**Q10.** In NVIDIA GPUs, what are the specialized cores called?
- A) Tensor cores
- B) CUDA cores
- C) Vector cores
- D) Scalar cores

**Q11 (Tricky).** A CPU with 16 cores running at 4 GHz and a GPU with 4096 cores running at 1.5 GHz are compared. Which statement is TRUE?
- A) The CPU will always be faster due to higher clock speed
- B) The GPU will excel at tasks with high data parallelism
- C) The CPU has more total processing power
- D) They will perform identically on all workloads

---

### **Section 3: SIMD vs MIMD**

**Q12.** What does SIMD stand for?
- A) Single Instruction Multiple Devices
- B) Single Instruction Multiple Data
- C) Sequential Instruction Multiple Data
- D) Synchronized Instruction Memory Data

**Q13.** In SIMD architecture, how many instruction decoders are typically present?
- A) One per data element
- B) Multiple decoders for redundancy
- C) A single decoder
- D) No decoders are needed

**Q14.** Which of the following is a characteristic of MIMD systems?
- A) All processors execute the same instruction simultaneously
- B) Asynchronous programming with explicit synchronization
- C) Limited to image processing applications
- D) Cannot handle different data sets

**Q15.** Which architecture is more suitable for general-purpose computing with diverse tasks?
- A) SIMD only
- B) MIMD only
- C) Both are equally suitable
- D) Neither can handle general-purpose computing

**Q16.** In terms of cost, how do SIMD and MIMD systems compare?
- A) SIMD is more expensive than MIMD
- B) MIMD is more expensive than SIMD
- C) Both have identical costs
- D) Cost depends only on the number of processors

**Q17 (Tricky).** An application needs to apply the same image filter to millions of pixels simultaneously. Which architecture is MOST appropriate?
- A) MIMD, because it's more efficient
- B) SIMD, because it excels at uniform operations on large datasets
- C) Either architecture would be equally efficient
- D) Neither architecture is suitable for image processing

**Q18.** What type of synchronization does SIMD use?
- A) Explicit synchronization
- B) Asynchronous synchronization
- C) Implicit (tacit) synchronization
- D) No synchronization is needed

---

### **Section 4: Data Parallelism vs Task Parallelism**

**Q19.** What characterizes data parallelism?
- A) Different tasks performed on the same data
- B) Same task performed on different subsets of data
- C) Sequential execution of multiple tasks
- D) Random data access patterns

**Q20.** In task parallelism, what determines the amount of parallelization?
- A) The size of the input data
- B) The number of independent tasks to be performed
- C) The clock speed of the processor
- D) The amount of available memory

**Q21.** Which type of computation is performed in data parallelism?
- A) Asynchronous
- B) Synchronous
- C) Random
- D) Sequential only

**Q22.** Consider computing the sum, average, and maximum of an array using different processors. This is an example of:
- A) Data parallelism
- B) Task parallelism
- C) Sequential processing
- D) Vector processing

**Q23 (Tricky).** A program divides a large array into 8 chunks and computes the same mathematical formula on each chunk using 8 threads. Which parallelism model is this?
- A) Task parallelism, because multiple threads are used
- B) Data parallelism, because the same operation is applied to different data subsets
- C) Both data and task parallelism equally
- D) Neither, this is sequential processing

**Q24.** What is the primary advantage of data parallelism over task parallelism regarding scalability?
- A) Data parallelism scales with input size; task parallelism scales with number of distinct tasks
- B) Task parallelism always scales better
- C) Both scale identically
- D) Neither provides scalability benefits

---

### **Section 5: Introduction to Heterogeneous Computing**

**Q25.** What defines a heterogeneous computing system?
- A) Uses only CPUs of different manufacturers
- B) Uses multiple types of processors (CPUs, GPUs, FPGAs, etc.)
- C) Uses processors from multiple vendors
- D) Uses only identical processor types

**Q26.** What is the primary advantage of heterogeneous computing?
- A) Lower cost than homogeneous systems
- B) Easier programming model
- C) Assigns workloads to the most suitable processor type
- D) Requires less power

**Q27.** In heterogeneous computing, what is the CPU typically called?
- A) Device
- B) Accelerator
- C) Host
- D) Kernel

**Q28.** Which component is referred to as the "device" in GPU-accelerated heterogeneous computing?
- A) CPU
- B) GPU
- C) Main memory
- D) Hard disk

**Q29.** What is the primary communication pathway between host and device in heterogeneous systems?
- A) Ethernet cable
- B) PCIe bus
- C) SATA connection
- D) USB interface

**Q30 (Tricky).** A heterogeneous system with CPU and GPU processes data. The CPU prepares and organizes data while the GPU performs matrix multiplication. Which statement is FALSE?
- A) The GPU directly accesses the CPU's main memory address space
- B) Data must be transferred between host and device memory
- C) The CPU coordinates the overall workflow
- D) The GPU performs the compute-intensive portion

---

### **Section 6: Portability and Scalability in Heterogeneous Parallel Computing**

**Q31.** What does "portability" mean in the context of parallel computing?
- A) The physical ability to move hardware between locations
- B) The ability to run applications efficiently on different hardware architectures
- C) The capability to use external storage devices
- D) The feature to compile code without modifications

**Q32.** Why are scalability and portability important for software cost control?
- A) They reduce hardware costs
- B) They minimize software redevelopment when upgrading systems
- C) They eliminate the need for testing
- D) They make programs run faster automatically

**Q33.** Which programming approach supports scalability through fine-grained problem decomposition?
- A) Monolithic kernel design
- B) Static thread allocation
- C) Dynamic thread scheduling
- D) Single-threaded execution

**Q34.** Which emerging standard helps address portability across different processor types?
- A) FORTRAN
- B) Assembly language
- C) OpenCL
- D) BASIC

**Q35 (Tricky).** An application runs efficiently on a 4-core CPU and scales well to 16 cores. When moved to a GPU with 1024 cores, performance does not improve. What is the likely issue?
- A) The application achieves scalability but lacks portability
- B) The application has good portability but poor scalability
- C) The application has neither scalability nor portability
- D) This is expected behavior for all applications

---

### **Section 7: CUDA C - Threads, Kernels, Toolkit, Unified Memory**

**Q36.** What keyword is used to declare a CUDA kernel function?
- A) `__device__`
- B) `__global__`
- C) `__host__`
- D) `__kernel__`

**Q37.** How is a CUDA kernel launched from host code?
- A) Using standard function call syntax: `kernel(args)`
- B) Using triple angle brackets: `kernel<<<grid, block>>>(args)`
- C) Using square brackets: `kernel[grid][block](args)`
- D) Using the `launch()` function

**Q38.** What is the purpose of CUDA Unified Memory?
- A) To increase GPU memory size
- B) To provide a single memory space accessible by both CPU and GPU
- C) To speed up CPU processing
- D) To eliminate the need for memory allocation

**Q39.** Which CUDA API function allocates unified memory?
- A) `cudaMalloc()`
- B) `cudaMemcpy()`
- C) `cudaMallocManaged()`
- D) `malloc()`

**Q40.** What happens when a CUDA kernel is launched?
- A) It executes synchronously and blocks the CPU
- B) It executes asynchronously and returns immediately
- C) It runs on the CPU
- D) It requires manual thread creation

**Q41.** To free memory allocated with `cudaMallocManaged()`, which function should be used?
- A) `free()`
- B) `delete()`
- C) `cudaFree()`
- D) `cudaFreeManaged()`

**Q42 (Tricky).** A kernel is launched with `kernel<<<256, 128>>>()`. How many total threads are launched?
- A) 256
- B) 128
- C) 384
- D) 32,768

---

### **Section 8: CUDA Parallelism Model - Thread-Warp-Block-Grid Hierarchy**

**Q43.** In the CUDA thread hierarchy, what is the smallest unit of execution?
- A) Grid
- B) Block
- C) Warp
- D) Thread

**Q44.** How many threads are in a standard CUDA warp?
- A) 8
- B) 16
- C) 32
- D) 64

**Q45.** What is the maximum number of threads allowed per block in CUDA?
- A) 256
- B) 512
- C) 1024
- D) 2048

**Q46.** Which built-in variable provides the thread's unique ID within a block?
- A) `blockIdx`
- B) `threadIdx`
- C) `blockDim`
- D) `gridDim`

**Q47.** Threads within the same block share which resource?
- A) Global memory only
- B) Registers
- C) Shared memory
- D) CPU cache

**Q48.** What is the highest level in the CUDA thread hierarchy?
- A) Thread
- B) Warp
- C) Block
- D) Grid

**Q49.** Blocks within a grid can communicate through:
- A) Shared memory
- B) Global memory
- C) Local memory
- D) Constant memory

**Q50 (Tricky).** If a block has dimensions `dim3(16, 16, 1)`, how many warps does it contain?
- A) 4
- B) 8
- C) 16
- D) 32

---

### **Section 9: Kernel-based SPMD Parallel Programming**

**Q51.** What does SPMD stand for?
- A) Single Program Multiple Devices
- B) Single Program Multiple Data
- C) Sequential Program Multiple Data
- D) Synchronized Parallel Multiple Data

**Q52.** In SPMD programming, what do all processing elements (threads) do?
- A) Execute different programs on different data
- B) Execute the same program on the same data
- C) Execute the same program in parallel, each with its own data
- D) Execute sequentially one after another

**Q53.** How does each thread in a CUDA SPMD kernel identify which data to process?
- A) Random selection
- B) Through its unique thread ID (threadIdx, blockIdx)
- C) The host assigns data explicitly
- D) First-come, first-served basis

**Q54.** What makes CUDA's execution model SPMD?
- A) Multiple kernels run simultaneously
- B) All threads execute the same kernel code but on different data elements
- C) Threads execute different kernels
- D) Only one thread executes at a time

**Q55 (Tricky).** In a vector addition kernel, Thread 5 computes `C[5] = A[5] + B[5]` while Thread 6 computes `C[6] = A[6] + B[6]`. This exemplifies:
- A) MIMD, because different data is being processed
- B) SPMD, because the same operation is performed by all threads on their respective data
- C) SIMD, because it's synchronous
- D) None of the above

---

### **Section 10: Mapping and Color to Grayscale Conversion**

**Q56.** In a 2D image processing kernel, how is the global row index typically calculated?
- A) `row = threadIdx.y`
- B) `row = blockIdx.y * blockDim.y + threadIdx.y`
- C) `row = blockIdx.x + threadIdx.x`
- D) `row = gridDim.y * threadIdx.y`

**Q57.** What is the standard formula for RGB to grayscale conversion?
- A) `Gray = R + G + B`
- B) `Gray = (R + G + B) / 3`
- C) `Gray = 0.21*R + 0.71*G + 0.07*B` (or similar weighted average)
- D) `Gray = max(R, G, B)`

**Q58.** For a 1024x768 image with block size (32, 32), how many blocks are needed in the x-dimension?
- A) 32
- B) 33
- C) 1024
- D) 768

**Q59.** Why is boundary checking important in image processing kernels?
- A) To prevent division by zero
- B) To ensure threads don't access memory outside the image bounds
- C) To optimize performance
- D) To enable shared memory usage

**Q60 (Tricky).** A grayscale kernel with block size (16, 16) processes a 100x100 image. Some threads will be idle because:
- A) 100 is not divisible by 16
- B) Too many threads are launched
- C) The grid is not large enough
- D) All threads will be active with proper bounds checking

---

### **Section 11: Thread Scheduling and Warps**

**Q61.** What hardware component schedules warps for execution on a Streaming Multiprocessor?
- A) CPU scheduler
- B) Warp scheduler
- C) Thread scheduler
- D) Block scheduler

**Q62.** How frequently can the warp scheduler switch between warps?
- A) Every millisecond
- B) Every microsecond
- C) Every clock cycle (nanosecond scale)
- D) Every 100 clock cycles

**Q63.** What is the primary benefit of rapid warp switching?
- A) Lower power consumption
- B) Hiding memory access latency
- C) Reducing code complexity
- D) Increasing clock speed

**Q64.** Threads within a warp execute:
- A) Completely independently with no coordination
- B) The same instruction at the same time
- C) Different instructions simultaneously
- D) In strict sequential order

**Q65.** How does GPU context switching differ from CPU context switching?
- A) GPU switching takes longer
- B) GPU switching requires saving contexts to memory
- C) GPU switching is much faster because thread contexts are already in registers
- D) There is no difference

**Q66 (Tricky).** A warp has 32 threads, but only 10 threads need to execute a particular `if` block. What happens?
- A) Only those 10 threads execute; the others are idle but the warp still takes execution time
- B) The 22 threads are reassigned to other warps
- C) The block is recompiled
- D) All 32 threads execute the code

---

### **Section 12: CUDA Memory Types**

**Q67.** Which CUDA memory type has the largest capacity?
- A) Shared memory
- B) Constant memory
- C) Global memory
- D) Register memory

**Q68.** Which memory type has the fastest access speed?
- A) Global memory
- B) Local memory
- C) Shared memory
- D) Registers

**Q69.** What is the scope of shared memory?
- A) Single thread
- B) Thread block
- C) Entire grid
- D) Across multiple kernels

**Q70.** Constant memory is optimized for:
- A) Write operations
- B) Broadcast read operations
- C) Random access
- D) Thread-local storage

**Q71.** Where is local memory actually implemented?
- A) On-chip SRAM
- B) CPU RAM
- C) A portion of global memory
- D) Cache memory

**Q72.** What is the lifetime of automatic variables in a CUDA kernel?
- A) Entire application
- B) Across multiple kernel launches
- C) Single kernel execution
- D) Single thread block

**Q73 (Tricky).** A kernel declares `__shared__ int buffer[256];`. Each thread block has:
- A) A shared single array accessible to all blocks
- B) Its own private copy of the 256-element array
- C) Access to the CPU's memory
- D) 256 separate arrays

---

### **Section 13: Tiled Parallel Algorithms**

**Q74.** What is the primary goal of tiling in parallel algorithms?
- A) To increase global memory size
- B) To reduce global memory accesses by using shared memory
- C) To simplify kernel code
- D) To increase the number of threads

**Q75.** In tiled matrix multiplication, what is loaded into shared memory?
- A) The entire matrices
- B) Single elements
- C) Tiles (submatrices) of the input matrices
- D) Only the result matrix

**Q76.** What synchronization primitive is essential in tiled algorithms?
- A) `cudaDeviceSynchronize()`
- B) `__syncthreads()`
- C) `__syncwarp()`
- D) No synchronization is needed

**Q77.** Why is `__syncthreads()` necessary in tiled matrix multiplication?
- A) To wait for data to be loaded into shared memory before computation
- B) To synchronize with the CPU
- C) To clear shared memory
- D) To launch new threads

**Q78.** In tiled algorithms, computation is divided into:
- A) Random chunks
- B) Phases, with each phase processing one tile
- C) Single large operation
- D) Sequential steps

**Q79 (Tricky).** For a 1024x1024 matrix multiplication with tile size 16x16, how many tile loads are required per output element?
- A) 1
- B) 16
- C) 64
- D) 1024

---

### **Section 14: Tiled Matrix Multiplication Kernel**

**Q80.** In a tiled matrix multiplication kernel, how is the shared memory typically declared?
- A) `int ds_M[TILE_WIDTH][TILE_WIDTH];`
- B) `__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];`
- C) `__global__ float ds_M[TILE_WIDTH][TILE_WIDTH];`
- D) `static float ds_M[TILE_WIDTH][TILE_WIDTH];`

**Q81.** What is the purpose of the loop over tiles in tiled matrix multiplication?
- A) To iterate through all blocks
- B) To process each tile of the input matrices sequentially
- C) To synchronize warps
- D) To allocate memory

**Q82.** Which threads load data into shared memory in a tiled kernel?
- A) Only thread (0,0) of each block
- B) Only threads in the first warp
- C) All threads in the block collaborate
- D) Threads are randomly selected

**Q83.** After loading a tile into shared memory, what must happen before computation?
- A) The kernel must return
- B) A barrier synchronization (`__syncthreads()`)
- C) Data must be copied back to global memory
- D) The CPU must be notified

**Q84 (Tricky).** In tiled matrix multiplication with block size equal to tile size, each thread loads:
- A) One element from matrix M and one from matrix N per phase
- B) An entire row of M
- C) An entire column of N
- D) No data; only thread (0,0) loads data

---

### **Section 15: Memory and Data Locality**

**Q85.** What does "memory coalescing" refer to in CUDA?
- A) Combining multiple memory allocations
- B) Threads in a warp accessing consecutive memory addresses
- C) Freeing unused memory
- D) Compressing data in memory

**Q86.** What is the primary benefit of coalesced memory access?
- A) Simplified code
- B) Reduced memory transactions and improved bandwidth utilization
- C) More available memory
- D) Automatic error correction

**Q87.** Which access pattern is coalesced for a warp of 32 threads?
- A) All threads access the same address
- B) Consecutive threads access consecutive addresses
- C) Threads access random addresses
- D) Threads access strided addresses with large stride

**Q88.** What is "data locality" in the context of CUDA?
- A) Storing data close to where it's used (on-chip memory)
- B) Using only local variables
- C) Accessing only nearby memory addresses
- D) Restricting data to a single thread

**Q89 (Tricky).** Thread i accesses address `data[i * 1024]` while thread i+1 accesses `data[(i+1) * 1024]`. Is this coalesced?
- A) Yes, because addresses are deterministic
- B) No, because consecutive threads access non-consecutive addresses
- C) Yes, if the data is in shared memory
- D) It depends on the GPU architecture

---

### **Section 16: Execution Efficiency - Threads and Warps**

**Q90.** What is "warp divergence"?
- A) Warps executing on different SMs
- B) Threads within a warp taking different execution paths
- C) Blocks executing in different order
- D) Memory access patterns

**Q91.** What happens when a warp encounters divergent code paths?
- A) The warp splits permanently into smaller warps
- B) Some threads are masked inactive while others execute, reducing efficiency
- C) The kernel terminates
- D) The divergent code is skipped

**Q92.** Which scenario causes warp divergence?
- A) All threads in a warp execute the same `if` branch
- B) Threads in a warp take different branches based on thread ID
- C) Using shared memory
- D) Accessing global memory

**Q93.** What is "occupancy" in CUDA?
- A) The percentage of SMs in use
- B) The ratio of active warps to maximum possible active warps
- C) The amount of memory in use
- D) The percentage of threads that are busy

**Q94.** What limits occupancy?
- A) Number of threads per block only
- B) Register usage, shared memory usage, and block size
- C) Only global memory size
- D) CPU speed

**Q95.** How can divergence be minimized?
- A) Use more threads
- B) Organize data so threads in a warp follow the same execution path
- C) Increase block size
- D) Use more shared memory

**Q96 (Tricky).** A warp executes an `if-else` statement where the first 16 threads take the `if` branch and the last 16 take the `else` branch. What is the execution overhead?
- A) No overhead, both execute in parallel
- B) The execution time is approximately the sum of both paths
- C) Only one branch executes
- D) The kernel fails

**Q97.** Higher occupancy always leads to:
- A) Better performance
- B) Lower performance
- C) Better latency hiding, but not necessarily better performance
- D) More memory usage

**Q98.** What happens when register usage per thread is very high?
- A) Occupancy increases
- B) Occupancy decreases because fewer blocks can fit on an SM
- C) Performance always improves
- D) The kernel cannot launch

**Q99.** Bank conflicts occur in:
- A) Global memory
- B) Constant memory
- C) Shared memory
- D) Texture memory

**Q100 (Tricky).** A kernel has high arithmetic intensity and already hides latency well at 25% occupancy. Increasing occupancy to 75% will likely:
- A) Improve performance significantly
- B) Have minimal or no performance improvement
- C) Decrease performance
- D) Cause the kernel to fail

---

## Answer Key with Explanations

### Section 1: Significance and Application of HPC

**Q1. Answer: C**
*Explanation:* HPC is designed for computationally intensive tasks like climate modeling, drug discovery, and physics simulations. Word processing and spreadsheets require minimal computational power.

**Q2. Answer: B**
*Explanation:* HPC systems can solve complex modeling problems that exceed the capacity of standard computers, enabling scientific discoveries and breakthroughs.

**Q3. Answer: B**
*Explanation:* Scalability refers to the ability of an application to efficiently utilize increasing hardware resources (more cores, faster processors, etc.).

**Q4. Answer: C**
*Explanation:* Financial services use HPC for real-time market analysis, risk assessment, algorithmic trading, and data analytics.

**Q5. Answer: C**
*Explanation:* The Frontier supercomputer achieves approximately 1.206 exaflops (quintillion floating-point operations per second).

---

### Section 2: Multicore CPU vs GPU

**Q6. Answer: B**
*Explanation:* CPUs have few powerful cores optimized for sequential, low-latency processing. GPUs have thousands of simpler cores optimized for massively parallel, high-throughput processing.

**Q7. Answer: B**
*Explanation:* GPU cores are individually less powerful but number in the thousands, enabling massive parallelism for suitable workloads.

**Q8. Answer: B**
*Explanation:* CPUs rely heavily on large cache hierarchies (L1, L2, L3) for low latency. GPUs tolerate higher latency and dedicate more transistors to computation rather than caching.

**Q9. Answer: C**
*Explanation:* GPUs excel at massively parallel operations like matrix multiplication, image processing, and scientific simulations.

**Q10. Answer: B**
*Explanation:* NVIDIA GPUs use CUDA cores for general-purpose parallel processing (also Tensor cores for AI workloads).

**Q11. Answer: B (Tricky)**
*Explanation:* Despite lower clock speed, the GPU's 4096 cores enable it to excel at highly parallel tasks like matrix operations, while the CPU excels at sequential tasks requiring high single-thread performance.

---

### Section 3: SIMD vs MIMD

**Q12. Answer: B**
*Explanation:* SIMD stands for Single Instruction Multiple Data.

**Q13. Answer: C**
*Explanation:* SIMD has a single instruction decoder since all processing units execute the same instruction.

**Q14. Answer: B**
*Explanation:* MIMD systems use asynchronous programming with explicit synchronization, allowing different instructions on different data.

**Q15. Answer: B**
*Explanation:* MIMD is more flexible and suitable for general-purpose computing with diverse tasks, while SIMD excels at uniform operations.

**Q16. Answer: B**
*Explanation:* MIMD systems are more expensive than SIMD due to multiple decoders and greater complexity.

**Q17. Answer: B (Tricky)**
*Explanation:* SIMD is ideal for applying the same operation (filter) to many data elements (pixels) simultaneously.

**Q18. Answer: C**
*Explanation:* SIMD uses implicit/tacit synchronization since all units execute the same instruction.

---

### Section 4: Data Parallelism vs Task Parallelism

**Q19. Answer: B**
*Explanation:* Data parallelism involves performing the same task on different subsets of data.

**Q20. Answer: B**
*Explanation:* Task parallelism scales with the number of independent tasks that can be executed concurrently.

**Q21. Answer: B**
*Explanation:* Data parallelism typically uses synchronous computation.

**Q22. Answer: B**
*Explanation:* Computing different operations (sum, average, max) on the same data is task parallelism.

**Q23. Answer: B (Tricky)**
*Explanation:* Same operation on different data subsets is the definition of data parallelism.

**Q24. Answer: A**
*Explanation:* Data parallelism scales with input size, while task parallelism is limited by the number of distinct tasks.

---

### Section 5: Introduction to Heterogeneous Computing

**Q25. Answer: B**
*Explanation:* Heterogeneous computing uses different types of processors (CPUs, GPUs, FPGAs, ASICs) in the same system.

**Q26. Answer: C**
*Explanation:* The key advantage is assigning each workload to the processor type best suited for it, optimizing both performance and energy efficiency.

**Q27. Answer: C**
*Explanation:* In heterogeneous computing terminology, the CPU is called the "host."

**Q28. Answer: B**
*Explanation:* The GPU or accelerator is referred to as the "device."

**Q29. Answer: B**
*Explanation:* The PCIe bus is the primary pathway for CPU-GPU communication.

**Q30. Answer: A (Tricky)**
*Explanation:* The GPU cannot directly access the CPU's main memory address space. Data must be explicitly transferred via the PCIe bus (though Unified Memory abstracts this).

---

### Section 6: Portability and Scalability in Heterogeneous Parallel Computing

**Q31. Answer: B**
*Explanation:* Portability means the application can run efficiently on different hardware architectures (x86, ARM, GPU, etc.).

**Q32. Answer: B**
*Explanation:* Portability and scalability minimize software redevelopment costs when upgrading or changing hardware.

**Q33. Answer: C**
*Explanation:* Dynamic thread scheduling with fine-grained problem decomposition supports scalability.

**Q34. Answer: C**
*Explanation:* OpenCL (and SYCL, HIP) are standards designed for portability across heterogeneous platforms.

**Q35. Answer: A (Tricky)**
*Explanation:* The application scales on CPUs but lacks portability to GPU architecture, likely due to algorithm design not suitable for massive parallelism.

---

### Section 7: CUDA C - Threads, Kernels, Toolkit, Unified Memory

**Q36. Answer: B**
*Explanation:* `__global__` declares a kernel function callable from host and executable on device.

**Q37. Answer: B**
*Explanation:* CUDA kernels are launched using triple angle bracket syntax: `kernel<<<grid, block>>>(args)`.

**Q38. Answer: B**
*Explanation:* Unified Memory provides a single pointer accessible from both CPU and GPU, with automatic migration.

**Q39. Answer: C**
*Explanation:* `cudaMallocManaged()` allocates unified memory.

**Q40. Answer: B**
*Explanation:* Kernel launches are asynchronous by default, returning immediately to the CPU.

**Q41. Answer: C**
*Explanation:* `cudaFree()` frees memory allocated with `cudaMallocManaged()`.

**Q42. Answer: D (Tricky)**
*Explanation:* 256 blocks × 128 threads/block = 32,768 total threads.

---

### Section 8: CUDA Parallelism Model - Thread-Warp-Block-Grid Hierarchy

**Q43. Answer: D**
*Explanation:* The thread is the smallest unit of execution.

**Q44. Answer: C**
*Explanation:* A warp contains 32 threads in modern CUDA architectures.

**Q45. Answer: C**
*Explanation:* Maximum 1024 threads per block in CUDA.

**Q46. Answer: B**
*Explanation:* `threadIdx` provides the thread's unique ID within its block.

**Q47. Answer: C**
*Explanation:* Threads within a block share shared memory (and can synchronize).

**Q48. Answer: D**
*Explanation:* Grid is the highest level, containing all blocks.

**Q49. Answer: B**
*Explanation:* Blocks communicate only through global memory (no block-level synchronization or shared memory across blocks).

**Q50. Answer: B (Tricky)**
*Explanation:* 16 × 16 × 1 = 256 threads. 256 threads ÷ 32 threads/warp = 8 warps.

---

### Section 9: Kernel-based SPMD Parallel Programming

**Q51. Answer: B**
*Explanation:* SPMD = Single Program Multiple Data.

**Q52. Answer: C**
*Explanation:* All threads execute the same program, but each operates on its own data.

**Q53. Answer: B**
*Explanation:* Threads use built-in variables (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`) to compute their unique ID and determine which data to process.

**Q54. Answer: B**
*Explanation:* All threads execute the same kernel code but on different data elements.

**Q55. Answer: B (Tricky)**
*Explanation:* This is SPMD because all threads execute the same addition operation on their respective array elements.

---

### Section 10: Mapping and Color to Grayscale Conversion

**Q56. Answer: B**
*Explanation:* Global row index = `blockIdx.y * blockDim.y + threadIdx.y`.

**Q57. Answer: C**
*Explanation:* Standard weighted formula: `Gray = 0.21*R + 0.71*G + 0.07*B` (approximate values).

**Q58. Answer: A**
*Explanation:* ⌈1024 / 32⌉ = 32 blocks in x-dimension.

**Q59. Answer: B**
*Explanation:* Boundary checking prevents threads from accessing out-of-bounds memory when the image size isn't evenly divisible by block size.

**Q60. Answer: A (Tricky)**
*Explanation:* 100 / 16 = 6.25, so you need 7 blocks in each dimension, meaning some threads in the 7th block will be beyond the image bounds and should be idle with proper bounds checking.

---

### Section 11: Thread Scheduling and Warps

**Q61. Answer: B**
*Explanation:* The warp scheduler on each SM schedules warps for execution.

**Q62. Answer: C**
*Explanation:* Warp schedulers can switch every clock cycle (nanosecond scale).

**Q63. Answer: B**
*Explanation:* Rapid warp switching hides memory latency by executing other warps while some wait for data.

**Q64. Answer: B**
*Explanation:* Threads in a warp execute the same instruction simultaneously (SIMT model).

**Q65. Answer: C**
*Explanation:* GPU context switching is much faster because thread contexts are already in registers on the SM.

**Q66. Answer: A (Tricky)**
*Explanation:* This is warp divergence. The 10 active threads execute while 22 are masked inactive, but the warp still consumes execution time.

---

### Section 12: CUDA Memory Types

**Q67. Answer: C**
*Explanation:* Global memory is the largest (gigabytes).

**Q68. Answer: D**
*Explanation:* Registers have the fastest access (single cycle).

**Q69. Answer: B**
*Explanation:* Shared memory scope is the thread block.

**Q70. Answer: B**
*Explanation:* Constant memory is cached and optimized for broadcast reads (all threads reading same address).

**Q71. Answer: C**
*Explanation:* Local memory is implemented in global memory (slow).

**Q72. Answer: C**
*Explanation:* Automatic variables exist only during kernel execution.

**Q73. Answer: B (Tricky)**
*Explanation:* Each block has its own private copy of the shared memory array.

---

### Section 13: Tiled Parallel Algorithms

**Q74. Answer: B**
*Explanation:* Tiling reduces global memory accesses by reusing data loaded into faster shared memory.

**Q75. Answer: C**
*Explanation:* Tiles (submatrices) are loaded into shared memory.

**Q76. Answer: B**
*Explanation:* `__syncthreads()` synchronizes all threads in a block.

**Q77. Answer: A**
*Explanation:* Synchronization ensures all threads have loaded their tile data before computation begins.

**Q78. Answer: B**
*Explanation:* Tiled algorithms use phases, processing one tile per phase.

**Q79. Answer: C (Tricky)**
*Explanation:* 1024 / 16 = 64 tiles in each dimension. Each output element requires one row of tiles from M and one column from N, which is 64 tile loads.

---

### Section 14: Tiled Matrix Multiplication Kernel

**Q80. Answer: B**
*Explanation:* `__shared__` keyword declares shared memory arrays.

**Q81. Answer: B**
*Explanation:* The loop iterates through tiles of the input matrices.

**Q82. Answer: C**
*Explanation:* All threads collaborate to load tiles into shared memory.

**Q83. Answer: B**
*Explanation:* `__syncthreads()` ensures all data is loaded before computation.

**Q84. Answer: A (Tricky)**
*Explanation:* Each thread loads one element from M and one from N per tile phase.

---

### Section 15: Memory and Data Locality

**Q85. Answer: B**
*Explanation:* Memory coalescing occurs when threads in a warp access consecutive memory addresses.

**Q86. Answer: B**
*Explanation:* Coalescing reduces memory transactions, improving bandwidth utilization (can be 2-10x faster).

**Q87. Answer: B**
*Explanation:* Consecutive threads accessing consecutive addresses is the coalesced pattern.

**Q88. Answer: A**
*Explanation:* Data locality means keeping frequently accessed data in fast, on-chip memory (shared memory, cache).

**Q89. Answer: B (Tricky)**
*Explanation:* Threads access addresses 1024 apart (large stride), which is not coalesced.

---

### Section 16: Execution Efficiency - Threads and Warps

**Q90. Answer: B**
*Explanation:* Warp divergence occurs when threads in a warp take different execution paths.

**Q91. Answer: B**
*Explanation:* Divergent threads are masked inactive while others execute, reducing efficiency but not splitting the warp.

**Q92. Answer: B**
*Explanation:* Data-dependent branches where threads take different paths cause divergence.

**Q93. Answer: B**
*Explanation:* Occupancy = active warps / maximum possible active warps.

**Q94. Answer: B**
*Explanation:* Occupancy is limited by register usage, shared memory usage, and block size.

**Q95. Answer: B**
*Explanation:* Organizing data so threads in a warp follow the same path minimizes divergence.

**Q96. Answer: B (Tricky)**
*Explanation:* The warp must execute both paths sequentially (with masking), taking approximately the sum of both execution times.

**Q97. Answer: C**
*Explanation:* Higher occupancy helps hide latency but doesn't guarantee better performance (may reduce per-thread resources).

**Q98. Answer: B**
*Explanation:* High register usage reduces the number of blocks that can fit on an SM, lowering occupancy.

**Q99. Answer: C**
*Explanation:* Bank conflicts occur in shared memory when multiple threads access the same bank.

**Q100. Answer: B (Tricky)**
*Explanation:* If the kernel already hides latency effectively at low occupancy, increasing occupancy provides minimal benefit and may reduce per-thread resources.

---

## Quick Reference Cheat Sheet

### **Key Definitions**

**HPC (High-Performance Computing)**: Use of supercomputers and parallel processing to solve complex computational problems at very high speeds.

**Scalability**: Ability of an application to efficiently utilize increasing hardware resources.

**Portability**: Ability to run efficiently on different hardware architectures without major code changes.

**Heterogeneous Computing**: System using multiple types of processors (CPU, GPU, FPGA) to optimize performance.

**SIMD**: Single Instruction Multiple Data - same instruction on multiple data elements simultaneously.

**MIMD**: Multiple Instruction Multiple Data - different instructions on different data, asynchronous.

**Data Parallelism**: Same operation on different data subsets (scales with data size).

**Task Parallelism**: Different operations possibly on same data (scales with number of tasks).

**SPMD**: Single Program Multiple Data - all threads execute same code on different data.

---

### **CUDA Core Concepts**

#### **Thread Hierarchy** (Bottom to Top)
1. **Thread**: Smallest execution unit
2. **Warp**: 32 threads (execution scheduling unit)
3. **Block**: Up to 1024 threads (share shared memory)
4. **Grid**: All blocks (launched by one kernel)

#### **Memory Hierarchy** (Fast to Slow)
1. **Registers**: Fastest, per-thread, automatic variables
2. **Shared Memory**: Fast on-chip, per-block, explicit `__shared__`
3. **Local Memory**: Actually in global memory, per-thread spill
4. **Constant Memory**: Read-only, cached, broadcast optimization
5. **Global Memory**: Largest, slowest, accessible by all

#### **Key CUDA Keywords**
- `__global__`: Kernel function (called from host, runs on device)
- `__device__`: Device function (called from device, runs on device)
- `__host__`: Host function (default)
- `__shared__`: Shared memory variable
- `__syncthreads()`: Block-level barrier synchronization

#### **Built-in Variables**
- `threadIdx.x/y/z`: Thread index within block
- `blockIdx.x/y/z`: Block index within grid
- `blockDim.x/y/z`: Block dimensions (threads per block)
- `gridDim.x/y/z`: Grid dimensions (blocks per grid)

#### **Kernel Launch Syntax**
```cuda
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
// Common: kernel<<<numBlocks, threadsPerBlock>>>(args);
```

#### **Unified Memory**
- Allocate: `cudaMallocManaged(&ptr, size)`
- Free: `cudaFree(ptr)`
- Single pointer accessible from CPU and GPU
- Automatic migration between host and device

---

### **Commonly Confused Terms**

| Term | Confusion | Clarification |
|------|-----------|---------------|
| **Block vs Warp** | Both are thread groups | Block: programmer-defined grouping. Warp: hardware scheduling unit (32 threads) |
| **Local vs Shared Memory** | Both sound "local" | Local: per-thread, slow. Shared: per-block, fast |
| **Host vs Device** | Which is which? | Host = CPU, Device = GPU/Accelerator |
| **Global vs Unified Memory** | Are they the same? | Global: standard GPU memory. Unified: special managed memory accessible from both CPU/GPU |
| **Occupancy vs Utilization** | Sound similar | Occupancy: ratio of active warps. Utilization: how effectively resources are used |
| **SIMD vs SPMD** | Both sound similar | SIMD: hardware architecture. SPMD: programming model |

---

### **Important Formulas & Calculations**

#### **Thread Indexing**
```
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

#### **Grid Size Calculation**
```
// Ensure enough threads to cover all data elements
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
// Or: int numBlocks = ceil(N / (float)threadsPerBlock);
```

#### **Warps per Block**
```
numWarps = ceil(threadsPerBlock / 32.0)
```

#### **RGB to Grayscale**
```
Gray = 0.21 * R + 0.71 * G + 0.07 * B
// Or: 0.299*R + 0.587*G + 0.114*B
```

---

### **Optimization Guidelines**

#### **Memory Optimization**
1. **Coalesce memory accesses**: Consecutive threads access consecutive addresses
2. **Use shared memory**: Cache frequently accessed data
3. **Minimize host-device transfers**: Keep data on GPU
4. **Avoid bank conflicts**: Pad shared memory arrays if needed

#### **Execution Optimization**
1. **Minimize warp divergence**: Organize data so threads follow same path
2. **Maximize occupancy** (to a point): Balance resources (registers, shared memory, threads)
3. **Use tiling**: Reduce global memory accesses
4. **Synchronize carefully**: Only use `__syncthreads()` when necessary

#### **Block/Grid Configuration**
- Block size should be multiple of 32 (warp size)
- Common block sizes: 128, 256, 512
- Too small: poor occupancy. Too large: fewer blocks per SM

---

### **Tiled Matrix Multiplication Key Points**

1. **Divide matrices into tiles** that fit in shared memory
2. **Load tiles collaboratively**: All threads in block load data
3. **Synchronize after loading**: `__syncthreads()`
4. **Compute partial result** using tile data
5. **Synchronize before next tile**: `__syncthreads()`
6. **Repeat for all tiles** until complete

**Memory Access Reduction**: Without tiling, each element loads multiple times from global memory. With tiling, loads once per tile into shared memory.

---

### **Performance Metrics**

**Warp Divergence**: When threads in warp take different paths
- Impact: Serializes execution, reduces throughput
- Solution: Data organization, predication instead of branching

**Occupancy**: Active warps / Max possible active warps
- Limited by: registers/thread, shared memory/block, threads/block
- Higher ≠ always better (beyond latency hiding needs)

**Memory Bandwidth**: GB/s throughput
- Maximize through coalescing
- Non-coalesced can drop to 15% of peak
- Coalesced can approach 90% of peak

---

### **Common Exam Traps**

1. **Thread count calculation**: Don't confuse blocks with threads
   - Total threads = gridDim × blockDim (multiply all dimensions)

2. **Memory scope**: Shared memory is per-block, not per-grid

3. **Synchronization**: `__syncthreads()` only syncs within block, not across blocks

4. **Occupancy misconception**: Higher occupancy doesn't always mean better performance

5. **Warp size**: Always 32 in modern CUDA, not configurable

6. **Memory speed ranking**: Registers > Shared > Local/Global (Local is slow!)

7. **Coalescing**: Requires consecutive *addresses*, not just any pattern

8. **CPU vs GPU strengths**: 
   - CPU: Sequential, low-latency, complex logic
   - GPU: Parallel, high-throughput, simple repeated operations

---

### **Quick Mnemonics**

**Memory Speed: "Really Smart Lions Go Slowly"**
- **R**egisters (fastest)
- **S**hared
- **L**ocal
- **G**lobal (slowest)

**Thread Hierarchy: "Tigers Will Bite Grizzlies"**
- **T**hreads
- **W**arps
- **B**locks
- **G**rids

**CUDA Memory Keywords: "GRACE"**
- **G**lobal
- **R**egisters
- **A**utomatic (local)
- **C**onstant
- **E**xplicit shared

**Optimization Priority: "Can't Make Programs Fast"**
- **C**oalesce memory
- **M**inimize transfers
- **P**arallelism (maximize)
- **F**ast memory (use shared)

---

### **Last-Minute Review Points**

✓ **Warp = 32 threads** (fixed, hardware scheduling unit)
✓ **Max threads/block = 1024**
✓ **SPMD** = all threads run same code on different data
✓ **Host = CPU**, **Device = GPU**
✓ **Shared memory** = per-block, fast, requires `__shared__`
✓ **Global memory** = per-grid, slow, largest
✓ **Coalescing** = consecutive threads → consecutive addresses
✓ **Divergence** = threads in warp take different paths → serialization
✓ **Tiling** = use shared memory to reduce global accesses
✓ **`__syncthreads()`** = barrier sync within block only
✓ **Occupancy** = active warps / max warps
✓ **Kernel launch**: `<<<blocks, threads>>>`
✓ **Unified Memory**: `cudaMallocManaged()` + `cudaFree()`

---

## Good Luck on Your Exam! 🚀

**Study Tips:**
1. Focus on understanding **concepts** not just memorization
2. Practice **thread indexing calculations**
3. Understand **when to use which memory type**
4. Know **trade-offs** (CPU vs GPU, SIMD vs MIMD, high vs low occupancy)
5. Review **code patterns** (kernel launch, tiling, bounds checking)

**Common Exam Topics:**
- Thread hierarchy and indexing
- Memory types and their characteristics
- Warp divergence and how to avoid it
- Tiled algorithms and synchronization
- CPU vs GPU architectural differences
- Coalesced memory access patterns

