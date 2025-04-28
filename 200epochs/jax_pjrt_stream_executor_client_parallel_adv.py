#!/usr/bin/env python3
"""
JAX timing test with focused VLOG logging for pjrt_stream_executor_client
with highly optimized data transfer and computation overlap
"""
import os
os.environ['JAX_ENABLE_X64'] = '0'  # Ensure we use float32 by default
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'  # Enable persistent compilation cache
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_persistent_cache'  # Set cache directory
os.environ['JAX_COMPILATION_CACHE_WRITE_ON_COMPILE'] = '1'  # Write cache immediately

import time
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import hashlib
import threading
import concurrent.futures
from queue import Queue

# Enable NumPy multithreading to use all available cores
num_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)

# Set environment variables for focused logging on pjrt_stream_executor_client
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'     # Show all logs including INFO
os.environ['TF_CPP_VMODULE'] = 'pjrt_stream_executor_client=5'  # Verbose logging just for this file
os.environ['JAX_LOG_COMPILES'] = '1'         # Log compilation events

# Set environment variable to enable detailed transfer timing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Helps isolate transfer times
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform-specific allocator

# Create cache directory
cache_dir = "/tmp/jax_direct_aot_cache"
os.makedirs(cache_dir, exist_ok=True)

def get_cache_key(size, jax_version):
    """Generate a unique cache key based on matrix size and JAX version"""
    key_str = f"matmul_{size}x{size}_{jax_version}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_compile_metadata(size):
    """Load compilation metadata if it exists"""
    cache_key = get_cache_key(size, jax.__version__)
    metadata_path = os.path.join(cache_dir, f"{cache_key}_metadata.pickle")
    
    compiled_before = False
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            if metadata.get('compiled') and metadata.get('jax_version') == jax.__version__:
                compiled_before = True
                print(f"Found existing compilation metadata from {time.ctime(metadata.get('timestamp'))}")
                return True, metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    return False, None

def save_compile_metadata(size, duration):
    """Save compilation metadata"""
    cache_key = get_cache_key(size, jax.__version__)
    metadata_path = os.path.join(cache_dir, f"{cache_key}_metadata.pickle")
    
    metadata = {
        'compiled': True,
        'size': size,
        'timestamp': time.time(),
        'jax_version': jax.__version__,
        'duration': duration
    }
    
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved compilation metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

def generate_matrix_pairx(size, seed=42):
    """Generate a pair of random matrices with the given size"""
    np.random.seed(seed)
    host_a = np.random.normal(0, 1, (size, size)).astype(np.float32)
    host_b = np.random.normal(0, 1, (size, size)).astype(np.float32)
    return host_a, host_b

def generate_matrix_pair(size, seed=42):
    """Generate a pair of random matrices with optimal performance
    
    Args:
        size: Size of the square matrices
        seed: Random seed
        
    Returns:
        Tuple of two float32 matrices
    """
    # Set the random seed
    np.random.seed(seed)
    
    # Generate directly with the correct dtype
    host_a = np.random.randn(size, size).astype(np.float32)
    host_b = np.random.randn(size, size).astype(np.float32)
    
    # Memory layout optimization (ensure C-contiguous)
    host_a = np.ascontiguousarray(host_a)
    host_b = np.ascontiguousarray(host_b)
    
    return host_a, host_b

def test_large_matrix_multiplication_direct_aot():
    """
    Test large matrix multiplication with direct AOT compilation and clear separation 
    between transfer, computation, and result storage phases
    """
    print("\n=== Testing Large Matrix Multiplication with Direct AOT Compilation ===")
    
    # Matrix size
    size = 4000
    bytes_per_matrix = size * size * 4  # 4 bytes for float32
    
    # Define simple matmul function
    def matmul_fn(x, y):
        return jnp.dot(x, y)
    
    # Create JIT-compiled function
    jitted_fn = jax.jit(matmul_fn)
    
    # Check if we have compilation metadata
    compiled_before, _ = load_compile_metadata(size)
    
    # Phase 1: Generate matrices
    print(f"Creating {size}x{size} matrices ({bytes_per_matrix/1e9:.2f} GB each)...")
    print("Generating host matrices...")
    host_a, host_b = generate_matrix_pair(size)
    print(f"Total host memory: {bytes_per_matrix*2/1e9:.2f} GB")
    
    # Phase 2: Transfer to device (clearly timed)
    print("\n--- Transfer Timing ---")
    transfer_start = time.time()
    dev_a = jax.device_put(host_a)
    dev_b = jax.device_put(host_b)
    dev_a.block_until_ready()
    dev_b.block_until_ready()
    transfer_end = time.time()
    
    transfer_duration = transfer_end - transfer_start
    transfer_rate = bytes_per_matrix*2 / transfer_duration / 1e9  # GB/s
    print(f"Transfer time: {transfer_duration:.6f} seconds")
    print(f"Transfer rate: {transfer_rate:.2f} GB/s")
    
    # Phase 3: Compilation (if needed)
    print("\n--- Compilation Phase ---")
    if not compiled_before:
        print("No valid compilation metadata found, performing compilation...")
        
        # Compile by executing once
        compilation_start = time.time()
        result = jitted_fn(dev_a, dev_b)
        result.block_until_ready()
        compilation_end = time.time()
        
        compilation_duration = compilation_end - compilation_start
        print(f"Initial compilation completed in {compilation_duration:.6f} seconds")
        
        # Save metadata about this compilation
        save_compile_metadata(size, compilation_duration)
    else:
        print("Using previously compiled function from JAX's cache")
    
    # Phase 4: Execution (clearly timed)
    print("\n--- Execution Phase ---")
    
    # Clear any pending operations
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), [])
    
    # Execute the function
    execution_start = time.time()
    result = jitted_fn(dev_a, dev_b)
    result.block_until_ready()
    execution_end = time.time()
    
    execution_duration = execution_end - execution_start
    
    # Calculate FLOPS
    flops = 2 * size**3  # 2*N^3 FLOPs for matrix multiplication
    tflops = flops / execution_duration / 1e12
    
    print(f"Execution completed in {execution_duration:.6f} seconds")
    print(f"Performance: {tflops:.2f} TFLOPS")
    
    # Phase 5: Transfer result back to host
    print("\n--- Result Transfer ---")
    result_transfer_start = time.time()
    host_result = np.array(result)
    result_transfer_end = time.time()
    result_transfer_duration = result_transfer_end - result_transfer_start
    
    print(f"Result transfer time: {result_transfer_duration:.6f} seconds")
    print(f"Result transfer rate: {bytes_per_matrix / result_transfer_duration / 1e9:.2f} GB/s")
    
    # Summary
    print("\n=== Direct AOT Performance Summary ===")
    print(f"Input data transfer: {transfer_duration:.6f} seconds")
    print(f"Execution time: {execution_duration:.6f} seconds")
    print(f"Result transfer: {result_transfer_duration:.6f} seconds")
    
    compute_transfer_ratio = execution_duration / transfer_duration
    print(f"\nExecution/Transfer ratio: {compute_transfer_ratio:.6f}")
    print(f"This means computation takes {compute_transfer_ratio*100:.2f}% of the time it takes to transfer the input data")
    
    return {
        'transfer_duration': transfer_duration,
        'execution_duration': execution_duration,
        'result_transfer_duration': result_transfer_duration,
        'tflops': tflops
    }

def test_large_matrix_multiplication_direct_aot_repeat(iterations=5):
    """Run the matrix multiplication test multiple times in a row"""
    print(f"\n=== Running Matrix Multiplication Test {iterations} Times ===")
    
    results = []
    total_start = time.time()
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")
        result = test_large_matrix_multiplication_direct_aot()
        results.append(result)
    
    total_end = time.time()
    total_time = total_end - total_start
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total time for {iterations} iterations: {total_time:.6f} seconds")
    print(f"Average time per iteration: {total_time/iterations:.6f} seconds")
    
    avg_transfer = sum(r['transfer_duration'] for r in results) / iterations
    avg_execution = sum(r['execution_duration'] for r in results) / iterations
    avg_result_transfer = sum(r['result_transfer_duration'] for r in results) / iterations
    avg_tflops = sum(r['tflops'] for r in results) / iterations
    
    print(f"Average input transfer time: {avg_transfer:.6f} seconds")
    print(f"Average execution time: {avg_execution:.6f} seconds")
    print(f"Average result transfer time: {avg_result_transfer:.6f} seconds")
    print(f"Average performance: {avg_tflops:.2f} TFLOPS")
    
    return results

def generate_and_transfer_matrix_pair(idx, size, num_matrices):
    """Generate and transfer a matrix pair in a separate function for parallelization"""
    seed = 42 + idx
    print(f"Worker {idx+1}: Generating matrix pair {idx+1}/{num_matrices}")
    
    # Generate matrices
    host_a, host_b = generate_matrix_pair(size, seed)
    
    # Transfer to device and measure time
    transfer_start = time.time()
    dev_a = jax.device_put(host_a)
    dev_b = jax.device_put(host_b)
    dev_a.block_until_ready()
    dev_b.block_until_ready()
    transfer_end = time.time()
    
    transfer_duration = transfer_end - transfer_start
    print(f"Worker {idx+1}: Matrix pair {idx+1} transferred in {transfer_duration:.6f} seconds")
    
    return dev_a, dev_b, idx, transfer_duration

class AdvancedMatrixWorker:
    """Worker class for highly optimized overlapping of computation and data transfer"""
    def __init__(self, size=16000, num_matrices=5, max_concurrent_transfers=2):
        self.size = size
        self.num_matrices = num_matrices
        self.bytes_per_matrix = size * size * 4  # 4 bytes for float32
        self.max_concurrent_transfers = max_concurrent_transfers
        
        # Define compute queue and results queue
        self.compute_queue = Queue()
        self.output_queue = Queue()
        
        # Define simple matmul function
        self.jitted_fn = jax.jit(lambda x, y: jnp.dot(x, y))
        
        # Ensure compilation is done
        compiled_before, _ = load_compile_metadata(size)
        if not compiled_before:
            print("Precompiling function...")
            small_a = np.ones((100, 100), dtype=np.float32)
            small_b = np.ones((100, 100), dtype=np.float32)
            dev_a = jax.device_put(small_a)
            dev_b = jax.device_put(small_b)
            result = self.jitted_fn(dev_a, dev_b)
            result.block_until_ready()
            save_compile_metadata(size, 0.0)  # Just to mark as compiled
    
    def compute_worker(self):
        """Thread function that computes on device data and transfers results back"""
        while True:
            # Get matrix pair from queue
            item = self.compute_queue.get()
            if item is None:  # End signal
                break
                
            dev_a, dev_b, idx, transfer_duration = item
            
            # Execute computation
            print(f"Compute: Processing matrix pair {idx+1}/{self.num_matrices}")
            execution_start = time.time()
            result = self.jitted_fn(dev_a, dev_b)
            result.block_until_ready()
            execution_end = time.time()
            
            execution_duration = execution_end - execution_start
            
            # Transfer result back
            result_transfer_start = time.time()
            host_result = np.array(result)
            result_transfer_end = time.time()
            
            result_transfer_duration = result_transfer_end - result_transfer_start
            
            # Calculate FLOPS
            flops = 2 * self.size**3  # 2*N^3 FLOPs for matrix multiplication
            tflops = flops / execution_duration / 1e12
            
            print(f"Compute: Matrix pair {idx+1} computed in {execution_duration:.6f} seconds ({tflops:.2f} TFLOPS)")
            print(f"Compute: Result transferred in {result_transfer_duration:.6f} seconds")
            
            # Add to output queue
            self.output_queue.put({
                'index': idx,
                'transfer_duration': transfer_duration,
                'execution_duration': execution_duration,
                'result_transfer_duration': result_transfer_duration,
                'tflops': tflops
            })
            
            # Signal that we're done with this item
            self.compute_queue.task_done()
    
    def run(self):
        """Run the highly optimized overlapped computation test with parallel transfers"""
        print(f"\n=== Testing Matrix Multiplication with Advanced Overlapped Data Transfer and Computation ===")
        print(f"Matrix size: {self.size}x{self.size} ({self.bytes_per_matrix/1e9:.2f} GB each)")
        print(f"Number of matrix pairs to process: {self.num_matrices}")
        print(f"Maximum concurrent transfers: {self.max_concurrent_transfers}")
        
        # Start compute thread
        compute_thread = threading.Thread(target=self.compute_worker)
        compute_thread.start()
        
        # Start timer
        total_start = time.time()
        
        # Use ThreadPoolExecutor to handle parallel transfers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_transfers) as executor:
            # Submit all transfer tasks
            future_to_idx = {
                executor.submit(generate_and_transfer_matrix_pair, i, self.size, self.num_matrices): i 
                for i in range(self.num_matrices)
            }
            
            # Process completed transfers as they finish
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    # Add to compute queue
                    self.compute_queue.put(result)
                except Exception as exc:
                    print(f"Matrix transfer {idx} generated an exception: {exc}")
        
        # Signal that no more transfers are coming
        self.compute_queue.put(None)
        
        # Wait for compute thread to finish
        compute_thread.join()
        
        # Stop timer
        total_end = time.time()
        total_duration = total_end - total_start
        
        # Collect results
        results = []
        while not self.output_queue.empty():
            results.append(self.output_queue.get())
        
        # Sort results by index
        results.sort(key=lambda x: x['index'])
        
        # Print summary statistics
        print("\n=== Advanced Overlapped Execution Summary ===")
        print(f"Total time for {self.num_matrices} matrix pairs: {total_duration:.6f} seconds")
        print(f"Average time per matrix pair: {total_duration/self.num_matrices:.6f} seconds")
        
        avg_transfer = sum(r['transfer_duration'] for r in results) / self.num_matrices
        avg_execution = sum(r['execution_duration'] for r in results) / self.num_matrices
        avg_result_transfer = sum(r['result_transfer_duration'] for r in results) / self.num_matrices
        avg_tflops = sum(r['tflops'] for r in results) / self.num_matrices
        
        print(f"Average input transfer time: {avg_transfer:.6f} seconds")
        print(f"Average execution time: {avg_execution:.6f} seconds")
        print(f"Average result transfer time: {avg_result_transfer:.6f} seconds")
        print(f"Average performance: {avg_tflops:.2f} TFLOPS")
        
        # Calculate speedup
        sequential_time = (avg_transfer + avg_execution + avg_result_transfer) * self.num_matrices
        speedup = sequential_time / total_duration
        
        print(f"\nEstimated sequential time: {sequential_time:.6f} seconds")
        print(f"Actual overlapped time: {total_duration:.6f} seconds")
        print(f"Speedup from overlapping: {speedup:.2f}x")
        
        return results

def test_large_matrix_multiplication_direct_aot_multiple():
    """
    Test large matrix multiplication with highly optimized overlapped data transfer and computation
    This function creates a pipeline where multiple data transfers can happen concurrently with computation
    """
    worker = AdvancedMatrixWorker(size=4000, num_matrices=50, max_concurrent_transfers=4)
    results = worker.run()
    return results

def main():
    print("=== JAX XLA Timing Test with Focused PJRT StreamExecutor Logging ===")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Log settings:")
    print(f"  TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL')}")
    print(f"  TF_CPP_VMODULE: {os.environ.get('TF_CPP_VMODULE')}")
    print(f"  JAX_LOG_COMPILES: {os.environ.get('JAX_LOG_COMPILES')}")

    # Modify this section to run the desired test
    
    # Scenario 1: Run five times in a row
    test_large_matrix_multiplication_direct_aot_repeat(iterations=50)
    
    # Scenario 2: Overlap data transfer and computation
    #test_large_matrix_multiplication_direct_aot_multiple()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
