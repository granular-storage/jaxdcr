#!/usr/bin/env python3
"""
CIFAR-100 CNN training with JAX - with compilation caching and separated timing measurements.

This script implements a convolutional neural network (CNN) for CIFAR-100 image classification using JAX,
with detailed timing separation between data loading, compilation, and computation phases.

Key features:
- Convolutional layers for better image feature extraction
- Compilation caching to avoid recompilation on subsequent runs
- Separated timing for data loading, compilation, and computation
- Clear measurement of initial compilation vs subsequent executions
- Learning rate schedule for improved training
adding
JAX timing test with focused VLOG logging for pjrt_stream_executor_client
with highly optimized data transfer and computation overlap
"""

import os
import time
import pickle
import hashlib
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from jax.scipy.special import logsumexp
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import threading
import concurrent.futures
from queue import Queue

# Set environment variables for JAX
os.environ['JAX_ENABLE_X64'] = '0'  # Ensure we use float32 by default
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'  # Enable persistent compilation cache
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cifar100_cnn_cache'  # Set cache directory
os.environ['JAX_COMPILATION_CACHE_WRITE_ON_COMPILE'] = '1'  # Write cache immediately

# Optional: Set for detailed JAX logging
# os.environ['JAX_LOG_COMPILES'] = '1'  # Log compilation events

# Enable NumPy multithreading to use all available cores
num_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)

# Set environment variables for focused logging on pjrt_stream_executor_client
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'     # Show all logs including INFO
#os.environ['TF_CPP_VMODULE'] = 'pjrt_stream_executor_client=5'  # Verbose logging just for this file

#os.environ['JAX_LOG_COMPILES'] = '1'         # Log compilation events

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type='GPU')

# Set environment variable to enable detailed transfer timing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Helps isolate transfer times
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform-specific allocator

# Create cache directory for our own caching mechanism
cache_dir = "/tmp/jax_cifar100_cnn_aot_cache"
os.makedirs(cache_dir, exist_ok=True)

# Create cache directory
cache_dir_mm = "/tmp/jax_direct_aot_cache"
os.makedirs(cache_dir_mm, exist_ok=True)

# Hyperparameters
initial_step_size = 0.01
num_epochs = 20
batch_size = 128
n_targets = 100
data_dir = '/tmp/tfds'  # Change this to your desired data directory

# CNN Architecture
class CNN:
    def __init__(self):
        # Architecture details (used for cache key)
        self.conv_channels = [32, 64, 128]
        self.fc_sizes = [512, 100]
        
    def init_params(self, key):
        """Initialize CNN parameters"""
        # Initialize convolutional layers
        conv_keys = random.split(key, len(self.conv_channels) + 1)
        
        # First convolutional layer (3 input channels)
        conv1_w = random.normal(conv_keys[0], (5, 5, 3, self.conv_channels[0])) * 0.1
        conv1_b = jnp.zeros(self.conv_channels[0])
        
        # Additional convolutional layers
        conv_params = [(conv1_w, conv1_b)]
        for i in range(1, len(self.conv_channels)):
            w = random.normal(conv_keys[i], 
                             (3, 3, self.conv_channels[i-1], self.conv_channels[i])) * 0.1
            b = jnp.zeros(self.conv_channels[i])
            conv_params.append((w, b))
            
        # Initialize fully connected layers
        # Calculate size after convolutions and pooling
        # After 3 pooling layers of stride 2, 32x32 -> 4x4
        flattened_size = 4 * 4 * self.conv_channels[-1]
        
        fc_keys = random.split(conv_keys[-1], len(self.fc_sizes))
        fc_params = []
        
        # First FC layer from flattened conv output
        w = random.normal(fc_keys[0], (flattened_size, self.fc_sizes[0])) * jnp.sqrt(2.0 / flattened_size)
        b = jnp.zeros(self.fc_sizes[0])
        fc_params.append((w, b))
        
        # Additional FC layers
        for i in range(1, len(self.fc_sizes)):
            w = random.normal(fc_keys[i], 
                             (self.fc_sizes[i-1], self.fc_sizes[i])) * jnp.sqrt(2.0 / self.fc_sizes[i-1])
            b = jnp.zeros(self.fc_sizes[i])
            fc_params.append((w, b))
            
        return {'conv': conv_params, 'fc': fc_params}
    
    def apply_conv_layer(self, x, w, b):
        """Apply a convolutional layer with ReLU activation"""
        # Use conv_general_dilated with appropriate dimension numbers
        y = lax.conv_general_dilated(
            x,                                  # input
            w,                                  # filter
            window_strides=(1, 1),             # stride
            padding='SAME',                    # padding
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')  # dimension format
        )
        y = y + b[None, None, None, :]  # Add bias to each channel
        return jnp.maximum(0, y)  # ReLU
    
    def apply_max_pool(self, x):
        """Apply 2x2 max pooling with stride 2"""
        return lax.reduce_window(
            x,                                  # input
            -jnp.inf,                          # init value
            lax.max,                           # computation
            (1, 2, 2, 1),                      # window dimensions
            (1, 2, 2, 1),                      # window stride
            'SAME'                             # padding
        )
    
    def predict(self, params, image):
        """Forward pass through the CNN"""
        # Reshape input to 4D: (batch_size, height, width, channels)
        # For single example: (1, 32, 32, 3)
        x = image.reshape(1, 32, 32, 3)
        
        # Apply convolutional layers with pooling
        for w, b in params['conv']:
            x = self.apply_conv_layer(x, w, b)
            x = self.apply_max_pool(x)
        
        # Flatten output
        x = x.reshape(-1)
        
        # Apply fully connected layers
        for i, (w, b) in enumerate(params['fc'][:-1]):
            x = jnp.dot(x, w) + b
            x = jnp.maximum(0, x)  # ReLU
        
        # Final layer (logits)
        final_w, final_b = params['fc'][-1]
        logits = jnp.dot(x, final_w) + final_b
        
        return logits - logsumexp(logits)  # Log probabilities
        
    def get_architecture_str(self):
        """Get string representation of architecture for cache key"""
        conv_str = "_".join(map(str, self.conv_channels))
        fc_str = "_".join(map(str, self.fc_sizes))
        return f"cnn_conv{conv_str}_fc{fc_str}"

# Instantiate CNN model
cnn_model = CNN()

# Cache management functions
def get_cache_key(arch_str, jax_version):
    """Generate a unique cache key based on network architecture and JAX version"""
    key_str = f"cifar100_{arch_str}_{jax_version}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_key_mm(size, jax_version):
    """Generate a unique cache key based on matrix size and JAX version"""
    key_str = f"matmul_{size}x{size}_{jax_version}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_compile_metadata():
    """Load compilation metadata if it exists"""
    arch_str = cnn_model.get_architecture_str()
    cache_key = get_cache_key(arch_str, jax.__version__)
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

def load_compile_metadata_mm(size):
    """Load compilation metadata if it exists"""
    cache_key = get_cache_key_mm(size, jax.__version__)
    metadata_path = os.path.join(cache_dir_mm, f"{cache_key}_metadata.pickle")
    
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

def save_compile_metadata(duration):
    """Save compilation metadata"""
    arch_str = cnn_model.get_architecture_str()
    cache_key = get_cache_key(arch_str, jax.__version__)
    metadata_path = os.path.join(cache_dir, f"{cache_key}_metadata.pickle")
    
    metadata = {
        'compiled': True,
        'architecture': arch_str,
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

def save_compile_metadata_mm(size, duration):
    """Save compilation metadata"""
    cache_key = get_cache_key_mm(size, jax.__version__)
    metadata_path = os.path.join(cache_dir_mm, f"{cache_key}_metadata.pickle")
    
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
    compiled_before, _ = load_compile_metadata_mm(size)
    
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
        save_compile_metadata_mm(size, compilation_duration)
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
        compiled_before, _ = load_compile_metadata_mm(size)
        if not compiled_before:
            print("Precompiling function...")
            small_a = np.ones((100, 100), dtype=np.float32)
            small_b = np.ones((100, 100), dtype=np.float32)
            dev_a = jax.device_put(small_a)
            dev_b = jax.device_put(small_b)
            result = self.jitted_fn(dev_a, dev_b)
            result.block_until_ready()
            save_compile_metadata_mm(size, 0.0)  # Just to mark as compiled
    
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
    worker = AdvancedMatrixWorker(size=4000, num_matrices=200, max_concurrent_transfers=4)
    results = worker.run()
    return results

# Vectorize the predict function to handle batches
batched_predict = vmap(cnn_model.predict, in_axes=(None, 0))

# Utility function to create one-hot encodings
def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Accuracy computation
@jit
def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

# Loss function
def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

# Calculate learning rate based on epoch
def get_step_size(epoch):
    """Implements a simple learning rate decay schedule"""
    if epoch < 10:
        return initial_step_size
    elif epoch < 20:
        return initial_step_size * 0.5
    else:
        return initial_step_size * 0.1

# Define the gradient update function (not JIT-compiled yet)
def update_raw(params, x, y, step_size):
    grads = grad(loss)(params, x, y)
    
    # Update convolutional layers
    new_conv_params = []
    for (w, b), (dw, db) in zip(params['conv'], grads['conv']):
        new_conv_params.append((w - step_size * dw, b - step_size * db))
    
    # Update fully connected layers
    new_fc_params = []
    for (w, b), (dw, db) in zip(params['fc'], grads['fc']):
        new_fc_params.append((w - step_size * dw, b - step_size * db))
    
    return {'conv': new_conv_params, 'fc': new_fc_params}

# Pre-process images
def preprocess_images(images):
    """Preprocess image data: normalize and standardize"""
    # Scale pixel values to [0, 1]
    images = images.astype(jnp.float32) / 255.0
    
    # Standardize each channel
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    
    # Reshape mean and std for broadcasting
    mean = mean.reshape(1, 1, 1, 3)
    std = std.reshape(1, 1, 1, 3)
    
    # Apply standardization
    orig_shape = images.shape
    if len(orig_shape) == 3:  # Single image
        images = images.reshape(32, 32, 3)
        images = (images - mean[0]) / std[0]
    else:  # Batch of images
        images = images.reshape(-1, 32, 32, 3)
        images = (images - mean) / std
        images = images.reshape(orig_shape)
    
    return images

# Pre-compile with JIT after initialization
update_jit = None  # Will be compiled later

def main():
    print("\n=== Starting CIFAR-100 CNN with Compilation Caching ===\n")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Time model initialization
    print("\n=== Phase 1: Model Initialization ===")
    init_start = time.time()
    params = cnn_model.init_params(random.PRNGKey(0))
    init_time = time.time() - init_start
    print(f"Model initialization time: {init_time:.4f} seconds")
    
    # Phase 2: Dataset loading (timed)
    print("\n=== Phase 2: Dataset Loading ===")
    data_load_start = time.time()
    
    # Load full datasets for evaluation
    cifar_data, info = tfds.load(name="cifar100", batch_size=-1, data_dir=data_dir, with_info=True)
    cifar_data = tfds.as_numpy(cifar_data)
    train_data, test_data = cifar_data['train'], cifar_data['test']
    num_classes = 100  # CIFAR-100 has 100 classes
    
    # Print info about the dataset structure
    print("Dataset info:")
    print(f"  Features: {list(info.features.keys())}")
    if 'label' in info.features:
        print(f"  Label classes: {info.features['label'].num_classes}")
    
    # Process full train set with improved preprocessing
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = preprocess_images(train_images)
    train_labels = one_hot(train_labels, num_classes)
    
    # Process full test set with improved preprocessing
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = preprocess_images(test_images)
    test_labels = one_hot(test_labels, num_classes)
    
    data_load_time = time.time() - data_load_start
    
    print(f'Train: {train_images.shape} {train_labels.shape}')
    print(f'Test: {test_images.shape} {test_labels.shape}')
    print(f"Dataset loading time: {data_load_time:.4f} seconds")
    
    # Create preprocessed batch dataset
    def get_train_batches():
        batch_load_start = time.time()
        # Create dataset with preprocessed images and shuffling
        N = train_images.shape[0]
        indices = np.random.permutation(N)
        shuffled_images = train_images[indices]
        shuffled_labels = train_labels[indices]
        
        # Create batches
        num_batches = N // batch_size
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = shuffled_images[start_idx:end_idx]
            batch_y = shuffled_labels[start_idx:end_idx]
            batches.append((batch_x, batch_y))
            
        batch_load_time = time.time() - batch_load_start
        return batches, batch_load_time
    
    # Phase 3: Compilation (timed)
    print("\n=== Phase 3: Function Compilation ===")
    
    # Check if we've compiled this network before
    compiled_before, metadata = load_compile_metadata()
    global update_jit
    
    if not compiled_before:
        print("No compilation metadata found, JIT-compiling update function...")
        compilation_start = time.time()
        
        # JIT-compile the update function
        update_jit = jit(update_raw)
        
        # Force compilation by running once with sample data
        sample_x = jnp.ones((batch_size, 32, 32, 3))
        sample_y = jnp.ones((batch_size, num_classes))
        _ = update_jit(params, sample_x, sample_y, initial_step_size)
        
        # Also force compilation of batched_predict
        _ = batched_predict(params, sample_x)
        
        compilation_end = time.time()
        compilation_time = compilation_end - compilation_start
        print(f"Compilation completed in {compilation_time:.4f} seconds")
        
        # Save metadata for future runs
        save_compile_metadata(compilation_time)
    else:
        print("Using previously compiled function from cache")
        compilation_time = metadata.get('duration', 0.0)
        print(f"Previous compilation took {compilation_time:.4f} seconds")
        
        # Still need to define the JIT function even though it's cached
        update_jit = jit(update_raw)
    
    # Phase 4: Training loop (with detailed timing)
    print("\n=== Phase 4: Training Loop ===")
    training_start = time.time()
    
    total_batch_load_time = 0
    total_compute_time = 0
    total_eval_time = 0
    
    # Track best accuracy
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Get learning rate for this epoch
        current_step_size = get_step_size(epoch)
        
        # Get batches and measure loading time
        batches, batch_load_time = get_train_batches()
        total_batch_load_time += batch_load_time
        
        # Training computation time
        compute_start = time.time()
        batch_count = 0
        for x, y in batches:
            batch_count += 1
            # Use the JIT-compiled update function with current learning rate
            #cpu to gpu batches memory transfer
            params = update_jit(params, x, y, current_step_size)
        
        # Make sure all operations are complete before timing
        # Need to handle nested structure
        jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), params)
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        # Evaluation computation time
        eval_start = time.time()
        
        # For larger datasets, evaluate on a subset for speed
        eval_subset_size = min(5000, train_images.shape[0])
        train_idx = np.random.choice(train_images.shape[0], eval_subset_size, replace=False)
        # transfers to gpu from cpu 
        train_acc = accuracy(params, train_images[train_idx], train_labels[train_idx])
        
        test_acc = accuracy(params, test_images, test_labels)
        
        # Make sure accuracy computation is complete
        train_acc = jax.block_until_ready(train_acc)
        test_acc = jax.block_until_ready(test_acc)
        eval_time = time.time() - eval_start
        total_eval_time += eval_time
        
        # Track best performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}:")
        print(f"  Learning rate: {current_step_size:.5f}")
        print(f"  Total epoch time: {epoch_time:.4f} sec")
        print(f"  Batch loading time: {batch_load_time:.4f} sec")
        print(f"  Training computation time: {compute_time:.4f} sec")
        print(f"  Evaluation time: {eval_time:.4f} sec")
        print(f"  Number of batches: {batch_count}")
        print(f"  Training set accuracy: {train_acc:.6f} ({train_acc*100:.2f}%)")
        print(f"  Test set accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
        print(f"  Best test accuracy so far: {best_test_acc:.6f} ({best_test_acc*100:.2f}%)")
        print()
    
    training_time = time.time() - training_start
    
    # Print summary statistics with clear separation of compilation and computation
    print("=== Training Summary ===")
    print(f"Model initialization time: {init_time:.4f} seconds")
    print(f"Initial dataset loading time: {data_load_time:.4f} seconds")
    print(f"Function compilation time: {compilation_time:.4f} seconds")
    print(f"Training loop time: {training_time:.4f} seconds")
    print(f"  - Batch loading time: {total_batch_load_time:.4f} seconds")
    print(f"  - Computation time: {total_compute_time:.4f} seconds")
    print(f"  - Evaluation time: {total_eval_time:.4f} seconds")
    print(f"Total time: {init_time + data_load_time + (0 if compiled_before else compilation_time) + training_time:.4f} seconds")
    print(f"Final test accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
    print(f"Best test accuracy: {best_test_acc:.6f} ({best_test_acc*100:.2f}%)")
    
    # If this was a reuse of compilation, show the time savings
    if compiled_before:
        print(f"\nReused cached compilation, saved approximately {compilation_time:.4f} seconds")

def main_mm():
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
    #test_large_matrix_multiplication_direct_aot_repeat(iterations=50)
    
    # Scenario 2: Overlap data transfer and computation
    test_large_matrix_multiplication_direct_aot_multiple()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
    main_mm()
