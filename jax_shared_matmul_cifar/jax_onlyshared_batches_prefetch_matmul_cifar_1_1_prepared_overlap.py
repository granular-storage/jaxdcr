#!/usr/bin/env python3
"""
CIFAR-100 CNN training with JAX - with precomputation caching and optimized streaming

Enhanced version with:
- Matrix and batch data precomputation and caching
- Separation of data generation from measurement
- Optimized memory access patterns for streaming data
- Clearly separated transfer and computation timing
"""

import os

os.environ["XLA_FLAGS"] = '--xla_gpu_enable_latency_hiding_scheduler=true'
# Set environment variables for focused logging on pjrt_stream_executor_client
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'     # Show all logs including INFO
#os.environ['TF_CPP_VMODULE'] = 'pjrt_stream_executor_client=5'  # Verbose logging just for this file

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
import shutil

import threading
import concurrent.futures
from queue import Queue
import signal

# Set environment variables for JAX
os.environ['JAX_ENABLE_X64'] = '0'  # Ensure we use float32 by default
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'  # Enable persistent compilation cache
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cifar100_cnn_cache'  # Set cache directory
os.environ['JAX_COMPILATION_CACHE_WRITE_ON_COMPILE'] = '1'  # Write cache immediately

# Enable NumPy multithreading to use all available cores
num_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type='GPU')

# Set environment variable to enable detailed transfer timing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Helps isolate transfer times
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform-specific allocator

# Create cache directories
cache_dir = "/tmp/jax_cifar100_cnn_aot_cache"
cache_dir_mm = "/tmp/jax_direct_aot_cache"
matrix_cache_dir = "/tmp/matrix_cache"
batch_cache_dir = "/tmp/batch_cache"

for directory in [cache_dir, cache_dir_mm, matrix_cache_dir, batch_cache_dir]:
    os.makedirs(directory, exist_ok=True)

# Hyperparameters
initial_step_size = 0.01
num_epochs = 200
batch_size = 128
n_targets = 100
data_dir = '/tmp/tfds'  # Change this to your desired data directory

# ================================
# PRECOMPUTATION UTILITY FUNCTIONS
# ================================

def precompute_matrices(size=4000, num_matrices=200, cache_dir=matrix_cache_dir, force=False):
    """Precompute matrices and save to disk cache
    
    Args:
        size: Size of square matrices
        num_matrices: Number of matrix pairs to generate
        cache_dir: Directory to store cached matrices
        force: If True, regenerate matrices even if they exist
    """
    if force:
        # Clear existing cache
        print(f"Clearing existing matrix cache in {cache_dir}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Precomputing {num_matrices} matrix pairs of size {size}x{size}...")
    for i in range(num_matrices):
        cache_file = os.path.join(cache_dir, f"matrix_pair_{i}.npz")
        
        if os.path.exists(cache_file) and not force:
            print(f"Matrix pair {i} already cached")
            continue
            
        # Generate matrices
        host_a, host_b = generate_matrix_pair(size, seed=42+i)
        
        # Save to cache
        np.savez(cache_file, a=host_a, b=host_b)
        print(f"Saved matrix pair {i} to cache")
        
    print("Matrix precomputation complete")

def load_cached_matrix(idx, size=4000, cache_dir=matrix_cache_dir):
    """Load a precomputed matrix pair from cache"""
    cache_file = os.path.join(cache_dir, f"matrix_pair_{idx}.npz")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Matrix pair {idx} not found in cache")
        
    data = np.load(cache_file)
    return data['a'], data['b']

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

def precompute_batches(train_images, train_labels, batch_size, num_epochs=200, 
                      cache_dir=batch_cache_dir, force=False):
    """Precompute batches for multiple epochs and cache them
    
    Args:
        train_images: Training images
        train_labels: Training labels
        batch_size: Size of each batch
        num_epochs: Number of epochs to precompute
        cache_dir: Directory to store cached batches
        force: If True, regenerate batches even if they exist
    """
    if force:
        # Clear existing cache
        print(f"Clearing existing batch cache in {cache_dir}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Precomputing batches for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        cache_file = os.path.join(cache_dir, f"batches_epoch_{epoch}.npz")
        
        if os.path.exists(cache_file) and not force:
            print(f"Batches for epoch {epoch} already cached")
            continue
            
        # Create shuffled batches
        N = train_images.shape[0]
        indices = np.random.permutation(N)
        shuffled_images = train_images[indices]
        shuffled_labels = train_labels[indices]
        
        # Split into batches
        num_batches = N // batch_size
        batch_x_list = []
        batch_y_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = shuffled_images[start_idx:end_idx]
            batch_y = shuffled_labels[start_idx:end_idx]
            batch_x_list.append(batch_x)
            batch_y_list.append(batch_y)
        
        # Save to cache
        np.savez(cache_file, 
                batch_x=np.stack(batch_x_list), 
                batch_y=np.stack(batch_y_list))
        print(f"Saved batches for epoch {epoch} to cache")
        
    print("Batch precomputation complete")

def create_cached_batch_loader(cache_dir=batch_cache_dir, prefetch=2):
    """Create a loader that accesses precomputed batches from cache
    
    Args:
        cache_dir: Directory with cached batches
        prefetch: Number of epochs to prefetch
        
    Returns:
        Tuple of (get_batches_fn, shutdown_fn)
    """
    batch_queue = Queue(maxsize=prefetch)
    stop_event = threading.Event()
    
    def loader_worker():
        epoch = 0
        while not stop_event.is_set():
            if batch_queue.qsize() < prefetch:
                cache_file = os.path.join(cache_dir, f"batches_epoch_{epoch}.npz")
                
                if not os.path.exists(cache_file):
                    print(f"Warning: No cache file for epoch {epoch}")
                    epoch = 0  # Reset to beginning if we reach the end
                    continue
                    
                # Load cached batches
                data = np.load(cache_file)
                batch_x_stack = data['batch_x']
                batch_y_stack = data['batch_y']
                
                # Convert to list of batch tuples
                num_batches = batch_x_stack.shape[0]
                batches = [(batch_x_stack[i], batch_y_stack[i]) for i in range(num_batches)]
                
                # Put in queue
                batch_queue.put((batches, 0.0, epoch))  # Zero timing since we're not measuring
                epoch += 1
            else:
                time.sleep(0.1)
    
    # Start thread
    loader_thread = threading.Thread(target=loader_worker)
    loader_thread.daemon = True
    loader_thread.start()
    
    def get_next_batches():
        if batch_queue.empty():
            print("Warning: Batch queue is empty, waiting for loader...")
        return batch_queue.get()
    
    def shutdown():
        stop_event.set()
        loader_thread.join(timeout=2.0)
        print("Batch loader shut down")
    
    return get_next_batches, shutdown

# ================================
# CNN MODEL DEFINITION
# ================================

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

# ================================
# CACHE MANAGEMENT FUNCTIONS
# ================================

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

# ================================
# ENHANCED MATRIX WORKER
# ================================

class EnhancedMatrixWorker:
    """Enhanced matrix multiplication worker using precomputed matrices"""
    def __init__(self, size=4000, num_matrices=5, max_concurrent_transfers=2, 
                cache_dir=matrix_cache_dir):
        self.size = size
        self.num_matrices = num_matrices
        self.bytes_per_matrix = size * size * 4  # 4 bytes for float32
        self.max_concurrent_transfers = max_concurrent_transfers
        self.cache_dir = cache_dir
        
        # Define compute queue and results queue
        self.compute_queue = Queue()
        self.output_queue = Queue()
        
        # Define simple matmul function
        self.jitted_fn = jax.jit(lambda x, y: jnp.dot(x, y))
        
        # Ensure compilation is done
        compiled_before, _ = load_compile_metadata_mm(size)
        if not compiled_before:
            print("Precompiling matrix multiplication function...")
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
    
    def transfer_matrix_pair(self, idx):
        """Load a precomputed matrix pair and transfer to device, measuring only transfer time"""
        try:
            # Load from cache
            host_a, host_b = load_cached_matrix(idx, self.size, self.cache_dir)
            
            # Transfer to device (measure only this part)
            transfer_start = time.time()
            dev_a = jax.device_put(host_a)
            dev_b = jax.device_put(host_b)
            dev_a.block_until_ready()
            dev_b.block_until_ready()
            transfer_end = time.time()
            
            transfer_duration = transfer_end - transfer_start
            print(f"Worker: Matrix pair {idx+1} transferred in {transfer_duration:.6f} seconds")
            
            return dev_a, dev_b, idx, transfer_duration
        except Exception as exc:
            print(f"Matrix transfer {idx} generated an exception: {exc}")
            raise
            
    def run(self):
        """Run the enhanced matrix computation with precomputed matrices"""
        print(f"\n=== Running Enhanced Matrix Multiplication with Precomputed Data ===")
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
                executor.submit(self.transfer_matrix_pair, i): i 
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
        print("\n=== Enhanced Matrix Multiplication Summary ===")
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

# ================================
# TRAINING FUNCTIONS
# ================================

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

# Helper decorator for timing
def timer_decorator(func):
    """Decorator for timing function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper

# ================================
# PRECOMPUTATION AND MAIN FUNCTIONS
# ================================

@timer_decorator
def precompute_all_data():
    """Precompute all data for experiments to separate generation from measurement"""
    print("\n=== Precomputation Phase: Generating and Caching All Data ===")
    
    # Load CIFAR-100 dataset once
    print("Loading CIFAR-100 dataset...")
    cifar_data, info = tfds.load(name="cifar100", batch_size=-1, data_dir=data_dir, with_info=True)
    cifar_data = tfds.as_numpy(cifar_data)
    train_data, test_data = cifar_data['train'], cifar_data['test']
    
    # Process full train set
    print("Preprocessing training data...")
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = preprocess_images(train_images)
    train_labels = one_hot(train_labels, 100)
    
    # Process full test set
    print("Preprocessing test data...")
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = preprocess_images(test_images)
    test_labels = one_hot(test_labels, 100)
    
    # Precompute matrices
    print("Precomputing matrices...")
    precompute_matrices(size=4000, num_matrices=200, force=False)
    
    # Precompute batches
    print("Precomputing training batches...")
    precompute_batches(train_images, train_labels, batch_size, num_epochs=num_epochs, force=False)
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

@timer_decorator
def main():
    """Main training function using precomputed cached data for clean timing"""
    print("\n=== Starting Enhanced CIFAR-100 CNN Training ===\n")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Phase 1: Model Initialization
    print("\n=== Phase 1: Model Initialization ===")
    init_start = time.time()
    params = cnn_model.init_params(random.PRNGKey(0))
    init_time = time.time() - init_start
    print(f"Model initialization time: {init_time:.4f} seconds")
    
    # Phase 2: Dataset loading (using precomputed data)
    print("\n=== Phase 2: Dataset Loading ===")
    data_load_start = time.time()
    
    # Load CIFAR-100 dataset (we'll only use this for evaluation metrics)
    print("Loading preprocessed dataset for evaluation...")
    cifar_data, info = tfds.load(name="cifar100", batch_size=-1, data_dir=data_dir, with_info=True)
    cifar_data = tfds.as_numpy(cifar_data)
    train_data, test_data = cifar_data['train'], cifar_data['test']
    
    # Process test set for evaluation
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = preprocess_images(test_images)
    test_labels = one_hot(test_labels, 100)
    
    data_load_time = time.time() - data_load_start
    print(f"Dataset loading time: {data_load_time:.4f} seconds")
    
    # Phase 3: Compilation
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
        sample_y = jnp.ones((batch_size, 100))
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
    
    # Phase 4: Initialize batch loader
    print("\n=== Phase 4: Initializing Cached Batch Loader ===")
    
    # Initialize the batch prefetcher with cached batches
    get_next_batches, shutdown_prefetcher = create_cached_batch_loader(
        cache_dir=batch_cache_dir, prefetch=2)
    
    # Phase 5: Setup Matrix Computation (True Overlapping)
    print("\n=== Phase 5: Setting Up Overlapping Matrix Computations ===")
    
    # Setup for matrix multiplication
    matrix_size = 4000
    num_matrices = 200
    
    # Create a function to load, transfer, and compute a single matrix multiplication
    def process_matrix_pair(idx):
        """Process a single matrix pair, measuring only transfer and computation"""
        # Load from cache
        try:
            host_a, host_b = load_cached_matrix(idx, matrix_size, matrix_cache_dir)
            
            # Transfer to device
            transfer_start = time.time()
            dev_a = jax.device_put(host_a)
            dev_b = jax.device_put(host_b)
            dev_a.block_until_ready()
            dev_b.block_until_ready()
            transfer_end = time.time()
            transfer_duration = transfer_end - transfer_start
            
            # Compute
            matmul_fn = jax.jit(lambda x, y: jnp.dot(x, y))
            compute_start = time.time()
            result = matmul_fn(dev_a, dev_b)
            result.block_until_ready()
            compute_end = time.time()
            compute_duration = compute_end - compute_start
            
            # Calculate FLOPS
            flops = 2 * matrix_size**3  # 2*N^3 FLOPs for matrix multiplication
            tflops = flops / compute_duration / 1e12
            
            return {
                'index': idx,
                'transfer_duration': transfer_duration,
                'compute_duration': compute_duration,
                'tflops': tflops
            }
        except Exception as e:
            print(f"Error processing matrix pair {idx}: {e}")
            return None
    
    # Create a thread-safe list to store matrix results
    matrix_results = []
    matrix_results_lock = threading.Lock()
    
    # Create a thread pool for matrix computations
    matrix_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    matrix_futures = []
    
    # Counter for tracking matrix computations
    matrix_idx = 0
    
    # Phase 6: Training loop with overlapping matrix computations
    print("\n=== Phase 6: Training Loop with Overlapping Matrix Computations ===")
    training_start = time.time()
    
    # Track timing statistics
    total_batch_load_time = 0
    total_compute_time = 0
    total_eval_time = 0
    
    # Track best accuracy
    best_test_acc = 0.0
    
    for epoch in range(min(num_epochs, 200)):
        epoch_start = time.time()
        
        # Get learning rate for this epoch
        current_step_size = get_step_size(epoch)
        
        # Submit a new matrix computation before each batch if we haven't reached the limit
        if matrix_idx < num_matrices:
            future = matrix_executor.submit(process_matrix_pair, matrix_idx)
            matrix_futures.append(future)
            matrix_idx += 1
            if matrix_idx % 10 == 0:
                print(f"Submitted matrix computation {matrix_idx}/{num_matrices}")
        
        # Get batches from cached loader
        batches, batch_load_time, prefetched_epoch = get_next_batches()
        total_batch_load_time += batch_load_time  # Should be near zero since we're not measuring generation
        
        # Training computation time - THIS IS WHAT WE'RE ACTUALLY MEASURING
        compute_start = time.time()
        batch_count = 0
        for x, y in batches:
            batch_count += 1
            # Use the JIT-compiled update function with current learning rate
            # This includes CPU to GPU transfer of the batch data
            params = update_jit(params, x, y, current_step_size)
            
            # Check if any matrix computations have completed
            completed_futures = []
            for future in matrix_futures:
                if future.done():
                    try:
                        result = future.result()
                        if result:
                            with matrix_results_lock:
                                matrix_results.append(result)
                        completed_futures.append(future)
                    except Exception as e:
                        print(f"Error getting matrix result: {e}")
                        completed_futures.append(future)
            
            # Remove completed futures
            for future in completed_futures:
                matrix_futures.remove(future)
                
            # Submit a new matrix computation if we haven't reached the limit
            if matrix_idx < num_matrices:
                future = matrix_executor.submit(process_matrix_pair, matrix_idx)
                matrix_futures.append(future)
                matrix_idx += 1
        
        # Make sure all operations are complete before timing
        jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), params)
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        # Evaluation computation time
        eval_start = time.time()
        
        # Evaluate on a subset for speed
        eval_subset_size = min(5000, test_images.shape[0])
        test_idx = np.random.choice(test_images.shape[0], eval_subset_size, replace=False)
        test_acc = accuracy(params, test_images[test_idx], test_labels[test_idx])
        
        # Make sure accuracy computation is complete
        test_acc = jax.block_until_ready(test_acc)
        eval_time = time.time() - eval_start
        total_eval_time += eval_time
                        
        # Track best performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        epoch_time = time.time() - epoch_start
        
        # Report matrix computation progress
        with matrix_results_lock:
            completed_matrices = len(matrix_results)
        
        if epoch % 5 == 0 or epoch == min(num_epochs, 200) - 1:
            print(f"Epoch {epoch}:")
            print(f"  Learning rate: {current_step_size:.5f}")
            print(f"  Total epoch time: {epoch_time:.4f} sec")
            print(f"  Training computation time: {compute_time:.4f} sec")
            print(f"  Evaluation time: {eval_time:.4f} sec")
            print(f"  Number of batches: {batch_count}")
            print(f"  Test set accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
            print(f"  Matrix computations: {completed_matrices}/{num_matrices} completed")
            if matrix_results:
                avg_matrix_time = sum(r['compute_duration'] for r in matrix_results) / len(matrix_results)
                avg_tflops = sum(r['tflops'] for r in matrix_results) / len(matrix_results)
                print(f"  Average matrix computation time: {avg_matrix_time:.4f} sec")
                print(f"  Average matrix performance: {avg_tflops:.2f} TFLOPS")
            print()
        else:
            # Print minimal info for other epochs
            print(f"Epoch {epoch}: test_acc={test_acc:.4f}, matrices={completed_matrices}/{num_matrices}, time={epoch_time:.2f}s")
    
    # Wait for any remaining matrix computations to complete
    print(f"Waiting for remaining {len(matrix_futures)} matrix computations to complete...")
    for future in concurrent.futures.as_completed(matrix_futures):
        try:
            result = future.result()
            if result:
                with matrix_results_lock:
                    matrix_results.append(result)
        except Exception as e:
            print(f"Error getting matrix result: {e}")
    
    # Clean up
    shutdown_prefetcher()
    matrix_executor.shutdown()
    training_time = time.time() - training_start
    
    # Sort matrix results by index
    with matrix_results_lock:
        matrix_results.sort(key=lambda x: x['index'])
        completed_matrices = len(matrix_results)
    
    # Print matrix computation statistics
    print("\n=== Matrix Computation Summary ===")
    print(f"Completed {completed_matrices}/{num_matrices} matrix computations")
    
    if matrix_results:
        avg_transfer = sum(r['transfer_duration'] for r in matrix_results) / len(matrix_results)
        avg_compute = sum(r['compute_duration'] for r in matrix_results) / len(matrix_results)
        avg_tflops = sum(r['tflops'] for r in matrix_results) / len(matrix_results)
        
        print(f"Average transfer time: {avg_transfer:.6f} seconds")
        print(f"Average compute time: {avg_compute:.6f} seconds")
        print(f"Average performance: {avg_tflops:.2f} TFLOPS")
    
    # Print training summary statistics
    print("\n=== Training Summary ===")
    print(f"Model initialization time: {init_time:.4f} seconds")
    print(f"Dataset loading time: {data_load_time:.4f} seconds")
    print(f"Function compilation time: {compilation_time:.4f} seconds")
    print(f"Training loop time: {training_time:.4f} seconds")
    print(f"  - Computation time: {total_compute_time:.4f} seconds")
    print(f"  - Evaluation time: {total_eval_time:.4f} seconds")
    print(f"Total time: {init_time + data_load_time + (0 if compiled_before else compilation_time) + training_time:.4f} seconds")
    print(f"Final test accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
    print(f"Best test accuracy: {best_test_acc:.6f} ({best_test_acc*100:.2f}%)")
    
    return {
        'params': params,
        'accuracy': best_test_acc,
        'training_time': training_time,
        'compute_time': total_compute_time,
        'matrix_results': matrix_results
    }

@timer_decorator
def main_mm():
    """Run matrix multiplication benchmark with precomputed matrices"""
    print("=== Enhanced Matrix Multiplication Benchmark ===")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Make sure matrices are precomputed
    print("Ensuring matrices are precomputed...")
    precompute_matrices(size=4000, num_matrices=200, force=False)
    
    # Create enhanced matrix worker
    worker = EnhancedMatrixWorker(
        size=4000, 
        num_matrices=200, 
        max_concurrent_transfers=4,
        cache_dir=matrix_cache_dir
    )
    
    # Run the benchmark measuring only transfer and computation time
    results = worker.run()
    
    print("\n=== Matrix Multiplication Benchmark Complete ===")
    return results

if __name__ == "__main__":
    # Uncomment this line to run the precomputation step separately
    # dataset = precompute_all_data()
    
    # Main training function with background matrix multiplication
    main()
    
    # Alternatively, run just the matrix multiplication benchmark
    # main_mm()