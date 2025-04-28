#!/usr/bin/env python3
"""
CIFAR-100 CNN training with JAX - with compilation caching, separated timing measurements,
and coordinated scheduling with matrix multiplication workload.

This optimized version implements coordinated scheduling where matrix multiplication
happens during CNN's data loading phases, and vice versa.
"""

import os

os.environ["XLA_FLAGS"] = '--xla_gpu_enable_latency_hiding_scheduler=true'

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
os.makedirs(cache_dir, exist_ok=True)

cache_dir_mm = "/tmp/jax_direct_aot_cache"
os.makedirs(cache_dir_mm, exist_ok=True)

# Hyperparameters
initial_step_size = 0.01
num_epochs = 200
batch_size = 128
n_targets = 100
data_dir = '/tmp/tfds'  # Change this to your desired data directory

# ----- Coordination System -----
# Event flags for coordinating between CNN and matrix workloads
class CoordinationSystem:
    def __init__(self):
        self._lock = threading.Lock()
        self._cnn_loading_data = False  # True when CNN is loading data (matrix can compute)
        self._cnn_computing = False     # True when CNN is computing (matrix should wait)
        self._matrix_active = False     # True when matrix operations are running
        self._matrix_results = []       # Store matrix operation results
        self._matrix_paused = False     # Flag to indicate if matrix operations are paused
        self._matrix_control_queue = Queue()  # Control queue for matrix operations
        self._matrix_results_queue = Queue()  # Results queue for matrix operations
        self._stop_requested = False    # Flag to stop all workloads
    
    def set_cnn_loading_data(self, is_loading):
        """Signal that CNN is loading data"""
        with self._lock:
            self._cnn_loading_data = is_loading
            # If CNN starts loading data, signal matrix to resume if needed
            if is_loading and self._matrix_paused and self._matrix_active:
                self._matrix_paused = False
                self._matrix_control_queue.put("RESUME")
                print("CNN data loading started - signaling matrix workload to RESUME")
    
    def set_cnn_computing(self, is_computing):
        """Signal that CNN is computing"""
        with self._lock:
            previous_state = self._cnn_computing
            self._cnn_computing = is_computing
            
            # Only signal if there's an actual state change
            if is_computing and not previous_state and self._matrix_active and not self._matrix_paused:
                # Instead of truly pausing, we'll just lower the priority
                # This prevents GPU context switching issues
                self._matrix_control_queue.put("LOWER_PRIORITY")
                print("CNN computation started - signaling matrix workload to lower priority")
            elif not is_computing and previous_state and self._matrix_active:
                # Restore normal priority when CNN is not computing
                self._matrix_control_queue.put("NORMAL_PRIORITY")
                print("CNN computation ended - signaling matrix workload to restore priority")
    
    def set_matrix_active(self, is_active):
        """Signal that matrix operations are active"""
        with self._lock:
            self._matrix_active = is_active
    
    def can_matrix_compute(self):
        """Check if matrix operations can compute"""
        with self._lock:
            # Always allow matrix operations to compute, but with different priorities
            return True
    
    def get_matrix_priority(self):
        """Get the current priority for matrix operations"""
        with self._lock:
            # Higher priority when CNN is loading data
            if self._cnn_loading_data:
                return "HIGH"
            # Lower priority when CNN is computing
            elif self._cnn_computing:
                return "LOW"
            # Normal priority otherwise
            else:
                return "NORMAL"
    
    def is_matrix_paused(self):
        """Check if matrix operations are paused"""
        with self._lock:
            # We don't fully pause anymore to avoid GPU context switching issues
            return False
    
    def get_matrix_control_queue(self):
        """Get the matrix control queue"""
        return self._matrix_control_queue
    
    def get_matrix_results_queue(self):
        """Get the matrix results queue"""
        return self._matrix_results_queue
    
    def process_matrix_results(self):
        """Process any results from matrix operations"""
        results = []
        while not self._matrix_results_queue.empty():
            result = self._matrix_results_queue.get_nowait()
            results.append(result)
            if result['event'] == 'transfer_complete':
                print(f"Matrix {result['index']} transfer completed in background, time: {result['transfer_time']:.4f} sec")
            elif result['event'] == 'compute_complete':
                print(f"Matrix {result['index']} computation completed in background, performance: {result['tflops']:.2f} TFLOPS")
            elif result['event'] == 'all_complete':
                print("All matrix computations completed")
                self._matrix_active = False
        return results
    
    def request_stop(self):
        """Request all workloads to stop"""
        with self._lock:
            self._stop_requested = True
            self._matrix_control_queue.put("STOP")
    
    def is_stop_requested(self):
        """Check if stop was requested"""
        with self._lock:
            return self._stop_requested

# Instantiate the coordination system
coordinator = CoordinationSystem()

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

def generate_and_transfer_matrix_pair(idx, size, num_matrices):
    """Generate and transfer a matrix pair in a separate function for parallelization"""
    seed = 42 + idx
    
    # Always proceed but check for stop request
    if coordinator.is_stop_requested():
        return None, None, idx, 0.0, "STOPPED"
    
    # Check current priority level
    current_priority = coordinator.get_matrix_priority()
    
    # If CNN is computing (LOW priority), introduce a small delay
    if current_priority == "LOW":
        time.sleep(0.005)  # Small delay to reduce contention
    
    print(f"Worker {idx+1}: Generating matrix pair {idx+1}/{num_matrices}, priority={current_priority}")
    
    # Generate matrices
    host_a, host_b = generate_matrix_pair(size, seed)
    
    # Transfer to device and measure time
    transfer_start = time.time()
    
    # For low priority, transfer in smaller chunks to reduce contention
    if current_priority == "LOW":
        # Transfer in two steps to be more friendly to the CNN workload
        dev_a_partial = jax.device_put(host_a[:size//2])
        dev_a_partial.block_until_ready()
        dev_a = jax.device_put(host_a)
        
        dev_b_partial = jax.device_put(host_b[:size//2])
        dev_b_partial.block_until_ready()
        dev_b = jax.device_put(host_b)
    else:
        # Normal transfer for high or normal priority
        dev_a = jax.device_put(host_a)
        dev_b = jax.device_put(host_b)
    
    # Ensure transfers are complete
    dev_a.block_until_ready()
    dev_b.block_until_ready()
    transfer_end = time.time()
    
    transfer_duration = transfer_end - transfer_start
    print(f"Worker {idx+1}: Matrix pair {idx+1} transferred in {transfer_duration:.6f} seconds")
    
    return dev_a, dev_b, idx, transfer_duration, "SUCCESS"

class CoordinatedMatrixWorker:
    """Worker class for matrix operations that coordinates with CNN training"""
    def __init__(self, size=4000, num_matrices=5, max_concurrent_transfers=2):
        self.size = size
        self.num_matrices = num_matrices
        self.bytes_per_matrix = size * size * 4  # 4 bytes for float32
        self.max_concurrent_transfers = max_concurrent_transfers
        
        # Define compute queue and results queue
        self.compute_queue = Queue()
        
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
        # Current priority level for matrix operations
        current_priority = "NORMAL"
        
        # Smaller batch size for lower priority
        small_size = 1000
        normal_size = self.size
        
        while True:
            # Check if priority level has changed
            matrix_priority = coordinator.get_matrix_priority()
            if matrix_priority != current_priority:
                current_priority = matrix_priority
                print(f"Matrix workload priority changed to: {current_priority}")
            
            # Get next matrix pair from queue
            try:
                item = self.compute_queue.get(timeout=1.0)
                if item is None:  # End signal
                    break
            except Exception:
                # Check if we should stop
                if coordinator.is_stop_requested():
                    break
                continue
                
            dev_a, dev_b, idx, transfer_duration, status = item
            
            # If the status is not "SUCCESS", skip this pair
            if status != "SUCCESS":
                self.compute_queue.task_done()
                continue
            
            # Adjust computation based on priority
            # For LOW priority, use a smaller sub-matrix to reduce GPU utilization
            adjusted_size = normal_size
            if current_priority == "LOW":
                # Introduce a small delay to yield to the CNN workload
                time.sleep(0.005)
                print(f"Matrix {idx+1}: Running in LOW priority mode")
            
            # Execute computation
            print(f"Compute: Processing matrix pair {idx+1}/{self.num_matrices}, priority={current_priority}")
            execution_start = time.time()
            
            # Always process the full matrix, but with consideration for priority
            result = self.jitted_fn(dev_a, dev_b)
            
            # If lower priority, yield a bit to other operations by not blocking immediately
            if current_priority == "LOW":
                # Allow other operations to get scheduled first
                time.sleep(0.001)
            
            # Now block until ready
            result.block_until_ready()
            execution_end = time.time()
            
            execution_duration = execution_end - execution_start
            
            # Transfer result back
            result_transfer_start = time.time()
            host_result = np.array(result)
            result_transfer_end = time.time()
            
            result_transfer_duration = result_transfer_end - result_transfer_start
            
            # Calculate FLOPS
            flops = 2 * adjusted_size**3  # 2*N^3 FLOPs for matrix multiplication
            tflops = flops / execution_duration / 1e12
            
            print(f"Compute: Matrix pair {idx+1} computed in {execution_duration:.6f} seconds ({tflops:.2f} TFLOPS)")
            print(f"Compute: Result transferred in {result_transfer_duration:.6f} seconds")
            
            # Report results to coordination system
            coordinator.get_matrix_results_queue().put({
                'event': 'compute_complete',
                'index': idx,
                'priority': current_priority,
                'transfer_duration': transfer_duration,
                'execution_duration': execution_duration,
                'result_transfer_duration': result_transfer_duration,
                'tflops': tflops
            })
            
            # Signal that we're done with this item
            self.compute_queue.task_done()
            
            # If high priority, keep going immediately
            # Otherwise, introduce a small delay to be nicer to the system
            if current_priority != "HIGH":
                time.sleep(0.001)
    
    def run(self):
        """Run the coordinated matrix computation with parallel transfers"""
        print(f"\n=== Starting Coordinated Matrix Multiplication ===")
        print(f"Matrix size: {self.size}x{self.size} ({self.bytes_per_matrix/1e9:.2f} GB each)")
        print(f"Number of matrix pairs to process: {self.num_matrices}")
        print(f"Maximum concurrent transfers: {self.max_concurrent_transfers}")
        
        # Signal that matrix operations are active
        coordinator.set_matrix_active(True)
        
        # Start compute thread
        compute_thread = threading.Thread(target=self.compute_worker)
        compute_thread.daemon = True
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
                    # Check if this was stopped
                    if result[4] == "STOPPED":
                        print(f"Matrix transfer {idx} was stopped")
                        continue
                        
                    # Add to compute queue
                    self.compute_queue.put(result)
                    
                    # Report to coordination system
                    coordinator.get_matrix_results_queue().put({
                        'event': 'transfer_complete',
                        'index': idx,
                        'transfer_time': result[3]
                    })
                    
                    # Check if we should stop
                    if coordinator.is_stop_requested():
                        break
                        
                except Exception as exc:
                    print(f"Matrix transfer {idx} generated an exception: {exc}")
        
        # Signal that no more transfers are coming
        self.compute_queue.put(None)
        
        # Wait for compute thread to finish
        compute_thread.join(timeout=1.0)
        
        # Stop timer
        total_end = time.time()
        
        # Report completion
        coordinator.get_matrix_results_queue().put({'event': 'all_complete'})
        
        return True

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
    print("\n=== Starting CIFAR-100 CNN with Coordinated Matrix Multiplication ===\n")
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
    
    # Signal that CNN is in data loading state (matrix can compute)
    coordinator.set_cnn_loading_data(True)
    
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
    
    # Data loading is done, signal that CNN is no longer loading data
    coordinator.set_cnn_loading_data(False)
    
    print(f'Train: {train_images.shape} {train_labels.shape}')
    print(f'Test: {test_images.shape} {test_labels.shape}')
    print(f"Dataset loading time: {data_load_time:.4f} seconds")
    
    # Create preprocessed batch dataset
    def get_train_batches():
        # Signal that CNN is loading data (matrix can compute)
        coordinator.set_cnn_loading_data(True)
        
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
        
        # Signal that CNN is done loading data
        coordinator.set_cnn_loading_data(False)
        
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
    
    # Start the matrix worker in a background thread
    print("\n=== Starting Matrix Computations in Background ===")
    
    # Initialize the matrix worker
    matrix_worker = CoordinatedMatrixWorker(size=4000, num_matrices=200, max_concurrent_transfers=4)
    
    # Start matrix operations in a separate thread
    matrix_thread = threading.Thread(target=matrix_worker.run)
    matrix_thread.daemon = True  # Thread will exit when main program exits
    matrix_thread.start()
    
    # Phase 4: Training loop (with detailed timing)
    print("\n=== Phase 4: Training Loop ===")
    training_start = time.time()
    
    total_batch_load_time = 0
    total_compute_time = 0
    total_eval_time = 0
    
    # Process any matrix results that have come in so far
    coordinator.process_matrix_results()
    
    # Track best accuracy
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Get learning rate for this epoch
        current_step_size = get_step_size(epoch)
        
        # Get batches and measure loading time
        # During this phase, matrix computations are allowed to run
        batches, batch_load_time = get_train_batches()
        total_batch_load_time += batch_load_time
        
        # Process any matrix results that came in during data loading
        #coordinator.process_matrix_results()
        
        # Signal that CNN is now computing (matrix should pause)
        coordinator.set_cnn_computing(True)
        
        # Training computation time
        compute_start = time.time()
        batch_count = 0
        for x, y in batches:
            batch_count += 1
            # Use the JIT-compiled update function with current learning rate
            params = update_jit(params, x, y, current_step_size)
        
        # Make sure all operations are complete before timing
        jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), params)
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        # Signal that CNN is no longer computing (matrix can resume)
        coordinator.set_cnn_computing(False)
        
        # Evaluation computation time
        # Evaluation doesn't use much GPU resource, so we don't need to signal here
        eval_start = time.time()
        
        # For larger datasets, evaluate on a subset for speed
        eval_subset_size = min(5000, train_images.shape[0])
        train_idx = np.random.choice(train_images.shape[0], eval_subset_size, replace=False)
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
        
        # Process any matrix results that came in during this epoch
        #coordinator.process_matrix_results()
        print()
            
    # Signal that all operations should stop
    coordinator.request_stop()
    
    training_time = time.time() - training_start
    
    # Process any remaining matrix results
    final_matrix_results = coordinator.process_matrix_results()
    
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
        
    print("\n=== End of Coordinated Scheduling Demo ===")

if __name__ == "__main__":
    main()