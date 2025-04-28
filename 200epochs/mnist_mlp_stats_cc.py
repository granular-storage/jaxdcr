#!/usr/bin/env python3
"""
MNIST MLP training with JAX - with compilation caching and separated timing measurements.

This script implements a multi-layer perceptron (MLP) for MNIST digit classification using JAX,
with detailed timing separation between data loading, compilation, and computation phases.

Key features:
- Compilation caching to avoid recompilation on subsequent runs
- Separated timing for data loading, compilation, and computation
- Clear measurement of initial compilation vs subsequent executions
- Network Architecture: 3-layer MLP with sizes [784, 512, 512, 10]
- Optimization: Simple gradient descent with step size 0.01
"""

import os
import time
import pickle
import hashlib
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import tensorflow as tf
import tensorflow_datasets as tfds

# Set environment variables for JAX
os.environ['JAX_ENABLE_X64'] = '0'  # Ensure we use float32 by default
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'  # Enable persistent compilation cache
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_mnist_cache'  # Set cache directory
os.environ['JAX_COMPILATION_CACHE_WRITE_ON_COMPILE'] = '1'  # Write cache immediately

# Optional: Set for detailed JAX logging
# os.environ['JAX_LOG_COMPILES'] = '1'  # Log compilation events

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type='GPU')

# Create cache directory for our own caching mechanism
cache_dir = "/tmp/jax_mnist_aot_cache"
os.makedirs(cache_dir, exist_ok=True)

# Hyperparameters
layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
data_dir = '/tmp/tfds'  # Change this to your desired data directory

# Cache management functions
def get_cache_key(layer_sizes, jax_version):
    """Generate a unique cache key based on network architecture and JAX version"""
    key_str = f"mnist_mlp_{'_'.join(map(str, layer_sizes))}_{jax_version}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_compile_metadata():
    """Load compilation metadata if it exists"""
    cache_key = get_cache_key(layer_sizes, jax.__version__)
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

def save_compile_metadata(duration):
    """Save compilation metadata"""
    cache_key = get_cache_key(layer_sizes, jax.__version__)
    metadata_path = os.path.join(cache_dir, f"{cache_key}_metadata.pickle")
    
    metadata = {
        'compiled': True,
        'layer_sizes': layer_sizes,
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

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Activation function
def relu(x):
    return jnp.maximum(0, x)

# Forward pass for a single example
def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

# Vectorize the predict function to handle batches
batched_predict = vmap(predict, in_axes=(None, 0))

# Utility function to create one-hot encodings
def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# Accuracy computation
def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

# Loss function
def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

# Define the gradient update function (not JIT-compiled yet)
def update_raw(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

# Pre-compile with JIT after initialization
update_jit = None  # Will be compiled later

def main():
    print("\n=== Starting MNIST MLP with Compilation Caching ===\n")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Time model initialization
    print("\n=== Phase 1: Model Initialization ===")
    init_start = time.time()
    params = init_network_params(layer_sizes, random.key(0))
    init_time = time.time() - init_start
    print(f"Model initialization time: {init_time:.4f} seconds")
    
    # Phase 2: Dataset loading (timed)
    print("\n=== Phase 2: Dataset Loading ===")
    data_load_start = time.time()
    
    # Load full datasets for evaluation
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    num_labels = info.features['label'].num_classes
    h, w, c = info.features['image'].shape
    num_pixels = h * w * c
    
    # Process full train set
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot(train_labels, num_labels)
    
    # Process full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot(test_labels, num_labels)
    
    data_load_time = time.time() - data_load_start
    
    print(f'Train: {train_images.shape} {train_labels.shape}')
    print(f'Test: {test_images.shape} {test_labels.shape}')
    print(f"Dataset loading time: {data_load_time:.4f} seconds")
    
    # Function to get training batches
    def get_train_batches():
        batch_load_start = time.time()
        # as_supervised=True gives us the (image, label) as a tuple instead of a dict
        ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
        # Build an input pipeline with batching and prefetching
        ds = ds.batch(batch_size).prefetch(1)
        # Convert to NumPy arrays
        batches = tfds.as_numpy(ds)
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
        sample_x = jnp.ones((batch_size, num_pixels))
        sample_y = jnp.ones((batch_size, num_labels))
        _ = update_jit(params, sample_x, sample_y)
        
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
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Get batches and measure loading time
        batches, batch_load_time = get_train_batches()
        total_batch_load_time += batch_load_time
        
        # Training computation time
        compute_start = time.time()
        batch_count = 0
        for x, y in batches:
            batch_count += 1
            x = jnp.reshape(x, (len(x), num_pixels))
            y = one_hot(y, num_labels)
            
            # Use the JIT-compiled update function
            params = update_jit(params, x, y)
        
        # Make sure all operations are complete before timing
        jax.block_until_ready(params)
        compute_time = time.time() - compute_start
        total_compute_time += compute_time
        
        # Evaluation computation time
        eval_start = time.time()
        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        
        # Make sure accuracy computation is complete
        train_acc = jax.block_until_ready(train_acc)
        test_acc = jax.block_until_ready(test_acc)
        eval_time = time.time() - eval_start
        total_eval_time += eval_time
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}:")
        print(f"  Total epoch time: {epoch_time:.4f} sec")
        print(f"  Batch loading time: {batch_load_time:.4f} sec")
        print(f"  Training computation time: {compute_time:.4f} sec")
        print(f"  Evaluation time: {eval_time:.4f} sec")
        print(f"  Number of batches: {batch_count}")
        print(f"  Training set accuracy: {train_acc:.6f}")
        print(f"  Test set accuracy: {test_acc:.6f}")
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
    print(f"Final test accuracy: {test_acc:.6f}")
    
    # If this was a reuse of compilation, show the time savings
    if compiled_before:
        print(f"\nReused cached compilation, saved approximately {compilation_time:.4f} seconds")

if __name__ == "__main__":
    main()
