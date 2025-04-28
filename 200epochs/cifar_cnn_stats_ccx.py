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

# Set environment variables for JAX
os.environ['JAX_ENABLE_X64'] = '0'  # Ensure we use float32 by default
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'  # Enable persistent compilation cache
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cifar100_cnn_cache'  # Set cache directory
os.environ['JAX_COMPILATION_CACHE_WRITE_ON_COMPILE'] = '1'  # Write cache immediately

# Optional: Set for detailed JAX logging
# os.environ['JAX_LOG_COMPILES'] = '1'  # Log compilation events

# Ensure TF does not see GPU and grab all GPU memory
tf.config.set_visible_devices([], device_type='GPU')

# Create cache directory for our own caching mechanism
cache_dir = "/tmp/jax_cifar100_cnn_aot_cache"
os.makedirs(cache_dir, exist_ok=True)

# Hyperparameters
initial_step_size = 0.01
num_epochs = 1000
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

if __name__ == "__main__":
    main()
