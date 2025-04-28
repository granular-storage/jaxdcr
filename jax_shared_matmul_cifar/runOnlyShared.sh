#!/bin/bash

# List of Python scripts to run
scripts=(
  "jax_onlyshared_batches_prefetch_matmul_cifar_1_1.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_1_2.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_2_1.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_2_2.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_3_1.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_4_1.py"
  "jax_onlyshared_batches_prefetch_matmul_cifar_4_2.py"
)

# Run each script and redirect output to corresponding .out file
for script in "${scripts[@]}"; do
  output_file="${script%.py}.out"
  echo "Running $script, output to $output_file"
  python3.12 "$script" > "$output_file" 2>&1
  echo "Finished running $script"
done

echo "All scripts completed"
