import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_gpu_utilization(data_file, output_file=None, time_unit='s'):
    # Read the data and convert types
    df = pd.read_csv(data_file, header=None, 
                     names=['timestamp', 'gpu_util', 'mem_util', 'mem_used', 'mem_total'])
    
    # Convert columns to numeric types
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    df = df.dropna()
    
    # Convert timestamp from ms to seconds for display
    start_time = df['timestamp'].iloc[0]
    df['time_sec'] = (df['timestamp'] - start_time) / 1000  # Convert to seconds from start
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot GPU Compute Throughput Utilization
    ax1.plot(df['time_sec'], df['gpu_util'], 'b-', linewidth=1)
    ax1.set_ylabel('Compute throughput (%)')
    ax1.set_ylim(0, 100)
    avg_gpu = df['gpu_util'].mean()
    ax1.axhline(y=avg_gpu, color='r', linestyle='--', label='average')
    ax1.set_title('(a) GPU Compute Throughput Utilization')
    ax1.legend()
    
    # Plot GPU Memory Bandwidth Utilization
    ax2.plot(df['time_sec'], df['mem_util'], 'b-', linewidth=1)
    ax2.set_xlabel(f'Time (s)')  # Always show in seconds
    ax2.set_ylabel('Memory bandwidth usage (%)')
    ax2.set_ylim(0, 100)
    avg_mem = df['mem_util'].mean()
    ax2.axhline(y=avg_mem, color='r', linestyle='--', label='average')
    ax2.set_title('(b) GPU Memory Bandwidth Utilization')
    ax2.legend()
    
    # Add some info about the data range
    total_duration = df['time_sec'].iloc[-1]
    data_points = len(df)
    sampling_rate = 1000 / (df['timestamp'].diff().mean())  # Convert to Hz
    
    plt.figtext(0.5, 0.01, 
                f"Total duration: {total_duration:.2f}s | Data points: {data_points} | " +
                f"Sampling rate: {sampling_rate:.1f}Hz",
                ha="center", fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Make room for the text at the bottom
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GPU utilization from monitoring data.')
    parser.add_argument('data_file', help='Path to the CSV data file')
    parser.add_argument('-o', '--output', help='Output image file path (optional)')
    parser.add_argument('-t', '--time_unit', default='sec', help='Time unit label (default: sec)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: File {args.data_file} not found.")
        exit(1)
        
    plot_gpu_utilization(args.data_file, args.output, args.time_unit)
