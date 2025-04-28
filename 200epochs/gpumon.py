import pynvml
import time
import csv
import sys
from datetime import datetime

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU

# Set up output
output_file = f"gpu_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
use_file = len(sys.argv) > 1 and sys.argv[1] == "--file"

if use_file:
    f = open(output_file, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['timestamp_ms', 'gpu_util', 'mem_util', 'mem_used', 'mem_total'])
else:
    print("timestamp_ms,gpu_util,mem_util,mem_used,mem_total")

# Get starting time
start_time = time.perf_counter()
next_sample_time = start_time
last_flush_time = start_time

try:
    while True:
        # Calculate when the next sample should be taken
        next_sample_time += 0.01  # 10ms intervals
        
        # Get metrics
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Calculate elapsed time
        current_time = time.perf_counter()
        elapsed_ms = int((current_time - start_time) * 1000)
        
        # Prepare row data
        row = [elapsed_ms, util.gpu, util.memory, mem_info.used, mem_info.total]
        
        # Output data
        if use_file:
            writer.writerow(row)
            
            # Flush file every second
            if current_time - last_flush_time >= 1.0:
                f.flush()
                last_flush_time = current_time
        else:
            print(','.join(map(str, row)))
        
        # Sleep until next sample time
        sleep_time = next_sample_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # If we're behind schedule, reset the next sample time
            next_sample_time = time.perf_counter()
            
except KeyboardInterrupt:
    pass
finally:
    pynvml.nvmlShutdown()
    if use_file:
        f.close()
        print(f"Data saved to {output_file}")
