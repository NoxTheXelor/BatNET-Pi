#!/usr/bin/env bash
# record ram and cpu usage
# set -x

output_file=$HOME/BirdNET-Pi/perf_logs/perf.csv

# Create the output file with column headers if it doesn't exist
if [ ! -e "$output_file" ]; then
  echo -e "---Timestamp---\tCPU (%)\tMem used (MiB)\tMem free (MiB)\tSwap used (MiB)\tSwap free (MiB)" >> "$output_file"
fi

# Get current date and time
timestamp=$(date +"%d-%m-%y %H:%M:%S")

# Use top to get CPU and memory usage
top_output=$(top -bn1 | grep "Cpu\|MiB Mem\|MiB Swap")

# Extract relevant data
cpu=$(echo "$top_output" | awk '{print $2}')
mem_used=$(echo "$top_output" | awk '{print $6}')
mem_free=$(echo "$top_output" | awk '{print $8}')
swap_used=$(echo "$top_output" | awk '{print $16}')
swap_free=$(echo "$top_output" | awk '{print $18}')

# Write data to the output file
echo -e "$timestamp,\t$cpu,\t$mem_used,\t$mem_free,\t$swap_used,\t$swap_free" >> "$output_file"

# Wait for 1 second before collecting the next data
