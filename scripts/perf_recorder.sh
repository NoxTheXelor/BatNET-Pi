#!/usr/bin/env bash
# record ram and cpu usage
# set -x
source /etc/birdnet/birdnet.conf
output_file="$HOME/BirdNET-Pi/perf_logs/perf.csv"

# Create the output file with column headers if it doesn't exist
if [ ! -f ${output_file} ]; then
  sudo touch ${output_file} && chmod g+rw ${output_file}
  sudo echo -e "\tTimestamp\tCPU \tMem used \tMem free \tSwap used \tSwap free " > ${output_file}
fi
while true; do
  # Get current date and time
  timestamp=$(date +"%d-%m %H:%M:%S")

  # Use top to get CPU and memory usage
  top_output=$(top -bn1 | grep "Cpu\|MiB Mem\|MiB Swap")

  # Extract relevant data
  cpu=$(echo "$top_output" | awk '{print $2}')
  mem_used=$(echo "$top_output" | awk '{print $6}')
  mem_free=$(echo "$top_output" | awk '{print $8}')
  swap_used=$(echo "$top_output" | awk '{print $16}')
  swap_free=$(echo "$top_output" | awk '{print $18}')

  # Write data to the output file
  echo -e "$timestamp,\t$cpu(%),\t$mem_used(MiB),\t$mem_free(MiB),\t$swap_used(MiB),\t$swap_free(MiB)" >> ${output_file}
  sleep 0.25
done