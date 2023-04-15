#!/usr/bin/env bash
# record ram and cpu usage
# set -x
source /etc/birdnet/birdnet.conf
output_file="/home/bird/BirdNET-Pi/perf_logs/perf.csv"

# Create the output file with column headers if it doesn't exist
if [ ! -f ${output_file} ]; then
  touch ${output_file} && chmod g+rw ${output_file}
  echo -e "Timestamp,Memory_Total,Free,Used,Swap_Free,Swap_Used,Swap_Total,CPU,Disk\n" > ${output_file}
fi
while true; do
  # Get current date and time
  timestamp=$(date +"%d-%m %H:%M:%S")

  # Use top to get CPU and memory usage
  MEMORY_TOTAL=$(free -m | awk 'NR==1{printf "%.2f MB,", $2 }')
  MEMORY_USED=$(free -m | awk 'NR==2{printf "%.2f MB,", $3 }')
  MEMORY_FREE=$(free -m | awk 'NR==2{printf "%.2f MB,", $4 }')
  SWAP_TOTAL=$(free -m | awk 'NR==4{printf "%.2f MB,", $2 }')
  SWAP_USED=$(free -m | awk 'NR==4{printf "%.2f MB,", $3 }')
  SWAP_FREE=$(free -m | awk 'NR==4{printf "%.2f MB,", $4 }')
  CPU=$(top -bn1 | grep load | awk '{printf "%.2f%%,", $(NF-2)}')
  DISK=$(df -h | awk '$NF=="/"{printf "%s\n", $5}')
  echo "$timestamp,$MEMORY_TOTAL,$MEMORY_FREE,$MEMORY_USED,$SWAP_FREE,$SWAP_USED,$SWAP_TOTAL,$CPU,$DISK" >> ${output_file}
  sleep 0.25  #approx 2 record/sec

done
