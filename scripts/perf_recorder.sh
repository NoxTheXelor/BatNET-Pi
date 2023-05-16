#!/usr/bin/env bash
# record ram and cpu usage
# set -x
source /etc/batnet/batnet.conf
name=$(date "+%d-%m-%Y")
output_file=$HOME/BatNET-Pi/perf_logs/batnet_perf_$name.csv
max_file=$HOME/BatNET-Pi/perf_logs/max.csv

# Create the output file with column headers if it doesn't exist
if [ ! -f ${output_file} ]; then
  #setting up perf file
  touch ${output_file} && chmod g+rw ${output_file}
  echo -e "Timestamp,Available(MB),Used(MB),Swap_Free(MB),Swap_Used(MB),CPU(%),Disk(%)" > ${output_file}
fi

if [ ! -f ${max_file} ]; then
  #gattering total ram, swap
  touch ${max_file} && chmod g+rw ${max_file}
  echo -e "Memory_Total(MB),Swap_Total(MB)" > ${max_file}
  MEMORY_TOTAL=$(free -m | awk 'NR==2{printf "%.2f", $2 }')
  SWAP_TOTAL=$(free -m | awk 'NR==3{printf "%.2f", $2 }')
  echo "$MEMORY_TOTAL,$SWAP_TOTAL" >> ${max_file}
fi

while true; do
  # Get current date and time
  timestamp=$(date +"%d-%m %H:%M:%S")

  # Use top to get CPU and memory usage
  MEMORY_USED=$(free -m | awk 'NR==2{printf "%.2f", $3 }')
  MEMORY_AVAILABLE=$(free -m | awk 'NR==2{printf "%.2f", $7 }')
  SWAP_USED=$(free -m | awk 'NR==3{printf "%.2f", $3 }')
  SWAP_FREE=$(free -m | awk 'NR==3{printf "%.2f", $4 }')
  CPU=$(top -bn1 | grep '%Cpu' | tail -1 | grep -P '(....|...) id,'|awk -v n_proc=$(nproc) '{print 100-($8/n_proc)}' )
  DISK=$(df -h | awk '$NF=="/"{printf "%s\n", $5}')
  echo "$timestamp,$MEMORY_AVAILABLE,$MEMORY_USED,$SWAP_FREE,$SWAP_USED,$CPU,$DISK" >> ${output_file}
  sleep 0.25  #approx 2 record/sec

done
