#!/usr/bin/env bash
# record ram and cpu usage
# set -x
source /etc/birdnet/birdnet.conf
output_file="/home/bird/BirdNET-Pi/perf_logs/perf.csv"

# Create the output file with column headers if it doesn't exist
if [ ! -f ${output_file} ]; then
  touch ${output_file} && chmod g+rw ${output_file}
  echo -e "Timestamp,Memory_Total(MB),Available(MB),Used(MB),Swap_Free(MB),Swap_Used(MB),CPU(%),Disk(%)\n" > ${output_file}
fi
while true; do
  # Get current date and time
  timestamp=$(date +"%d-%m %H:%M:%S")

  # Use top to get CPU and memory usage
  MEMORY_TOTAL=$(free -m | awk 'NR==2{printf "%.2f", $2 }')
  MEMORY_USED=$(free -m | awk 'NR==2{printf "%.2f", $3 }')
  MEMORY_AVAILABLE=$(free -m | awk 'NR==2{printf "%.2f", $7 }')
  SWAP_USED=$(free -m | awk 'NR==3{printf "%.2f", $3 }')
  SWAP_FREE=$(free -m | awk 'NR==3{printf "%.2f", $4 }')
  CPU=$(top -bn1 | grep '%Cpu' | tail -1 | grep -P '(....|...) id,'|awk -v n_proc=$(nproc) '{print 100-($8/n_proc)}' )
  DISK=$(df -h | awk '$NF=="/"{printf "%s\n", $5}')
  echo "$timestamp,$MEMORY_TOTAL,$MEMORY_AVAILABLE,$MEMORY_USED,$SWAP_FREE,$SWAP_USED,$CPU,$DISK" >> ${output_file}
  sleep 0.25  #approx 2 record/sec

done
