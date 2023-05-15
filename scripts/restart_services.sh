#!/usr/bin/env bash
# Restarts ALL services and removes ALL unprocessed audio
source /etc/batnet/batnet.conf
set -x
my_dir=$HOME/BatNET-Pi/scripts


sudo systemctl stop batnet_server.service
sudo systemctl stop batnet_recording.service

services=(chart_viewer.service
  spectrogram_viewer.service
  icecast2.service
  extraction.service
  batnet_recording.service
  batnet_log.service)

for i in  "${services[@]}";do
  sudo systemctl restart "${i}"
done

sudo systemctl start batnet_server.service
sleep 5

for i in {1..5}; do
  # We want to loop here (5*5seconds) until the server is running and listening on its port
  systemctl is-active --quiet batnet_server.service \
	  && grep 5050 <(netstat -tulpn 2>&1) \
	  && logger "[$0] batnet_server.service is running" \
	  && break

  sleep 5
done

# Let's check a final time to ensure the server is running
systemctl is-active --quiet batnet_server.service && grep 5050 <(netstat -tulpn 2>&1)
status=$?

if (( status != 0 )); then
  logger "[$0] Unable to start batnet_server.service... Looping until it start properly"

  until grep 5050 <(netstat -tulpn 2>&1);do
    sudo systemctl restart batnet_server.service
    sleep 45
  done
fi

# Finally start the batnet_analysis.service
sudo systemctl restart batnet_analysis.service
