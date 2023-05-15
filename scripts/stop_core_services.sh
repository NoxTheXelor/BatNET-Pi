#!/usr/bin/env bash
# Restarts ALL services and removes ALL unprocessed audio


services=(batnet_recording.service
custom_recording.service
batnet_analysis.service
batnet_server.service
chart_viewer.service
extraction.service
spectrogram_viewer.service)

for i in  "${services[@]}";do
  sudo systemctl stop  ${i}
done
sudo rm -rf ${RECS_DIR}/$(date +%B-%Y/%d-%A)/*
