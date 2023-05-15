#!/usr/bin/env bash
# Make sox spectrogram
source /etc/batnet/batnet.conf
analyzing_now="$(cat $HOME/BatNET-Pi/analyzing_now.txt)"
spectrogram_png=${EXTRACTED}/spectrogram.png
sox -V1 "${analyzing_now}" -n remix 1 rate 128k spectrogram -c "${analyzing_now//$HOME\/}" -o "${spectrogram_png}"
