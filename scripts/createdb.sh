#!/usr/bin/env bash
source /etc/batnet/batnet.conf
sqlite3 $HOME/BatNET-Pi/scripts/bats.db << EOF
DROP TABLE IF EXISTS detections;
CREATE TABLE IF NOT EXISTS detections (
  Date DATE,
  Time TIME,
  Sci_Name VARCHAR(100) NOT NULL,
  Com_Name VARCHAR(100) NOT NULL,
  Confidence FLOAT,
  Lat FLOAT,
  Lon FLOAT,
  Cutoff FLOAT,
  Week INT,
  Sens FLOAT,
  Overlap FLOAT,
  File_Name VARCHAR(100) NOT NULL);
EOF
chown $USER:$USER $HOME/BatNET-Pi/scripts/bats.db
chmod g+w $HOME/BatNET-Pi/scripts/bats.db
