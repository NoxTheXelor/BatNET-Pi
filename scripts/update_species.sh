#!/usr/bin/env bash
# Update the species list
#set -x
source /etc/batnet/batnet.conf
if [ -f $HOME/BatNET-Pi/scripts/bats.db ];then
sqlite3 $HOME/BatNET-Pi/scripts/bats.db "SELECT DISTINCT(Com_Name) FROM detections" | sort >  ${IDFILE}
fi
