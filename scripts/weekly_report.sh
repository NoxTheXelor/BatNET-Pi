#!/usr/bin/env bash
source /etc/batnet/batnet.conf
if [ ${APPRISE_WEEKLY_REPORT} == 1 ];then
	NOTIFICATION=$(curl 'localhost/views.php?view=Weekly%20Report&ascii=true')
	NOTIFICATION=${NOTIFICATION#*#}
	firstLine=`echo "${NOTIFICATION}" | head -1`
	NOTIFICATION=`echo "${NOTIFICATION}" | tail -n +2`
	$HOME/BatNET-Pi/batnet/bin/apprise -vv -t "${firstLine}" -b "${NOTIFICATION}" --input-format=html --config=$HOME/BatNET-Pi/apprise.txt
fi