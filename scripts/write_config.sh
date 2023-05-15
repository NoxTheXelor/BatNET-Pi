#!/usr/bin/bash
# Writes variables to config file
batnetpi_dir=$HOME/BatNET-Pi
baters_conf=${batnetpi_dir}/Baters_Guide_Installer_Configuration.txt
sed -i s/'^LATITUDE=$'/"LATITUDE=${new_lat}"/g ${baters_conf}
sed -i s/'^LONGITUDE=$'/"LONGITUDE=${new_lon}"/g ${baters_conf}
sed -i s/'^CADDY_PWD=$'/"CADDY_PWD=${caddy_pwd}"/g ${baters_conf}
sed -i s/'^ICE_PWD=$'/"ICE_PWD=${ice_pwd}"/g ${baters_conf}
sed -i s/'^DB_PWD=$'/"DB_PWD=${db_pwd}"/g ${baters_conf}
sed -i s/'^BATWEATHER_ID=$'/"BATWEATHER_ID=${batweather_id}"/g ${baters_conf}
sed -i s/'^BATNETPI_URL=$'/"BATNETPI_URL=${batnetpi_url/\/\//\\\/\\\/}"/g ${baters_conf}
sed -i s/'^WEBTERMINAL_URL=$'/"WEBTERMINAL_URL=${extractionlog_url/\/\//\\\/\\\/}"/g ${baters_conf}
sed -i s/'^BATNETLOG_URL=$'/"BATNETLOG_URL=${batnetlog_url/\/\//\\\/\\\/}"/g ${baters_conf}
