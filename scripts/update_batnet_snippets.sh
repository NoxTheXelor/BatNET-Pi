#!/usr/bin/env bash
# Update BatNET-Pi
source /etc/batnet/batnet.conf
trap 'exit 1' SIGINT SIGHUP
USER=$(awk -F: '/1000/ {print $1}' /etc/passwd)
HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)
my_dir=$HOME/BatNET-Pi/scripts

# Sets proper permissions and ownership
sudo -E chown -R $USER:$USER $HOME/*
sudo chmod -R g+wr $HOME/*

if ! grep PRIVACY_THRESHOLD /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "PRIVACY_THRESHOLD=0" >> /etc/batnet/batnet.conf
  git -C $HOME/BatNET-Pi rm $my_dir/privacy_server.py
fi
if [ -f $my_dir/privacy_server ] || [ -L /usr/local/bin/privacy_server.py ];then
  rm -f $my_dir/privacy_server.py
  rm -f /usr/local/bin/privacy_server.py
fi

# Adds python virtual-env to the python systemd services
if ! grep 'BatNET-Pi/batnet/' $HOME/BatNET-Pi/templates/batnet_server.service &>/dev/null || ! grep 'BatNET-Pi/batnet' $HOME/BatNET-Pi/templates/chart_viewer.service &>/dev/null;then
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$HOME/BatNET-Pi/batnet/bin/python3 /usr/local/bin/server.py|" ~/BatNET-Pi/templates/batnet_server.service
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$HOME/BatNET-Pi/batnet/bin/python3 /usr/local/bin/daily_plot.py|" ~/BatNET-Pi/templates/chart_viewer.service
  sudo systemctl daemon-reload && restart_services.sh
fi

if grep privacy ~/BatNET-Pi/templates/batnet_server.service &>/dev/null;then
  sudo -E sed -i 's/privacy_server.py/server.py/g' \
    ~/BatNET-Pi/templates/batnet_server.service
  sudo systemctl daemon-reload
  restart_services.sh
fi
if ! grep APPRISE_NOTIFICATION_TITLE /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_TITLE=\"New BatNET-Pi Detection\"" >> /etc/batnet/batnet.conf
fi
if ! grep APPRISE_NOTIFICATION_BODY /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFICATION_BODY=\"A \$sciname \$comname was just detected with a confidence of \$confidence\"" >> /etc/batnet/batnet.conf
fi
if ! grep APPRISE_NOTIFY_EACH_DETECTION /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_EACH_DETECTION=0 " >> /etc/batnet/batnet.conf
fi
if ! grep APPRISE_NOTIFY_NEW_SPECIES /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_NEW_SPECIES=0 " >> /etc/batnet/batnet.conf
fi
if ! grep APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_NOTIFY_NEW_SPECIES_EACH_DAY=0 " >> /etc/batnet/batnet.conf
fi
if ! grep APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_MINIMUM_SECONDS_BETWEEN_NOTIFICATIONS_PER_SPECIES=0 " >> /etc/batnet/batnet.conf
fi

# If the config does not contain the DATABASE_LANG setting, we'll want to add it.
# Defaults to not-selected, which config.php will know to render as a language option.
# The user can then select a language in the web interface and update with that.
if ! grep DATABASE_LANG /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "DATABASE_LANG=not-selected" >> /etc/batnet/batnet.conf
fi

apprise_installation_status=$(~/BatNET-Pi/batnet/bin/python3 -c 'import pkgutil; print("installed" if pkgutil.find_loader("apprise") else "not installed")')
if [[ "$apprise_installation_status" = "not installed" ]];then
  $HOME/BatNET-Pi/batnet/bin/pip3 install -U pip
  $HOME/BatNET-Pi/batnet/bin/pip3 install apprise
fi
[ -f $HOME/BatNET-Pi/apprise.txt ] || sudo -E -ucaddy touch $HOME/BatNET-Pi/apprise.txt
if ! which lsof &>/dev/null;then
  sudo apt update && sudo apt -y install lsof
fi
if ! grep RTSP_STREAM /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "RTSP_STREAM=" >> /etc/batnet/batnet.conf
fi
if grep bash $HOME/BatNET-Pi/templates/web_terminal.service &>/dev/null;then
  sudo sed -i '/User/d;s/bash/login/g' $HOME/BatNET-Pi/templates/web_terminal.service
  sudo systemctl daemon-reload
  sudo systemctl restart web_terminal.service
fi
[ -L ~/BatSongs/Extracted/static ] || ln -sf ~/BatNET-Pi/homepage/static ~/BatSongs/Extracted
if ! grep FLICKR_API_KEY /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FLICKR_API_KEY=" >> /etc/batnet/batnet.conf
fi
if systemctl list-unit-files pushed_notifications.service &>/dev/null;then
  sudo systemctl disable --now pushed_notifications.service
  sudo rm -f /usr/lib/systemd/system/pushed_notifications.service
  sudo rm $HOME/BatNET-Pi/templates/pushed_notifications.service
fi

if [ ! -f $HOME/BatNET-Pi/model/labels.txt ];then
  [ $DATABASE_LANG == 'not-selected' ] && DATABASE_LANG=en
  $my_dir/install_language_label.sh -l $DATABASE_LANG \
  && logger "[$0] Installed new language label file for '$DATABASE_LANG'";
fi

if ! grep FLICKR_FILTER_EMAIL /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FLICKR_FILTER_EMAIL=" >> /etc/batnet/batnet.conf
fi

pytest_installation_status=$(~/BatNET-Pi/batnet/bin/python3 -c 'import pkgutil; print("installed" if pkgutil.find_loader("pytest") else "not installed")')
if [[ "$pytest_installation_status" = "not installed" ]];then
  $HOME/BatNET-Pi/batnet/bin/pip3 install -U pip
  $HOME/BatNET-Pi/batnet/bin/pip3 install pytest==7.1.2 pytest-mock==3.7.0
fi

[ -L ~/BatSongs/Extracted/weekly_report.php ] || ln -sf ~/BatNET-Pi/scripts/weekly_report.php ~/BatSongs/Extracted

if ! grep weekly_report /etc/crontab &>/dev/null;then
  sed "s/\$USER/$USER/g" $HOME/BatNET-Pi/templates/weekly_report.cron | sudo tee -a /etc/crontab
fi
if ! grep APPRISE_WEEKLY_REPORT /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "APPRISE_WEEKLY_REPORT=1" >> /etc/batnet/batnet.conf
fi

if ! grep SILENCE_UPDATE_INDICATOR /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "SILENCE_UPDATE_INDICATOR=0" >> /etc/batnet/batnet.conf
fi

if ! grep '\-\-browser.gatherUsageStats false' $HOME/BatNET-Pi/templates/batnet_stats.service &>/dev/null ;then
  sudo -E sed -i "s|ExecStart=.*|ExecStart=$HOME/BatNET-Pi/batnet/bin/streamlit run $HOME/BatNET-Pi/scripts/plotly_streamlit.py --browser.gatherUsageStats false --server.address localhost --server.baseUrlPath \"/stats\"|" $HOME/BatNET-Pi/templates/batnet_stats.service
  sudo systemctl daemon-reload && restart_services.sh
fi

# Make IceCast2 a little more secure
sudo sed -i 's|<!-- <bind-address>.*|<bind-address>127.0.0.1</bind-address>|;s|<!-- <shoutcast-mount>.*|<shoutcast-mount>/stream</shoutcast-mount>|' /etc/icecast2/icecast.xml
sudo systemctl restart icecast2

if ! grep FREQSHIFT_TOOL /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_TOOL=sox" >> /etc/batnet/batnet.conf
fi
if ! grep FREQSHIFT_HI /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_HI=6000" >> /etc/batnet/batnet.conf
fi
if ! grep FREQSHIFT_LO /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_LO=3000" >> /etc/batnet/batnet.conf
fi
if ! grep FREQSHIFT_PITCH /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "FREQSHIFT_PITCH=-1500" >> /etc/batnet/batnet.conf
fi
if ! grep HEARTBEAT_URL /etc/batnet/batnet.conf &>/dev/null;then
  sudo -u$USER echo "HEARTBEAT_URL=" >> /etc/batnet/batnet.conf
fi

sudo systemctl daemon-reload
restart_services.sh
