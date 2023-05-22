#!/usr/bin/env bash
# This installs the services that have been selected
set -x # Uncomment to enable debugging
trap 'rm -f ${tmpfile}' EXIT
trap 'exit 1' SIGINT SIGHUP
tmpfile=$(mktemp)

config_file=$my_dir/batnet.conf
export USER=$USER
export HOME=$HOME

export PYTHON_VIRTUAL_ENV="$HOME/BatNET-Pi/batnet/bin/python3"

install_depends() {
  apt install -y debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
  apt -qqq update && apt -qqy upgrade
  echo "icecast2 icecast2/icecast-setup boolean false" | debconf-set-selections
  apt install -qqy caddy ftpd sqlite3 php-sqlite3 alsa-utils \
    pulseaudio avahi-utils sox libsox-fmt-mp3 php php-fpm php-curl php-xml \
    php-zip icecast2 swig ffmpeg wget unzip curl cmake make bc libjpeg-dev \
    zlib1g-dev python3-dev python3-pip python3-venv lsof
}


set_hostname() {
  if [ "$(hostname)" == "raspberrypi" ];then
    hostnamectl set-hostname batnetpi
    sed -i 's/raspberrypi/batnetpi/g' /etc/hosts
  fi
}

update_etc_hosts() {
  sed -ie s/'$(hostname).local'/"$(hostname).local ${BATNETPI_URL//https:\/\/} ${WEBTERMINAL_URL//https:\/\/} ${BATNETLOG_URL//https:\/\/}"/g /etc/hosts
}

install_scripts() {
  ln -sf ${my_dir}/scripts/* /usr/local/bin/
}

install_batnet_analysis_timer() {
  echo "Installing batnet_analysis.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_analysis.timer
[Unit]
Description=BatNET Analysis Timer

[Timer]
OnCalendar= *-*-* *:00:15
AccuracySec= 1s
Persistent=True
Unit= batnet_analysis.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_analysis.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable batnet_analysis.timer
}

install_batnet_analysis() {
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_analysis.service
[Unit]
Description=BatNET Analysis
After=batnet_server.service
Requires=batnet_server.service
[Service]
Type=simple
Restart=on-success
User=${USER}
ExecStart=/usr/local/bin/batnet_analysis.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_analysis.service /usr/lib/systemd/system
  systemctl daemon-reload
}

install_batnet_server_timer() {
  echo "Installing batnet_server.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_server.timer
[Unit]
Description=BatNET Analysis Timer

[Timer]
OnCalendar= *-*-* *:00:10
AccuracySec= 1s 
Persistent=True
Unit= batnet_server.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_server.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable batnet_server.timer
}

install_batnet_server() {
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_server.service
[Unit]
Description=BatNET Analysis Server
Before=batnet_analysis.service
[Service]
Type=simple
Restart=on-success
User=${USER}
ExecStart=$PYTHON_VIRTUAL_ENV /usr/local/bin/server.py
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_server.service /usr/lib/systemd/system
  systemctl daemon-reload
}

install_extraction_service() {
  cat << EOF > $HOME/BatNET-Pi/templates/extraction.service
[Unit]
Description=BatNET BatSound Extraction
[Service]
Restart=on-failure
RestartSec=3
Type=simple
User=${USER}
ExecStart=/usr/bin/env bash -c 'while true;do extract_new_batsounds.sh;sleep 3;done'
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/extraction.service /usr/lib/systemd/system
  systemctl enable extraction.service
}

create_necessary_dirs() {
  echo "Creating necessary directories"
  [ -d ${EXTRACTED} ] || sudo -u ${USER} mkdir -p ${EXTRACTED}
  [ -d ${EXTRACTED}/By_Date ] || sudo -u ${USER} mkdir -p ${EXTRACTED}/By_Date
  [ -d ${EXTRACTED}/Charts ] || sudo -u ${USER} mkdir -p ${EXTRACTED}/Charts
  [ -d ${PROCESSED} ] || sudo -u ${USER} mkdir -p ${PROCESSED}
  [ -d $HOME/BatNET-Pi/perf_logs/ ] || sudo -u ${USER} mkdir -p $HOME/BatNET-Pi/perf_logs/


  sudo -u ${USER} ln -fs $my_dir/exclude_species_list.txt $my_dir/scripts
  sudo -u ${USER} ln -fs $my_dir/include_species_list.txt $my_dir/scripts
  sudo -u ${USER} ln -fs $my_dir/homepage/* ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/model/labels.txt ${my_dir}/scripts
  sudo -u ${USER} ln -fs $my_dir/scripts ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/play.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/spectrogram.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/overview.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/stats.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/todays_detections.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/history.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/scripts/weekly_report.php ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/homepage/images/favicon.ico ${EXTRACTED}
  sudo -u ${USER} ln -fs ${HOME}/phpsysinfo ${EXTRACTED}
  sudo -u ${USER} ln -fs $my_dir/templates/phpsysinfo.ini ${HOME}/phpsysinfo/
  sudo -u ${USER} ln -fs $my_dir/templates/green_bootstrap.css ${HOME}/phpsysinfo/templates/
  sudo -u ${USER} ln -fs $my_dir/templates/index_bootstrap.html ${HOME}/phpsysinfo/templates/html
  chmod -R g+rw $my_dir
  chmod -R g+rw ${RECS_DIR}
}

generate_BatDB() {
  echo "Generating BatDB.txt"
  if ! [ -f $my_dir/BatDB.txt ];then
    sudo -u ${USER} touch $my_dir/BatDB.txt
    echo "Date;Time;Sci_Name;Com_Name;Confidence;Lat;Lon;Cutoff;Week;Sens;Overlap" | sudo -u ${USER} tee -a $my_dir/BatDB.txt
  elif ! grep Date $my_dir/BatDB.txt;then
    sudo -u ${USER} sed -i '1 i\Date;Time;Sci_Name;Com_Name;Confidence;Lat;Lon;Cutoff;Week;Sens;Overlap' $my_dir/BatDB.txt
  fi
  chown $USER:$USER ${my_dir}/BatDB.txt && chmod g+rw ${my_dir}/BatDB.txt
}

set_login() {
  if ! [ -d /etc/lightdm ];then
    systemctl set-default multi-user.target
    ln -fs /lib/systemd/system/getty@.service /etc/systemd/system/getty.target.wants/getty@tty1.service
    cat > /etc/systemd/system/getty@tty1.service.d/autologin.conf << EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $USER --noclear %I \$TERM
EOF
  fi
}
#############################################################################################
#Stop recording TIMER ==> determines when to stop the service
install_stop_record_perf_timer() {
  echo "Installing stop_perf_recorder.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/stop_perf_recorder.timer
[Unit]
Description= Stop Recording CPU and RAM usage TIMER

[Timer]
OnCalendar= *-*-* *:50:00
AccuracySec= 1min
Persistent=True
Unit= stop_perf_recorder.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/stop_perf_recorder.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable stop_perf_recorder.timer
}
#Stop recording SERVICE ==> stop the service when timer says so
install_stop_record_perf_service() {
  echo "Installing stop_perf_recorder.service"
  cat << EOF > $HOME/BatNET-Pi/templates/stop_perf_recorder.service
[Unit]
Description=BatNET Stop Recording SERVICE

[Service]
Type=simple
User=${USER}
ExecStart= systemctl stop perf_recorder.service

[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/stop_perf_recorder.service /usr/lib/systemd/system
  systemctl daemon-reload
}
#start recording perf timer
install_record_perf_timer() {
  echo "Installing perf_recorder.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/perf_recorder.timer
[Unit]
Description= Start Recording CPU and RAM usage TIMER

[Timer]
OnCalendar= *-*-* *:55:00
AccuracySec= 1s
Persistent=True
Unit= perf_recorder.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/perf_recorder.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable perf_recorder.timer
}
#service to start and stop
#service storing cpu and ram usage
#ExecStart=/bin/bash $HOME/BatNET-Pi/scripts/perf_recorder.sh current
#ExecStart=/usr/local/bin/perf_recorder.sh previous ==> error 203
#ExecStart=/usr/bin/env bash -c"perf_recorder.sh" pre-previous
install_recording_perf_service() {
  echo "Installing perf_recorder.service"
  cat << EOF > $HOME/BatNET-Pi/templates/perf_recorder.service
[Unit]
Description=Recorder of CPU and RAM usage
[Service]
Type=simple
ExecStart=/bin/bash $HOME/BatNET-Pi/scripts/perf_recorder.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/perf_recorder.service /usr/lib/systemd/system
  systemctl daemon-reload
}
#############################################################################################
#############################################################################################
#Stop recording TIMER
install_stop_recording_timer() {
  echo "Installing stop_recording.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/stop_recording.timer
[Unit]
Description= BatNET Stop Recording TIMER

[Timer]
OnCalendar= *-*-* *:20:00
AccuracySec= 1min
Persistent=True
Unit= stop_recording.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/stop_recording.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable stop_recording.timer
}
#Stop recording SERVICE
install_stop_recording_service() {
  echo "Installing stop_recording.service"
  cat << EOF > $HOME/BatNET-Pi/templates/stop_recording.service
[Unit]
Description=BatNET Stop Recording SERVICE

[Service]
Type=simple
ExecStart=/usr/bin/env bash -c "sudo systemctl stop batnet_recording.service"

[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/stop_recording.service /usr/lib/systemd/system
  systemctl daemon-reload
}
#Start recording TIMER
install_recording_timer() {
  echo "Installing batnet_recording.timer"
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_recording.timer
[Unit]
Description=BatNET Recording Timer

[Timer]
OnCalendar= *-*-* *:00:00
AccuracySec= 1s
Persistent=True
Unit= batnet_recording.service

[Install]
WantedBy=timers.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_recording.timer /usr/lib/systemd/system
  systemctl daemon-reload
  systemctl enable batnet_recording.timer
}

install_recording_service() {
  echo "Installing batnet_recording.service"
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_recording.service
[Unit]
Description=BatNET Recording
[Service]
Environment=XDG_RUNTIME_DIR=/run/user/1000
Type=simple
User=${USER}
ExecStart=/usr/local/bin/batnet_recording.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_recording.service /usr/lib/systemd/system
  systemctl daemon-reload
}
#############################################################################################
install_custom_recording_service() {
  echo "Installing custom_recording.service"
  cat << EOF > $HOME/BatNET-Pi/templates/custom_recording.service
[Unit]
Description=BatNET Custom Recording
[Service]
Environment=XDG_RUNTIME_DIR=/run/user/1000
Type=simple
User=${USER}
ExecStart=/usr/local/bin/custom_recording.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/custom_recording.service /usr/lib/systemd/system
}

install_Caddyfile() {
  [ -d /etc/caddy ] || mkdir /etc/caddy
  if [ -f /etc/caddy/Caddyfile ];then
    cp /etc/caddy/Caddyfile{,.original}
  fi
  if ! [ -z ${CADDY_PWD} ];then
  HASHWORD=$(caddy hash-password --plaintext ${CADDY_PWD})
  cat << EOF > /etc/caddy/Caddyfile
http:// ${BATNETPI_URL} {
  root * ${EXTRACTED}
  file_server browse
  handle /By_Date/* {
    file_server browse
  }
  handle /Charts/* {
    file_server browse
  }
  basicauth /views.php?view=File* {
    batnet ${HASHWORD}
  }
  basicauth /Processed* {
    batnet ${HASHWORD}
  }
  basicauth /scripts* {
    batnet ${HASHWORD}
  }
  basicauth /stream {
    batnet ${HASHWORD}
  }
  basicauth /phpsysinfo* {
    batnet ${HASHWORD}
  }
  basicauth /terminal* {
    batnet ${HASHWORD}
  }
  reverse_proxy /stream localhost:8000
  php_fastcgi unix//run/php/php7.4-fpm.sock
  reverse_proxy /log* localhost:8080
  reverse_proxy /stats* localhost:8501
  reverse_proxy /terminal* localhost:8888
}
EOF
  else
    cat << EOF > /etc/caddy/Caddyfile
http:// ${BATNETPI_URL} {
  root * ${EXTRACTED}
  file_server browse
  handle /By_Date/* {
    file_server browse
  }
  handle /Charts/* {
    file_server browse
  }
  reverse_proxy /stream localhost:8000
  php_fastcgi unix//run/php/php7.4-fpm.sock
  reverse_proxy /log* localhost:8080
  reverse_proxy /stats* localhost:8501
  reverse_proxy /terminal* localhost:8888
}
EOF
  fi

  systemctl enable caddy
  usermod -aG $USER caddy
  usermod -aG video caddy
}

install_avahi_aliases() {
  cat << 'EOF' > $HOME/BatNET-Pi/templates/avahi-alias@.service
[Unit]
Description=Publish %I as alias for %H.local via mdns
After=network.target network-online.target
Requires=network-online.target
[Service]
Restart=always
RestartSec=3
Type=simple
ExecStart=/bin/bash -c "/usr/bin/avahi-publish -a -R %I $(hostname -I |cut -d' ' -f1)"
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/avahi-alias@.service /usr/lib/systemd/system
  systemctl enable avahi-alias@"$(hostname)".local.service
}

install_batnet_stats_service() {
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_stats.service
[Unit]
Description=BatNET Stats
[Service]
Restart=on-failure
RestartSec=5
Type=simple
User=${USER}
ExecStart=$HOME/BatNET-Pi/batnet/bin/streamlit run $HOME/BatNET-Pi/scripts/plotly_streamlit.py --browser.gatherUsageStats false --server.address localhost --server.baseUrlPath "/stats"

[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_stats.service /usr/lib/systemd/system
  systemctl enable batnet_stats.service
}

install_spectrogram_service() {
  cat << EOF > $HOME/BatNET-Pi/templates/spectrogram_viewer.service
[Unit]
Description=BatNET-Pi Spectrogram Viewer
[Service]
Restart=always
RestartSec=10
Type=simple
User=${USER}
ExecStart=/usr/local/bin/spectrogram.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/spectrogram_viewer.service /usr/lib/systemd/system
  systemctl enable spectrogram_viewer.service
}

install_chart_viewer_service() {
  echo "Installing the chart_viewer.service"
  cat << EOF > $HOME/BatNET-Pi/templates/chart_viewer.service
[Unit]
Description=BatNET-Pi Chart Viewer Service
[Service]
Restart=always
RestartSec=120
Type=simple
User=$USER
ExecStart=$PYTHON_VIRTUAL_ENV /usr/local/bin/daily_plot.py
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/chart_viewer.service /usr/lib/systemd/system
  systemctl enable chart_viewer.service
}

install_gotty_logs() {
  sudo -u ${USER} ln -sf $my_dir/templates/gotty \
    ${HOME}/.gotty
  sudo -u ${USER} ln -sf $my_dir/templates/bashrc \
    ${HOME}/.bashrc
  cat << EOF > $HOME/BatNET-Pi/templates/batnet_log.service
[Unit]
Description=BatNET Analysis Log
[Service]
Restart=on-failure
RestartSec=3
Type=simple
User=${USER}
Environment=TERM=xterm-256color
ExecStart=/usr/local/bin/gotty --address localhost -p 8080 -P log --title-format "BatNET-Pi Log" batnet_log.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/batnet_log.service /usr/lib/systemd/system
  systemctl enable batnet_log.service
  cat << EOF > $HOME/BatNET-Pi/templates/web_terminal.service
[Unit]
Description=BatNET-Pi Web Terminal
[Service]
Restart=on-failure
RestartSec=3
Type=simple
Environment=TERM=xterm-256color
ExecStart=/usr/local/bin/gotty --address localhost -w -p 8888 -P terminal --title-format "BatNET-Pi Terminal" login
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/web_terminal.service /usr/lib/systemd/system
  systemctl enable web_terminal.service
}

configure_caddy_php() {
  echo "Configuring PHP for Caddy"
  sed -i 's/www-data/caddy/g' /etc/php/*/fpm/pool.d/www.conf
  systemctl restart php7\*-fpm.service
  echo "Adding Caddy sudoers rule"
  cat << EOF > /etc/sudoers.d/010_caddy-nopasswd
caddy ALL=(ALL) NOPASSWD: ALL
EOF
  chmod 0440 /etc/sudoers.d/010_caddy-nopasswd
}

install_phpsysinfo() {
  sudo -u ${USER} git clone https://github.com/phpsysinfo/phpsysinfo.git \
    ${HOME}/phpsysinfo
}

config_icecast() {
  if [ -f /etc/icecast2/icecast.xml ];then
    cp /etc/icecast2/icecast.xml{,.prebatnetpi}
  fi
  sed -i 's/>admin</>batnet</g' /etc/icecast2/icecast.xml
  passwords=("source-" "relay-" "admin-" "master-" "")
  for i in "${passwords[@]}";do
  sed -i "s/<${i}password>.*<\/${i}password>/<${i}password>${ICE_PWD}<\/${i}password>/g" /etc/icecast2/icecast.xml
  done
  sed -i 's|<!-- <bind-address>.*|<bind-address>127.0.0.1</bind-address>|;s|<!-- <shoutcast-mount>.*|<shoutcast-mount>/stream</shoutcast-mount>|'

  systemctl enable icecast2.service
}

install_livestream_service() {
  cat << EOF > $HOME/BatNET-Pi/templates/livestream.service
[Unit]
Description=BatNET-Pi Live Stream
After=network-online.target
Requires=network-online.target
[Service]
Environment=XDG_RUNTIME_DIR=/run/user/1000
Restart=always
Type=simple
RestartSec=3
User=${USER}
ExecStart=/usr/local/bin/livestream.sh
[Install]
WantedBy=multi-user.target
EOF
  ln -sf $HOME/BatNET-Pi/templates/livestream.service /usr/lib/systemd/system
  systemctl enable livestream.service
}

install_cleanup_cron() {
  sed "s/\$USER/$USER/g" $my_dir/templates/cleanup.cron >> /etc/crontab
}

install_weekly_cron() {
  sed "s/\$USER/$USER/g" $my_dir/templates/weekly_report.cron >> /etc/crontab
}

chown_things() {
  chown -R $USER:$USER $HOME/Bat*
}

install_services() {
  set_hostname
  update_etc_hosts
  set_login

  install_depends
  install_scripts
  install_Caddyfile
  install_avahi_aliases
  install_batnet_analysis
  install_batnet_analysis_timer  # analysis timer
  install_batnet_server
  install_batnet_server_timer    # server timer
  install_batnet_stats_service

  install_stop_recording_timer
  install_stop_recording_service
  install_recording_timer
  install_recording_service

  install_custom_recording_service # But does not enable

  install_stop_record_perf_timer    #
  install_stop_record_perf_service  #
  install_record_perf_timer   #
  install_recording_perf_service    #

  install_extraction_service
  install_spectrogram_service
  install_chart_viewer_service
  install_gotty_logs
  install_phpsysinfo
  install_livestream_service
  install_cleanup_cron
  install_weekly_cron

  create_necessary_dirs
  generate_BatDB
  configure_caddy_php
  config_icecast
  USER=$USER HOME=$HOME ${my_dir}/scripts/createdb.sh
}

if [ -f ${config_file} ];then
  source ${config_file}
  install_services
  chown_things
else
  echo "Unable to find a configuration file. Please make sure that $config_file exists."
fi
