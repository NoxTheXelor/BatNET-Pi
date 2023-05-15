#!/usr/bin/env bash
# Install BatNET script
set -x # Debugging
exec > >(tee -i installation-$(date +%F).txt) 2>&1 # Make log
set -e # exit installation if anything fails

my_dir=$HOME/BatNET-Pi
export my_dir=$my_dir

cd $my_dir/scripts || exit 1

if [ "$(uname -m)" != "aarch64" ];then
  echo "BatNET-Pi requires a 64-bit OS.
It looks like your operating system is using $(uname -m),
but would need to be aarch64.
Please take a look at https://batnetwiki.pmcgui.xyz for more
information"
  exit 1
fi

#Install/Configure /etc/batnet/batnet.conf
./install_config.sh || exit 1
sudo -E HOME=$HOME USER=$USER ./install_services.sh || exit 1
source /etc/batnet/batnet.conf

install_batnet() {
  cd ~/BatNET-Pi || exit 1
  echo "Establishing a python virtual environment"
  sudo apt-get -y update #missing ensurepip - fix 1
  sudo apt-get -y install python3-venv
  python3 -m venv batnet
  source ./batnet/bin/activate
  pip3 install -U -r $HOME/BatNET-Pi/requirements.txt
  #cd scripts/bat_utils || exit 1
  #python3 setup.py build_ext --inplace
  #cd ../.. 
}

[ -d ${RECS_DIR} ] || mkdir -p ${RECS_DIR} &> /dev/null

install_batnet

cd $my_dir/scripts || exit 1

./install_language_label.sh -l $DATABASE_LANG || exit 1

exit 0