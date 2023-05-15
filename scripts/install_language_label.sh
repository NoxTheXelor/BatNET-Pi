#!/usr/bin/env bash

usage() { echo "Usage: $0 -l <language i18n id>" 1>&2; exit 1; }

while getopts "l:" o; do
  case "${o}" in
    l)
      lang=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

HOME=$(awk -F: '/1000/ {print $6}' /etc/passwd)

label_file_name="labels_${lang}.txt"

unzip -o $HOME/BatNET-Pi/model/labels_l18n.zip $label_file_name \
  -d $HOME/BatNET-Pi/model \
  && mv -f $HOME/BatNET-Pi/model/$label_file_name $HOME/BatNET-Pi/model/labels.txt \
  && logger "[$0] Changed language label file to '$label_file_name'";

exit 0
