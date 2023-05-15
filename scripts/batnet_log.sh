#!/usr/bin/env bash
journalctl --no-hostname -q -o short -fu batnet_analysis -ubatnet_server -uextraction | sed "s/$(date "+%b %d ")//g;s/${HOME//\//\\/}\///g;/Line/d;/find/d;/systemd/d;s/ .*\[.*\]: /---/"
