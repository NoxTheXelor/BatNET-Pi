<h1 align="center">
  BatNET-Pi
</h1>
<p align="center">
A realtime acoustic bat classification system for the Raspberry Pi 4B
</p>


## Introduction
BaTNET-Pi is based on the BirdNET-Pi project. It is able to cover 10 hours of recording, to detect and classify bat sounds from a USB microphone or sound card.

## Branchs
- Testing has been used to measures the performances of the various model tested
- optiXGB contained the BatNET-Pi powered by a binary neural network model
- main and batnet are both powered by BatDetect2 (and then the most 2 recent branchs) but batnet differs by the replacement of all occurence of the word "bird" by the word "bat"

## Requirements
* A Raspberry Pi 4
* An SD Card with the **_64-bit version of RaspiOS_** installed (please use Bullseye) -- Lite is recommended, but the installation works on RaspiOS-ARM64-Full as well. Downloads available within the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
* A ultrasonic USB Microphone or Sound Card

## Installation
The installation is similar to the [BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) one just [here](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Installation-Guide).
Please note that the comments made in the BirdNET-Pi installation section also apply here.
Note also that the web interface run, as it was left untouched, but is not filled by the detections made by the system.

The system can be installed with:

- for the main branch
```
curl -s https://raw.githubusercontent.com/NoxTheXelor/BatNET-Pi/main/newinstaller.sh | bash
```
- for the batnet branch
```
curl -s https://raw.githubusercontent.com/NoxTheXelor/BatNET-Pi/batnet/newinstaller.sh | bash
```
- for the optiXGB branch
```
curl -s https://raw.githubusercontent.com/NoxTheXelor/BatNET-Pi/optiXGB/newinstaller.sh | bash
```
- for the testing branch
```
curl -s https://raw.githubusercontent.com/NoxTheXelor/BatNET-Pi/testing/newinstaller.sh | bash
```
The installer takes care of any and all necessary updates, so you can run that as the very first command upon the first boot, if you'd like.

The installation creates a log in `$HOME/installation-$(date "+%F").txt`.

Please take a look at the [wiki](https://github.com/mcguirepr89/BirdNET-Pi/wiki) and [discussions](https://github.com/mcguirepr89/BirdNET-Pi/discussions) for information on
- [BirdNET-Pi for bat](https://github.com/mcguirepr89/BirdNET-Pi/discussions/300)
- [making your installation public](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Sharing-Your-BirdNET-Pi)
- [backing up and restoring your database](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Backup-and-Restore-the-Database)
- [adjusting your sound card settings](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Adjusting-your-sound-card)
- [suggested USB microphones](https://github.com/mcguirepr89/BirdNET-Pi/discussions/39)
- [building your own microphone](https://github.com/DD4WH/SASS/wiki/Stereo--(Mono)-recording-low-noise-low-cost-system)
- [privacy concerns and options](https://github.com/mcguirepr89/BirdNET-Pi/discussions/166)
- [beta testing](https://github.com/mcguirepr89/BirdNET-Pi/discussions/11)
- [and more!](https://github.com/mcguirepr89/BirdNET-Pi/discussions)

## Uninstallation
```
/usr/local/bin/uninstall.sh && cd ~ && rm -drf BirdNET-Pi
```

