#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu_custom:$LD_LIBRARY_PATH;


pushd .
cd /code/DataSets/CIFAR-10/

date >> /data/cifarlog.txt
echo "Downloading data..." >> /data/cifarlog.txt

/usr/bin/python3.5 install_cifar10.py
ls -l > /data/cifar-download-log.txt

date >> /data/cifarlog.txt
echo "Download complete." >> /data/cifarlog.txt
ls -l >> /data/cifarlog.txt

cd /code

date >> /data/cifarlog.txt
echo "Started training" >> /data/cifarlog.txt

python3.5  TrainResNet_CIFAR10.py -n resnet20 -logdir /data/cifar-tf-log -outputdir /data/cifar-output

date >> /data/cifarlog.txt
echo "Completed training" >> /data/cifarlog.txt
echo "--------------------" >> /data/cifarlog.txt

popd

