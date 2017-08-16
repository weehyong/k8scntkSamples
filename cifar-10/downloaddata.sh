#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu_custom:$LD_LIBRARY_PATH;

pushd .
cd /code/DataSets/CIFAR-10/
/usr/bin/python3.5 install_cifar10.py
ls -l > /data/cifar-download-log.txt

cd /code

popd


