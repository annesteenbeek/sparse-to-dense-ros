#!/bin/bash

if [ ! -d "data"]; then
    mkdir data
fi

pushd data
wget -i ../datasets.txt
tar -xvf *.tar.gz
rm -f *.tar.gz

