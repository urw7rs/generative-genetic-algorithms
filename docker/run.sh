#!/bin/bash

sudo docker build -t gga -f docker/Dockerfile .
sudo docker run --gpus all --ipc host -v $(pwd):/gga -v /mnt/data:/data -it gga bash
