#!/bin/bash

docker run -it --rm \
    --gpus all \
    --device /dev/nvidia0 --device /dev/nvidia-modeset \
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    --device /dev/nvidiactl --network host \
    -v "$(realpath ./results):/content/DECA/results" \
    deca_deca-jupyter jupyter notebook --allow-root --NotebookApp.allow_origin=https://colab.research.google.com --port=8080 --NotebookApp.port_retries=0;