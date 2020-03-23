# Keep notes about what we're doing here.

Platform: Google Cloud
project natgeo-dams


VM:
* natgeo-dams-training-server: 8 gpus
    * nvidia-smi will tell you the GPU status
    * /home/richpsharp/natgeo-dams is the root repo and already has training data installed in it

Model: https://github.com/fizyr/keras-retinanet
* This Docker image has Tensorflow 2.1.0 installed w/ GPU support, gsutils, and keras-retinanet
    * docker build dockerfile-dir -f dockerfile-dir/docker-gpu -t natcap/dam-inference-server-gpu:0.0.1 && docker run -it --rm -v `pwd`:/usr/local/natgeo_dams --gpus all natcap/dam-inference-server-gpu:0.0.1 python

    * git pull && docker build dockerfile-dir -f dockerfile-dir/docker-gpu -t natcap/dam-inference-server-gpu:0.0.1 && docker run -it --rm -v `pwd`:/usr/local/natgeo_dams --gpus all natcap/dam-inference-server-gpu:0.0.1 "bash -c retinanet-train --gpu 0 --multiprocessing --workers 16 --batch-size 8 csv ./training_data/annotations.csv ./training_data/classes.csv"

* To train:
    * note multi gpu doesn't seem to work

Data: natgeo-dams-data bucket
* gs://natgeo-dams-data/training_data holds the training data annotation.csv, classes.csv and the directory structure relative to them

