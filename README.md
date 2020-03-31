# natgeo-dams

1) download the model:
    * Copy ``https://storage.googleapis.com/natgeo-dams-data/models/natgeo_dams_model_resnet50_csv_64.h5`` to the current working directory.

2) build the docker image:
    * ``docker build -f dockerfile-dir\docker-microsoft-api dockerfile-dir -t natcap/1.0-natgeo-dams:1``

3) start the docker container:
    * ``docker run -it --rm -v `pwd`:/usr/local/natgeo_dams --gpus all natcap/1.0-natgeo-dams:1 bash``
    * In the bash terminal run: ``python retinet_inference_natgeo_dams.py --gpu 0 ./natgeo_dams_model_resnet50_csv_64.h5``

4) run inference on Earth:
    * ``python retinet_inference_natgeo_dams.py --gpu 0 ./natgeo_dams_model_resnet50_csv_64.h5``

# training data

Our training data are hosted on a publically readable Microsoft Azure bucket located at: https://natgeodamstrainingdata.blob.core.windows.net/training-data. **NOTE** this is a Azure storage link, not a web browser link and hence must be accessed with the Microsoft Storage Explorer or command line ``az`` tool.
