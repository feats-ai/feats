# FEATS Files
## [calibration/](https://github.com/feats-ai/feats/tree/main/src/feats/calibration)
Folder storing the calibration files for different sensors. The files are organized using the following structure:
```
📂 nocontact_samples/           # Folder containing no contact samples of train and test sensors
    📄 {nocontact_X_sensor}.npy     
    📄 ...
📄 {calibration}.npy            # Numpy file containing the sensors calibration file
```

## [data/](https://github.com/feats-ai/feats/tree/main/src/feats/data)
Folder containing CalculiX results and labels for training, validation, and testing. The files are organized as follows:
```
📂 ccx/                         # Folder containing CalculiX results
    📂 {result_folder}          # CalculiX result files
    📂 ...
📂 labels/                      # Folder containing labels
    📂 train/                   # Training labels
        📄 {train_label}.npy    # Numpy file with training labels
        📄 ...
    📂 val/                     # Validation labels
        📄 {val_label}.npy      # Numpy file with validation labels
        📄 ...
    📂 test/                    # Testing labels
        📄 {test_label}.npy     # Numpy file with testing labels
        📄 ...
    📄 {normalization}.npy
```

## [homography/](https://github.com/feats-ai/feats/tree/main/src/feats/homography)
Folder containing files related to homography calculations. These files are used to calculate the projection from the FEA to the image plane of the GelSight Mini sensor. The files are organized as follows:
```
📄 cuboid_10-2-284.png          # Image from the GelSight Mini sensor used to get the point correspondences
📄 homography.npy               # Numpy file containing the homography matrix
📄 point_correspondences.yaml   # YAML file with the point correspondences
```

## [models/](https://github.com/feats-ai/feats/tree/main/src/feats/models)
Model weights (.pt) files from training. The files are structured as follows:
```
📄 {model_name}.pt              # PyTorch model weights file
📄 ...
```

## [runs/](https://github.com/feats-ai/feats/tree/main/src/feats/runs)
Folder containing log files from Weights and Biases. The files are organized as follows:
```
📂 wandb/                        # Folder containing Weights and Biases logs
    📂 run-{timestamp}/          # Folder for each run with a timestamp
        📄 ...
```

## [scripts/](https://github.com/feats-ai/feats/tree/main/src/feats/scripts)
Scripts related to label creation, data processing and sensor calibration. The files are organized as follows:
```
📄 calibrate_sensor.py           # Script for sensor calibration
📄 label_generation.py           # Script for generating labels
📄 split_data.py                 # Script for splitting data into train, validation, and test sets
📄 homography_estimation.py      # Script for estimating homography
📄 normalize.py                  # Script for normalizing data
📂 utils/                        # Utility scripts
    📄 ccx2paraview.py           # Script for converting CalculiX results to ParaView format
    📄 ccx_results_reader.py     # Script for reading CalculiX results
```

## [src/](https://github.com/feats-ai/feats/tree/main/src/feats/src)
Source files for training and inference of the U-net. The files are organized as follows:
```
📂 data/                         # Data loading and processing
    📄 dataloader.py             # Dataloader class and helper functions
📂 models/                       # Model architecture files
    📄 unet.py                   # UNet model definition
📂 predict/                      # Inference scripts
    📄 inference_speed.py        # Script for measuring inference speed
    📄 live.py                   # Script for live predictions
    📄 predict_config.yaml       # Configuration file for predictions
📂 train/                        # Training scripts and configurations
    📄 evaluate.py               # Script for evaluating models
    📄 train.py                  # Script for training models
    📄 train_config.yaml         # Configuration file for training
    📄 visualize.py              # Script for visualizing training results
    📄 visualize_config.yaml     # Configuration file for visualization
📂 utils/                        # Utility scripts
    📄 ulimit.sh                 # Script for setting system limits
```