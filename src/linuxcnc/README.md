# LinuxCNC Files
## [cad_files/](https://github.com/feats-ai/feats/edit/main/src/linuxcnc/cad_files)
Folder containing the CAD files for the setup of the CNC machine. The files are organized using the following structure:
```
📄 cnc_interface.step            # STEP file of interface part to CNC machine
📄 cnc_interface.stl             # STL file of interface part to CNC machine
📄 gelsight-mini-adapter.step    # STEP file of GelSight Mini sensor adapter
📄 gelsight-mini-adapter.stl     # STL file of GelSight Mini sensor adapter
📄 resense-hex21-adapter.step    # STEP file of Resense Hex 21 sensor adapter
📄 resense-hex21-adapter.stl     # STL file of Resense Hex 21 sensor adapter
```

## [data/](https://github.com/feats-ai/feats/edit/main/src/linuxcnc/data)
Folder containing the data collected from the CNC machine. The files are organized using the following structure:
```
📂 cnc_coordinates/              # Folder containing the coordinates to move the CNC machine to
    📂 {Indenter}/               # Folder for specific indenter (e.g. Sphere)
      📄 {indenter1}.npy         # Numpy file containing the coordinates for the CNC machine
      📄 ...
📂 raw_data/                     # Folder containing the raw data collected from the sensors
    📂 {Indenter}/               # Folder for specific indenter (e.g. Sphere)
      📄 {%Y%m%d-%H%M%S}.npy     # Numpy file containing the raw data collected from the sensors for one measurement
      📄 ...
```

## [scripts/](https://github.com/feats-ai/feats/edit/main/src/linuxcnc/scripts)
Scripts related to data collection on the CNC machine. The files are organized using the following structure:
```
📄 cnc_coordinate_generator.py   # Script to generate coordinates for the CNC machine to move to
📄 visualize_indenter_data.py    # Script to visualize the data collected from one folder
```

## [src/](https://github.com/feats-ai/feats/edit/main/src/linuxcnc/src)
Source files for running the data collection on the CNC machine. The files are organized using the following structure:
```
📂 gelsight_mini/                # Source files for the GelSight Mini device
    📄 gsdevice.py               # Script to control the GelSight device (can be run independently)
📂 resense_hex_21/               # Source files for the Resense Hex 21 device
    📂 resense/                  # Library to control the Resense Hex 21 device
        📄 ...
    📄 hex21.py                  # Script to control the Resense Hex 21 device (can be run independently - need to adapt relative import path)
        ...
📄 cnc_data_aquisition.py        # Script to run the data collection on the CNC machine (make sure to start the server on the CNC machine first)
```

## [utils/](https://github.com/feats-ai/feats/edit/main/src/linuxcnc/utils)
Files for server and client class for remote procedure calls. The files are organized using the following structure:
```
📄 RPC.py                        # Module for Server and Client class for Remote Procedure Calls
📄 startSrc.py                   # Script to start the server on the CNC machine
📄 move_cnc.py                   # Script to move the CNC machine to a specific coordinate
```
