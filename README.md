## CT Reconstructions

### Info
This repository contains reconstructions of raw CT-data by using CIL (Core Imaging Library).
To understand how to use CIL and utility functions to reconstruct, go check .ipynb files in the "demos" folder.

### Short Guide for CT reconstruction
1. Read PCA file with "pca_pca_file" utility function to acquire CT scan parameters
2. Set up CT scan variables with "get_params" utility function
3. Set up acquisition geometry with "set_acquisition_geometry" utility function
4. Read raw CT data using "read_ct_data" utility function
5. Convert the transmission data to the absorption data using "convert_data" utility function
6. (Optional) Convert the geometry of the data from fan-beam to the parallel using "convert_fan_to_parallel_geometry" utility function
7. Reconstruct using the algorithm suitable for given geometry.

### Data
The acquisition data is in fan-beam geometry and needs to be converted into parallel-beam geometry, if testing reconstruction with parallel-beam based algorithms.