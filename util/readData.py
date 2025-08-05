# A library to read the .tif CT-scan data as Acquisition data.
import numpy as np # type: ignore
from cil.io import TIFFStackReader # type: ignore
from cil.utilities.display import show2D # type: ignore
from cil.framework import AcquisitionGeometry # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore

def read_ct_data(
        filename: str,
        acquisition_geometry: AcquisitionGeometry,
        skip: int = 10
        ) -> AcquisitionData:
    
    """
    Reads .tif files and converts to CIL's
    acquisition data.

    Parameters
    ----------
    filename : str
        Folder path name to the .tif files
    acquisition_geometry : AcquisitionGeometry
        CIL's acquisition geometry data class
    skip : int, default 10
        Downsampling parameter for the acquisition data
    
    Returns
    -------
    acquisition_data : acquisitionData
        Converted acquisition data from the .tif files
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string, got {type(filename)}.")
    if len(filename) == 0:
        raise ValueError(f"Invalid filename, got {filename}.")
    if not isinstance(acquisition_geometry, AcquisitionGeometry):
        raise TypeError(f"acquisition_geometry must be CIL's acquisitionGeometry data class, got {type(acquisition_geometry)}.")
    if not isinstance(skip, int):
        raise TypeError(f"skip must be an integer, got {type(skip)}.")
    if skip <= 0:
        raise ValueError(f"skip must be a positive integer, got {skip}.")
    
    # Region of interest
    roi = {'axis_0': (0, acquisition_geometry.num_projections, skip), 'axis_1': -1, 'axis_2': (0, 1000,1)}

    # .tif file reader setup
    reader = TIFFStackReader(file_name=filename, transpose=False, roi=roi)

    # acquisition data from the .tif files
    acquisition_data = reader.read_as_AcquisitionData(acquisition_geometry)

    del reader
    print(acquisition_data)
    show2D(datacontainers=acquisition_data, origin='upper-right');

    return acquisition_data

def read_cl_data(
        filename: str,
        acquisition_geometry: AcquisitionGeometry,
        skip: int = 2,
        crop: int = 0
        ) -> AcquisitionData:
    
    """
    Reads .tif files and converts to CIL's
    acquisation data.

    Parameters
    ----------
    filename : str
        Folder path name to the .tif files
    acquisition_geometry : acquisitionGeometry
        CIL's acquisition geometry data class
    skip : int, default 10
        Downsampling parameter for the acquisition data
    crop : int, default 0
        Crop parameter for image cropping
    
    Returns
    -------
    acquisition_data : acquisitionData
        Converted acquisition data from the .tif files
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string, got {type(filename)}.")
    if len(filename) == 0:
        raise ValueError(f"Invalid filename, got {filename}.")
    if not isinstance(acquisition_geometry, AcquisitionGeometry):
        raise TypeError(f"acquisition_geometry must be CIL's acquisitionGeometry data class, got {type(acquisition_geometry)}.")
    if not isinstance(skip, int):
        raise TypeError(f"skip must be an integer, got {type(skip)}.")
    if skip <= 0:
        raise ValueError(f"skip must be a positive integer, got {skip}.")
    if not isinstance(crop, int):
        raise TypeError(f"crop must be an integer, got {type(crop)}.")

    # Normal region of interest
    roi = {
        'axis_0': (None, -1, skip),
        'axis_1': (None, None, None), 
        'axis_2': (None, None, None)
        }

    # Update angles with skipping
    rotation_sector = np.ceil(acquisition_geometry.angles[-1])
    angles_list = np.linspace(0, rotation_sector, acquisition_geometry.num_projections // skip, endpoint=False)
    acquisition_geometry.set_angles(angles_list)


    # Update roi and panel if cropping
    if crop != 0:

        roi = {
            'axis_0': (None, None, skip),
            'axis_1': (crop, -crop, None), 
            'axis_2': (crop, -crop, None)
            }

        num_pixels_x = (acquisition_geometry.pixel_num_h - 2*crop)
        num_pixels_y = (acquisition_geometry.pixel_num_v - 2*crop)


        acquisition_geometry.set_panel(
            num_pixels=[num_pixels_x,num_pixels_y],
            pixel_size=acquisition_geometry.pixel_size_h,
            origin='top-left'
            )

    
    # .tif file reader setup
    reader = TIFFStackReader(file_name=filename, roi=roi, mode='slice')

    # Acquisition data from the .tif files
    acquisition_data = reader.read_as_AcquisitionData(acquisition_geometry)

    del reader
    print(acquisition_data)
    show2D(datacontainers=acquisition_data, origin='upper-right');

    return acquisition_data