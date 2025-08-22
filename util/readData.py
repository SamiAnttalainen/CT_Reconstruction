# A library to read the .tif CT-scan data as Acquisition data.
import numpy as np # type: ignore
from cil.io import TIFFStackReader # type: ignore
from cil.utilities.display import show2D # type: ignore
from cil.framework import AcquisitionGeometry # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore
from util.validateParameter import validate_parameter

def read_ct_data(
        filename: str,
        acquisition_geometry: AcquisitionGeometry,
        num_projections: int,
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
    num_projections
        Number of image projections
    
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
    validate_parameter(filename, 'filename', str)
    if len(filename) == 0:
        raise ValueError(f"Invalid filename, got {filename}.")
    validate_parameter(acquisition_geometry, 'acquisition_geometry', AcquisitionGeometry)
    validate_parameter(num_projections, 'num_projections', int, must_be_positive=True)

    # Region of interest
    roi = {'axis_0': (0, num_projections, skip), 'axis_1': -1, 'axis_2': (0, 1000,1)}

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
    validate_parameter(filename, 'filename', str)
    if len(filename) == 0:
        raise ValueError(f"Invalid filename, got {filename}.")
    validate_parameter(acquisition_geometry, 'acquisition_geometry', AcquisitionGeometry)
    validate_parameter(skip, 'skip', int, must_be_positive=True)
    validate_parameter(crop, 'crop', int)


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