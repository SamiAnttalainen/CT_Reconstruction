# A library to set the Acquisition geometry from getParameters.py's data.
import numpy as np # type: ignore
from cil.framework import AcquisitionGeometry # type: ignore
from util.getParameters import get_ct_parameters
from util.getParameters import get_cl_parameters
from util.validateParameter import validate_parameter
        

def set_ct_acquisition_geometry(
        params: dict,
        skip: int = 10,
        origin: str ='bottom-right',
        ) -> AcquisitionGeometry:
    
    """
    Sets the Acquisition geometry for the 3D Cone-beam computerized tomography.

    Parameters
    ----------
    params: dict
        Computerized tomography parameter dictionary
    skip : int, default 10
        Skip parameter to downsample scanning data
    origin : str, default "bottom-right"
        parameter string to set up the origin of the acquisition geometry.
        String must be one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    ctype : None or str, default None
        Computerized type. "None" is normal tomography and "CL" is laminography

    Returns
    -------
    acquisition_geometry : AcquistionGeometry
        CIL's acquistion geometry data class

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    validate_parameter(params, "params", dict)
    validate_parameter(skip, "skip", int, must_be_positive=True)
    validate_parameter(origin, "origin", str)
    
    # Validate origin parameter
    valid_origins = {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
    if origin not in valid_origins:
        raise ValueError(f"origin must be one of {valid_origins}, got '{origin}'.")
    
    source_to_origin_distance, \
    _, \
    origin_to_detector_distance, \
    pixel_size, \
    num_pixels, \
    num_projections, \
    _ = get_ct_parameters(params)

    
    angles = np.linspace(0, 360, num_projections//skip, endpoint=False) # Downsampled angles

    acquisition_geometry = AcquisitionGeometry.create_Cone3D(
        source_position=[0, -source_to_origin_distance, 0],
        detector_position=[0, origin_to_detector_distance, 0],
        units='mm',
        detector_direction_x=[1, 0, 0],
        rotation_axis_direction=[0,0,1],  
    )

    acquisition_geometry.set_panel(
        num_pixels=num_pixels,
        pixel_size=(pixel_size,pixel_size),
        origin=origin
        )

    acquisition_geometry.set_angles(angles)
    acquisition_geometry.set_labels(('angle', 'vertical', 'horizontal'))
    print(acquisition_geometry)
    return acquisition_geometry



def set_cl_acquisition_geometry(
        params: dict,
        skip: int = 10,
        origin: str ='bottom-right',
        ) -> AcquisitionGeometry:
    
    """
    Sets the Acquisition geometry for the 3D Cone-beam computerized laminography.

    Parameters
    ----------
    source_to_origin_distance : float
        Distance from the X-ray source to the origin
    origin_to_detector_distance : float
        Distance from the origin to the detector
    pixel_size : float
        Detector pixel size
    num_pixels : tuple 
        Number of pixels in the projection image as a tuple
    num_projections : int
        Number of projections used in the scan
    skip : int, default 10
        Skip parameter to downsample scanning data
    origin : str, default "bottom-right"
        parameter string to set up the origin of the acquisition geometry.
        String must be one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'

    Returns
    -------
    acquisition_geometry : AcquistionGeometry
        CIL's acquistion geometry data class

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    validate_parameter(params, "params", dict)
    validate_parameter(skip, "skip", int, must_be_positive=True)
    validate_parameter(origin, "origin", str)

    
    # Validate origin parameter
    valid_origins = {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
    if origin not in valid_origins:
        raise ValueError(f"origin must be one of {valid_origins}, got '{origin}'.")

    # Set acquisition geometry for laminography
    
    _, \
    source_to_detector_distance, \
    origin_to_detector_distance, \
    pixel_size, \
    num_pixels, \
    _, \
    _, \
    _, \
    angles_list, \
    rotation_axis, \
    height_offset = get_cl_parameters(params)

    acquisition_geometry = AcquisitionGeometry.create_Cone3D(
        source_position=[0.0, -source_to_detector_distance,0.0], 
        detector_position=[0.0, origin_to_detector_distance, 0.0],
        rotation_axis_position=height_offset,
        rotation_axis_direction= rotation_axis)
    
    acquisition_geometry.set_angles(
        angles=angles_list,
        angle_unit='degree'
        )

    acquisition_geometry.set_panel(
        num_pixels=num_pixels, 
        pixel_size=pixel_size,
        origin=origin
        )
     
    acquisition_geometry.set_labels(['angle','vertical','horizontal'])

    print(acquisition_geometry)
    return acquisition_geometry