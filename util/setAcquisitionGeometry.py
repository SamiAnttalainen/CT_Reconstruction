# A library to set the Acquisition geometry from getParameters.py's data.
import numpy as np # type: ignore
from cil.framework import AcquisitionGeometry # type: ignore

def _validate_parameter(value, name: str, expected_types, allow_none: bool = False, 
                       must_be_positive: bool = False):
    """Helper function for parameter validation."""
    if allow_none and value is None:
        return
    
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = [t.__name__ for t in expected_types]
        else:
            type_names = [expected_types.__name__]
        
        if allow_none:
            type_names.append("None")
        
        type_str = " or ".join(type_names)
        raise TypeError(f"{name} must be {type_str}, got {type(value)}.")
    
    if must_be_positive and value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def set_acquistion_geometry(
        source_to_oringin_distance: float,
        origin_to_detector_distance: float,
        pixel_size: float,
        num_pixels: tuple,
        num_projections: int,
        skip: int = 10,
        origin: str ='bottom-right'
        ) -> AcquisitionGeometry:
    
    """
    Sets the Acquistion geometry for the 3D Cone-beam tomography.

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
    _validate_parameter(source_to_oringin_distance, "source_to_origin_distance", float, must_be_positive=True)
    _validate_parameter(origin_to_detector_distance, "origin_to_detector_distance", float, must_be_positive=True)
    _validate_parameter(pixel_size, "pixel_size", (int, float), must_be_positive=True)
    _validate_parameter(num_pixels, "num_pixels", tuple)
    _validate_parameter(num_projections, "num_projections", int, must_be_positive=True)
    _validate_parameter(skip, "skip", int, must_be_positive=True)
    _validate_parameter(origin, "origin", str)
    
    # Validate origin parameter
    valid_origins = {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
    if origin not in valid_origins:
        raise ValueError(f"origin must be one of {valid_origins}, got '{origin}'.")

    
    angles = np.linspace(0, 360, num_projections//skip, endpoint=False) # Downsampled angles

    acquisition_geometry = AcquisitionGeometry.create_Cone3D(
        source_position=[0, -source_to_oringin_distance, 0],
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