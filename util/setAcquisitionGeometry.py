# A library to set the Acquisition geometry from getParameters.py's data.
import numpy as np
from cil.framework import AcquisitionGeometry # type: ignore

def set_acquistion_geometry(
        source_to_oringin_distance,
        origin_to_detector_distance,
        pixel_size,
        num_pixels,
        num_projections,
        skip,
        origin='bottom-right'
        ):
    
    """
    A library to set the Acquistion geometry for the 3D Cone-beam tomography.

    Args:
        source_to_origin_distance: Distance from the X-ray source to the origin
        origin_to_detector_distance: Distance from the origin to the detector
        pixel_size: Detector pixel size
        num_pixels: Number of pixels in the projection image
        num_projections: Number of projections used in the scan
        skip: Skip parameter to downsample scanning data
        origin: parameter to set up the origin of the acquisition geometry (default = 'bottom-right)

    Returns:
        ag: Acquistion geometry CIL's data format

    """
    
    angles = np.linspace(0, 360, num_projections//skip, endpoint=False) # Downsampled angles

    ag = AcquisitionGeometry.create_Cone3D(
        source_position=[0, -source_to_oringin_distance, 0],
        detector_position=[0, origin_to_detector_distance, 0],
        units='mm',
        detector_direction_x=[1, 0, 0],
        rotation_axis_direction=[0,0,1],  
    )

    ag.set_panel(
        num_pixels=num_pixels,
        pixel_size=(pixel_size,pixel_size),
        origin='bottom-right'
        )

    ag.set_angles(angles)
    ag.set_labels(('angle', 'vertical', 'horizontal'))
    print(ag)
    return ag