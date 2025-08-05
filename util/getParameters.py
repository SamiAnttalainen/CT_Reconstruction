# A library for setting up CT-scan parameters from the 'params' dictionary, which is read first by using pcaReader.py.
import numpy as np # type: ignore

def get_ct_parameters(
        params: dict,
        ) -> tuple[float, float, float, float, int, int, int]:

    """
    Reads parameters dictionary and sets up the CT-scan parameters.

    Parameters
    ----------
    params : dict
        CT-scan parameter dictionary from pcaReader function

    Returns
    -------
    SOD : float
        Source-Origin-distance in millimeters
    SDD : float
        Source-Detector-distance in millimeters
    ODD : float
        Origin-Detector-distance in millimeters
    pixel_size : float
        Detector pixel size in millimeters
    num_pixels : int
        Number of pixels in the projection image
    num_projs : int
        Number of projections used in the scan
    intensity : int
        White level intensity

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dictionary, got {type(params)}.")
    
    try:
        # Source and detector distances
        SOD = float(params['FOD'])  # Source-Origin-distance
        SDD = float(params['FDD'])  # Source-Detector-distance
        
        # Validate geometric constraints
        if SOD <= 0:
            raise ValueError("SOD must be positive")
        if SDD <= 0:
            raise ValueError("SDD must be positive")
        if SDD <= SOD:
            raise ValueError("SDD must be greater than SOD")
            
        ODD = SDD - SOD  # Origin-Detector-distance
        
        print(f'SOD: {SOD:.3f} mm')
        print(f'SDD: {SDD:.3f} mm')
        print(f'ODD: {ODD:.3f} mm')

        # Detector pixel size
        pixel_size = float(params['PixelsizeX'])
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        print(f'Detector pixel size: {pixel_size:.3f} mm')

        # Image dimensions
        dim_x = int(params['DimX'])
        dim_y = int(params['DimY'])
        
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError("Image dimensions must be positive")
            
        num_pixels = (dim_x, dim_y)
        print(f'Projection image size: {dim_x} x {dim_y}')

        # Number of projections
        num_projs = int(params['NumberImages'])
        if num_projs <= 0:
            raise ValueError("Number of projections must be positive")
        print(f'{num_projs} projection angles used')

        # White level intensity
        intensity = int(params['FreeRay'])
        if intensity < 0:
            raise ValueError("Intensity must be non-negative")
        print(f'White level intensity: {intensity}')
            
    except (ValueError, TypeError) as e:
        if "invalid literal" in str(e) or "int()" in str(e):
            raise ValueError(f"Invalid parameter format: {e}")
        raise
    return SOD, SDD, ODD, pixel_size, num_pixels, num_projs, intensity



def get_cl_parameters(
        params: dict,
        ctype: str = None,
        skip: int = 1
        ) -> tuple[float, float, float, float, int, int, int, float, np.ndarray, np.ndarray, np.ndarray]:

    """
    Reads parameters dictionary and sets up the CT-scan parameters.

    Parameters
    ----------
    params : dict
        CT-scan parameter dictionary from pcaReader function
    skip : int, default 1
        Projection angle skipping parameter

    Returns
    -------
    SOD : float
        Source-Origin-distance in millimeters
    SDD : float
        Source-Detector-distance in millimeters
    ODD : float
        Origin-Detector-distance in millimeters
    pixel_size : float
        Detector pixel size in millimeters
    num_pixels : int
        Number of pixels in the projection image
    num_projs : int
        Number of projections used in the scan
    intensity : int
        White level intensity
    tilt : float
        Tilt angle of the beam in radians
    angles_list : np.ndarray
        Projection angles list
    rotation_axis : np.ndarray
        Rotation axis direction

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dictionary, got {type(params)}.")
    if ctype is not None and not isinstance(ctype, str):
        raise TypeError(f"ctype must be None or a string, got {type(ctype)}.")
    
    try:
        # Source and detector distances
        SOD = float(params['FOD'])  # Source-Origin-distance
        SDD = float(params['FDD'])  # Source-Detector-distance
        
        # Validate geometric constraints
        if SOD <= 0:
            raise ValueError("SOD must be positive")
        if SDD <= 0:
            raise ValueError("SDD must be positive")
        if SDD <= SOD:
            raise ValueError("SDD must be greater than SOD")
            
        ODD = SDD - SOD  # Origin-Detector-distance
        
        print(f'SOD: {SOD:.3f} mm')
        print(f'SDD: {SDD:.3f} mm')
        print(f'ODD: {ODD:.3f} mm')

        # Detector pixel size
        pixel_size = float(params['PixelsizeX'])
        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
        print(f'Detector pixel size: {pixel_size:.3f} mm')

        # Image dimensions
        dim_x = int(params['DimX'])
        dim_y = int(params['DimY'])
        
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError("Image dimensions must be positive")
            
        num_pixels = (dim_x, dim_y)
        print(f'Projection image size: {dim_x} x {dim_y}')

        # Number of projections
        num_projs = int(params['NumberImages'])
        if num_projs <= 0:
            raise ValueError("Number of projections must be positive")
        print(f'{num_projs} projection angles used')

        # White level intensity
        intensity = int(params['FreeRay'])
        if intensity < 0:
            raise ValueError("Intensity must be non-negative")
        print(f'White level intensity: {intensity}')
        
        # Returns additional parameters for laminography
                
        # Beam tilt angle in radians
        tilt = params["Tilt"] * np.pi / 180
        print(f'Tilt angle in radians: {tilt}')

        angles_list = np.linspace(0, params["RotationSector"], params["NumberImages"] // skip, endpoint=False)

        rotation_axis = -np.array([0,-np.sin(tilt), np.cos(tilt)]) # Points down for correct rotation direction
                  
        height_offset = params["PlanarCTRotCenter"] * np.array([0,-np.sin(tilt), np.cos(tilt)]) # Offset along the rotation axis direction (because the machine assumes rotating table remains level and detector rotates)

        

    except (ValueError, TypeError) as e:
        if "invalid literal" in str(e) or "int()" in str(e):
            raise ValueError(f"Invalid parameter format: {e}")
        raise

    return SOD, SDD, ODD, pixel_size, num_pixels, num_projs, intensity, tilt, angles_list, rotation_axis, height_offset
    
