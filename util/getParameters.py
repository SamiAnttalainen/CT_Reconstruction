# A library for setting up CT-scan parameters from the 'params' dictionary, which is read first by using pcaReader.py.

def get_parameters(
        params: dict
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


    # Extract and validate required parameters
    required_keys = ['FOD', 'FDD', 'PixelsizeX', 'DimX', 'DimY', 
                     'NumberImages', 'FreeRay']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
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
