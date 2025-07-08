# A library for setting up CT-scan parameters from the 'params' dictionary, which is read first by using pcaReader.py.

def get_parameters(params):

    """
    A function for reading pcaReader.py's parameters dictionary and for setting up the CT-scan parameters.

    Args:
        params: CT-scan dictionary from pcaReader function.

    Returns:
        SOD: Source-Origin-distance in millimeters.
        SDD: Source-Detector-distance in millimeters.
        ODD: Origin-Detector-distance in millimeters.
        pixel_size: Detector pixel size in millimeters.
        num_pixels: Number of pixels in the projection image.
        num_projs: Number of projections used in the scan.
    """


    SOD = params['FOD'] # Source-Origin-distance
    SDD = params['FDD'] # Source-Detector-distance
    ODD = SDD - SOD # Origin-Detector-distance

    print(f'SOD: {SOD:.3f} mm')
    print(f'SDD: {SDD:.3f} mm')
    print(f'ODD: {ODD:.3f} mm')

    pixel_size = params['PixelsizeX']
    print(f'Detector pixel size: {pixel_size:.3f} mm')

    num_pixels = (params['DimX'], params['DimY']) # Width and height of the projection image
    num_pixels = (num_pixels[0], num_pixels[1])
    print(f'The original size of the projection image: {params['DimX']} x {params['DimY']}, set to {num_pixels[0]} x {num_pixels[1]}')

    # Projections
    num_projs = params['NumberImages']
    print(f'{num_projs} projection angles used')

    # White level intensity
    max_k = params['FreeRay']
    print(f'White level intensity: {max_k = }')

    return SOD, SDD, ODD, pixel_size, num_pixels, num_projs, max_k
