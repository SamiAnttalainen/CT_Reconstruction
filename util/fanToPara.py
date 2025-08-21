""" A library to convert fan-beam data geometry to parallel-beam data geometry."""
import numpy as np #type: ignore
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # type: ignore
from skimage.transform import iradon # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore
from util.validateParameter import validate_parameter


def convert_fan_to_parallel_geometry(
        data: np.ndarray,
        idx: int,
        source_origin_distance: float,
        detector_pixel_size: float = 0.2,
        direction: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Convert CIL's 2D fan-beam sinogram projections to parallel-beam.

    Parameters
    ----------
    data : np.ndarray
        CIL's absorption data datacontainer
    idx : int
        Index of the slice
    source_origin_distance : float
        Source-Origin-distance in millimeters
    detector_pixel_size : float, default 0.2
        Detector pixel size in millimeters
    direction: bool, default False
        Direction is vertical if false, else horizontal

    Returns
    -------
    Psinogram : np.ndarray
        Converted parallel-beam sinogram
    Ploc : np.ndarray
        Parallel-beam sensor locations
    Pangles : np.ndarray
        Parallel-beam projection angles

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    validate_parameter(data, "data", AcquisitionData)
    validate_parameter(idx, "idx", (int, float), must_be_positive=True)
    validate_parameter(source_origin_distance, "source_origin_distance", (int, float), must_be_positive=True)
    validate_parameter(detector_pixel_size, "detector_pixel_size", (int, float), must_be_positive=True)
    validate_parameter(direction, 'direction', bool)


    # A 2D Fan-beam sinogram from the 3D Cone-beam data.
    if direction:
        fan_sinogram = data.get_slice(horizontal=idx, force=True).as_array()
    else:   
        fan_sinogram = data.get_slice(vertical=idx, force=True).as_array()

    # Rotation increment of the projection angles
    rotation_increment = float(data.geometry.angles[1] - data.geometry.angles[0])
    # print(f'Rotation increment: {rotation_increment}')

    # Source-Origin-distance in pixels
    D = source_origin_distance / detector_pixel_size

    parallel_sinogram, parallel_detector_positions, parallel_angles_deg = fan_to_parallel(
        F=fan_sinogram,
        D=D,
        FanRotationIncrement=rotation_increment
        )
    
    return parallel_sinogram, parallel_detector_positions, parallel_angles_deg


def fan_to_parallel(
    F: np.ndarray,
    D: float,
    FanSensorGeometry: str = 'line',
    FanSensorSpacing: float = 1.0,
    FanRotationIncrement: float = 1.0,
    FanCoverage: str = 'cycle',
    Interpolation: str = 'linear',
    ParallelSensorSpacing: float = None,
    ParallelCoverage: str = 'halfcycle',
    ParallelRotationIncrement: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Convert fan-beam sinogram to parallel-beam sinogram.

    Python version of Matlab's fan2para function to convert CIL's 2D 
    fan-beam sinogram projection slice to parallel-beam sinogram projection 
    slice. Matlab version documentation URL: 
    https://se.mathworks.com/help/images/ref/fan2para.html

    Parameters
    ----------
    F : np.ndarray
        Fan-beam sinogram in numpy.ndarray class format
    D : float
        Source-Origin-distance in pixels
    FanSensorGeometry : str, default "line"
        Type of the sensor geometry. If 'line', then the sensors are spaced 
        at equal distances along a line that is parallel to the x' axis. 
        Else, sensors are spaced at equal angles along a circular arc
    FanSensorSpacing : float, default 1.0
        Spacing of the sensors in the fan-beam detector
    FanRotationIncrement : float, default 1.0
        Fan-beam rotation increment in degrees
    FanCoverage : str, default "cycle"
        Range of fan-beam rotation. If 'cycle', rotates 360 degrees, else 
        rotates minimum amount to represent the sinogram
    Interpolation : str, default "linear"
        Type of interpolation used. Must be one of 'linear', 'spline', 
        'pchip' or 'nearest'
    ParallelSensorSpacing : float or None, default None
        Spacing of the sensors in the parallel-beam detector. If None, 
        the spacing is the same as in fan-beam
    ParallelCoverage : str, default "halfcycle"
        Range of parallel-beam rotation. 'halfcycle' is [0, 180) degrees 
        and 'cycle' is [0, 360) degrees
    ParallelRotationIncrement : float or None, default None
        Parallel-beam rotation increment in degrees. If None, the increment 
        is the same as in fan-beam

    Returns
    -------
    parallel_sinogram : np.ndarray
        Parallel-beam sinogram in np.ndarray format
    parallel_detector_positions : np.ndarray
        Parallel-beam detector sensor positions
    parallel_angles_deg : np.ndarray
        Parallel-beam projection angles in degrees

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """


    # Parameter validation
    validate_parameter(F, "F", np.ndarray)
    validate_parameter(D, "D", (int, float), must_be_positive=True)
    validate_parameter(FanSensorGeometry, "FanSensorGeometry", str)
    validate_parameter(FanSensorSpacing, "FanSensorSpacing", (int, float), 
                       must_be_positive=True)
    validate_parameter(FanRotationIncrement, "FanRotationIncrement", 
                       (int, float), must_be_positive=True)
    validate_parameter(FanCoverage, "FanCoverage", str)
    validate_parameter(Interpolation, "Interpolation", str)
    validate_parameter(ParallelSensorSpacing, "ParallelSensorSpacing", 
                       (int, float), allow_none=True, must_be_positive=True)
    validate_parameter(ParallelCoverage, "ParallelCoverage", str)
    validate_parameter(ParallelRotationIncrement, "ParallelRotationIncrement", 
                       (int, float), allow_none=True, must_be_positive=True)
    
    # --- Fan-Beam Geometry Setup ---
    num_fan_angles, num_detectors = F.shape
    
    # Original fan-beam projection angles
    theta_deg_orig = np.arange(num_fan_angles) * FanRotationIncrement
    
    # Fan-beam detector positions
    detector_center_idx = (num_detectors - 1) // 2
    detector_positions = (np.arange(num_detectors) - detector_center_idx) * FanSensorSpacing
    
    # Convert detector positions to fan angles
    if FanSensorGeometry == 'line':
        # For linear detector array
        fan_angles_deg = np.rad2deg(np.arctan(detector_positions / D))
    else:
        # For arc detector array
        fan_angles_deg = detector_positions

    # --- Parallel-Beam Grid Setup ---
    # Set default parallel-beam rotation increment if not specified
    if ParallelRotationIncrement is None:
        ParallelRotationIncrement = FanRotationIncrement
    
    # Define parallel-beam projection angles
    if ParallelCoverage == 'halfcycle':
        parallel_angles_deg = np.arange(0, 180, ParallelRotationIncrement)
    else:
        parallel_angles_deg = np.arange(0, 360, ParallelRotationIncrement)
    
    # Calculate parallel-beam detector positions
    fan_angles_rad = np.deg2rad([fan_angles_deg.min(), fan_angles_deg.max()])
    parallel_range = D * np.sin(fan_angles_rad)
    parallel_min, parallel_max = parallel_range
    
    # Set default parallel sensor spacing if not specified
    if ParallelSensorSpacing is None:
        ParallelSensorSpacing = FanSensorSpacing
    
    # Create parallel-beam detector position array
    num_parallel_detectors = int(np.ceil((parallel_max - parallel_min) / ParallelSensorSpacing)) + 1
    parallel_center = (parallel_max + parallel_min) / 2
    parallel_half_width = (num_parallel_detectors - 1) / 2 * ParallelSensorSpacing
    parallel_detector_positions = np.linspace(
        parallel_center - parallel_half_width, 
        parallel_center + parallel_half_width, 
        num_parallel_detectors
    )

    # --- Data Interpolation Setup ---
    fan_data_interp = F
    fan_angles_interp = theta_deg_orig

    # Apply circular padding for full cycle coverage
    if FanCoverage == 'cycle':
        padding_size = int(np.ceil(num_fan_angles / 4))
        
        # Pad the fan-beam data array
        data_pad_start = F[-padding_size:, :]
        data_pad_end = F[:padding_size, :]
        fan_data_interp = np.vstack([data_pad_start, F, data_pad_end])
        
        # Pad the corresponding angles, wrapping around 360 degrees
        angles_pad_start = theta_deg_orig[-padding_size:] - 360
        angles_pad_end = theta_deg_orig[:padding_size] + 360
        fan_angles_interp = np.hstack([angles_pad_start, theta_deg_orig, angles_pad_end])
    
    # Set up interpolation method
    interp_method_map = {
        'linear': 'linear', 
        'spline': 'cubic', 
        'pchip': 'pchip', 
        'nearest': 'nearest'
    }
    interp_method = interp_method_map.get(Interpolation, 'linear')

    # --- Two-Step Interpolation Process ---
    
    # Step 1: Angular interpolation (fan angles to parallel angles)
    angular_interp_data = np.zeros((num_detectors, len(parallel_angles_deg)))
    for detector_idx in range(num_detectors):
        # Shift angles by fan angle for this detector
        shifted_angles = fan_angles_interp - fan_angles_deg[detector_idx]
        
        # Interpolate to parallel projection angles
        interp_func = interp1d(
            shifted_angles, fan_data_interp[:, detector_idx], 
            kind=interp_method, bounds_error=False, fill_value=0.0
        )
        angular_interp_data[detector_idx, :] = interp_func(parallel_angles_deg)
        
    # Step 2: Spatial interpolation (fan positions to parallel positions)
    fan_detector_parallel_pos = D * np.sin(np.deg2rad(fan_angles_deg))
    parallel_sinogram = np.zeros((len(parallel_detector_positions), len(parallel_angles_deg)))
    
    for angle_idx in range(len(parallel_angles_deg)):
        # Interpolate from fan detector positions to parallel detector positions
        interp_func = interp1d(
            fan_detector_parallel_pos, angular_interp_data[:, angle_idx], 
            kind=interp_method, bounds_error=False, fill_value=0.0
        )
        parallel_sinogram[:, angle_idx] = interp_func(parallel_detector_positions)
    
    print(f'Rebinning is complete.')
    print(f'Shape of the parallel-beam sinogram: {parallel_sinogram.shape}.')
    print(f'Shape of the parallel-beam sensor locations vector: {parallel_detector_positions.shape}')
    print(f'Shape of the parallel-beam angles: {parallel_angles_deg.shape}.')
    plt.figure()
    plt.imshow(parallel_sinogram,
            aspect='auto',
            cmap='gray')
    plt.title("Rebinned Parallel-beam Sinogram")
    plt.show()

    return parallel_sinogram, parallel_detector_positions, parallel_angles_deg


def reconstruct_parallel_sinogram(
        parallel_sinogram: np.ndarray,
        parallel_angles: np.ndarray,
        filter_name: str,
        interpolation: str,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        cmap: str = 'gray'
        ) -> np.ndarray:

    """
    Function reconstructs the projection image using skimage's iradon function.
    The function is intended for parallel-beam sinograms.

    Parameters
    ----------
    parallel_sinogram : np.ndarray
        Parallel-beam sinogram in numpy.darray class format
    parallel_angles : np.ndarray
        Parallel-beam projection angles in numpy.darray class format
    filter_name : str
        Filter for the reconstructions. Possible filters are 'hamming', 'hann', 'cosine', 'ramp' and 'shepp-logan'
    interpolation : str
        Interpolation method for the reconstruction. Possible methods are 'linear', 'nearest' and 'cubic'
    lower_bound : int or float, default 0.0
        Lower bound of the image color
    upper_bound : int or float, default 1.0
        Upper bound of the image color
    cmap : str, default "gray"
        Colormap for the image. Possible colormaps are for example 'gray', 'hot', 'inferno'

    Returns
    -------
    recon : np.ndarray
        Reconstruction of the projection image in numpy.darray class format
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    validate_parameter(parallel_sinogram, "parallel_sinogram", np.ndarray)
    validate_parameter(parallel_angles, "parallel_angles", np.ndarray)
    validate_parameter(filter_name, "filter_name", str)
    validate_parameter(interpolation, "interpolation", str)
    validate_parameter(lower_bound, "lower_bound", (int, float))
    validate_parameter(upper_bound, "upper_bound", (int, float))
    if upper_bound < lower_bound:
        raise ValueError("Upper bound needs to greater than lower bound.")
    validate_parameter(cmap, "cmap", str)

    # Reconstuction of the image
    recon = iradon(parallel_sinogram, theta=parallel_angles, filter_name=filter_name, interpolation=interpolation)

    print('Reconstruction complete.')
    plt.figure()
    plt.imshow(
        np.fliplr(recon),  # Horizontal flip (mirror left-right)
        cmap=cmap,
        vmin=lower_bound,
        vmax=upper_bound,
        origin='upper')
    plt.title('Parallel-beam reconstruction using rebinned sinogram')
    plt.show()

    return recon