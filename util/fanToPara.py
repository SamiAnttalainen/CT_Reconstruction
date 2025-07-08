# A library to convert fan-beam data geometry to parallel-beam data geometry.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # type: ignore
from skimage.transform import iradon # type: ignore


def convert_fan_to_parallel_geometry(data, idx, source_origin_distance, detector_pixel_size=0.2):

    """
    A function converts CIL's 2D fan-beam sinogram projections into parallel-beam sinogram projections.

    Args:
        data: CIL's absorption data datacontainer.
        idx: Index of the slice.
        source_origin_distance: Source-Origin-distance in millimeters.
        detector_pixel_size: Detector pixel size in millimeters (default = 0.2 mm).

    Returns:
        Psinogram: Converted parallel-beam sinogram.
        Ploc: Parallel-beam sensor locations.
        Pangles: Parallel-beam projection angles.
    """

    # A 2D Fan-beam sinogram from the 3D Cone-beam data.
    Fsinogram = data.get_slice(vertical=idx, force=True).as_array()

    # Rotation increment of the projection angles
    rotation_increment = data.geometry.angles[1] - data.geometry.angles[0]
    # print(f'Rotation increment: {rotation_increment}')

    # Source-Origin-distance in pixels
    D = source_origin_distance / detector_pixel_size

    Psinogram, Ploc, Pangles = fan_to_parallel(
        F=Fsinogram,
        D=D,
        FanRotationIncrement=rotation_increment
        )
    
    return Psinogram, Ploc, Pangles


def fan_to_parallel(
    F,
    D,
    FanSensorGeometry='line',
    FanSensorSpacing=1.0,
    FanRotationIncrement=1.0,
    FanCoverage='cycle',
    Interpolation='linear',
    ParallelSensorSpacing=None,
    ParallelCoverage='halfcycle',
    ParallelRotationIncrement=None
):
    
    """
    A Python version of Matlab's fan2para function to convert CIL's 2D fan-beam sinogram projection slice to parallel-beam sinogram projection slice.
    Matlab version documentation URL: https://se.mathworks.com/help/images/ref/fan2para.html

    Args:
        F: Fan-beam sinogram in numpy.darray class format.
        D: Source-Origin-distance in pixels.
        FanSensorGeometry: Type of the sensor geometry. If 'line', then the sensors are spaced at equal distances
        along a line that is parallel to the x' axis. Else, Sensors are spaced at equal angles along a circular arc. 
        FanSensorSpacing: Spacing of the sensors in the fan-beam detector (default: 1.0).
        FanRotationIncrement: Fan-beam rotation increment in degrees.
        FanCoverage: Range of fan-beam rotation. If 'cycle', rotates 360 degrees, else rotates minimum amount to represent
        the sinogram.
        Interpolation: Type of interpolation used. The string has to be one of 'linear', 'spline', 'pchip' or 'nearest'.
        ParallelSensorSpacing: Spacing of the sensors in the parallel-beam detector. If None is specified, then the spacing is
        the same as in fan-beam.
        ParallelCoverage: Range of parallel-beam rotation. 'half-cycle' is [0, 180) degrees and 'cycle' is [0, 360) degrees.
        ParallelRotationIncrement: Parallel-beam rotation increment in degrees. If None is specified, then the increment is the
        same as in fan-beam.

    Returns:
        P: Parallel-beam sinogram sinogram.
        ploc: Parallel-beam sensor locations.
        ptheta_deg: Parallel-beam projection angles in degrees.
    """
    # --- 1. Setup Fan-Beam Geometry (same as before) ---
    num_fan_angles, m = F.shape
    
    # ... (The rest of the initial setup is identical to the previous version)
    
    theta_deg_orig = np.arange(num_fan_angles) * FanRotationIncrement
    
    m2cn = (m - 1) // 2
    g = (np.arange(m) - m2cn) * FanSensorSpacing
    if FanSensorGeometry == 'line':
        gamma_deg = np.rad2deg(np.arctan(g / D))
    else:
        gamma_deg = g

    # --- 2. Setup Parallel-Beam Grid (same as before) ---
    # ... (This section is also identical)
    if ParallelRotationIncrement is None:
        ParallelRotationIncrement = FanRotationIncrement
    if ParallelCoverage == 'halfcycle':
        ptheta_deg = np.arange(0, 180, ParallelRotationIncrement)
    else:
        ptheta_deg = np.arange(0, 360, ParallelRotationIncrement)
    # ... (ploc calculation is the same)
    gamma_range_rad = np.deg2rad([gamma_deg.min(), gamma_deg.max()])
    ploc_range = D * np.sin(gamma_range_rad)
    ploc_min, ploc_max = ploc_range
    if ParallelSensorSpacing is None:
        ParallelSensorSpacing = FanSensorSpacing
    num_ploc = int(np.ceil((ploc_max - ploc_min) / ParallelSensorSpacing)) + 1
    ploc_center = (ploc_max + ploc_min) / 2
    ploc_half_width = (num_ploc - 1) / 2 * ParallelSensorSpacing
    ploc = np.linspace(ploc_center - ploc_half_width, ploc_center + ploc_half_width, num_ploc)


    # --- 3. Handle 'cycle' Coverage and Perform Interpolation ---
    F_interp = F
    theta_deg_interp = theta_deg_orig

    # **NEW**: Implement MATLAB's circular padding for 'cycle' coverage
    if FanCoverage == 'cycle':
        # print("Using 'cycle' coverage. Padding sinogram data to fill corners.")
        n4 = int(np.ceil(num_fan_angles / 4))
        
        # Pad the fan-beam data array
        F_pad_start = F[-n4:, :]
        F_pad_end = F[:n4, :]
        F_interp = np.vstack([F_pad_start, F, F_pad_end])
        
        # Pad the corresponding angles, wrapping around 360 degrees
        theta_pad_start = theta_deg_orig[-n4:] - 360
        theta_pad_end = theta_deg_orig[:n4] + 360
        theta_deg_interp = np.hstack([theta_pad_start, theta_deg_orig, theta_pad_end])

    # ... (The two-step interpolation logic now runs on the padded data)
    
    interp_map = {'linear': 'linear', 'spline': 'cubic', 'pchip': 'pchip', 'nearest': 'nearest'}
    interp_kind = interp_map.get(Interpolation, 'linear')

    # Step 1: Angular Interpolation
    Fsh = np.zeros((m, len(ptheta_deg)))
    for i in range(m):
        shifted_theta = theta_deg_interp - gamma_deg[i]
        interp_func = interp1d(shifted_theta, F_interp[:, i], kind=interp_kind,
                               bounds_error=False, fill_value=0.0)
        Fsh[i, :] = interp_func(ptheta_deg)
        
    # Step 2: Spatial Interpolation
    t = D * np.sin(np.deg2rad(gamma_deg))
    P = np.zeros((len(ploc), len(ptheta_deg)))
    for i in range(len(ptheta_deg)):
        interp_func = interp1d(t, Fsh[:, i], kind=interp_kind,
                               bounds_error=False, fill_value=0.0)
        P[:, i] = interp_func(ploc)
    
    print(f'Rebinning is complete.')
    print(f'Shape of the parallel sinogram: {P.shape}.')
    print(f'Shape of the parallel angles: {ptheta_deg.shape}.')
    plt.figure(figsize=(8,6))
    plt.imshow(P,
            aspect='auto',
            cmap='hot')
    plt.title("Rebinned Parallel-beam Sinogram")
    plt.show()

    return P, ploc, ptheta_deg


def reconstruct_parallel_sinogram(Psinogram, Pangles, filter_name, interpolation, lb=0, ub=1, cmap='gray'):

    """
    Function reconstructs the projection image using skimage's iradon function. The function is intended for
    only parallel-beam sinograms.

    Args:
        Psinogram: Parallel-beam sinogram in numpy.darray class format.
        Pangles: Parallel-beam projection angles in numpy.darray class format.
        filter_name: Filter for the reconstructions. Possible filters are 'hamming', 'hann', 'cosine', 'ramp' and 'shepp-logan'.
        interpolation: Interpolation method for the reconstruction. Possible methods are 'linear', 'nearest' and 'cubic'.
        lb: Lower bound of the image color (Default: 0). 
        ub: Upper bound of the image color (Default: 1).
        cmap: Colormap for the image. Possible colormaps are for example 'gray', 'hot', 'inferno'.

    Returns:
        recon: Reconstruction of the projection image in numpy.darray class format.
    """

    # Reconstuction of the image
    recon = iradon(Psinogram, theta=Pangles, filter_name=filter_name, interpolation=interpolation)

    print('Reconstruction complete.')
    plt.figure(figsize=(7,7))
    plt.imshow(recon, cmap=cmap, vmin=lb, vmax=ub)
    plt.title('Parallel-beam reconstruction using rebinned sinogram')
    plt.show()

    return recon