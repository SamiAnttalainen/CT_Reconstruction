import numpy as np # type: ignore
from scipy.interpolate import griddata # type: ignore

def rebin_fan_to_parallel(fan_sinogram_data, geometry):
    """
    Correctly re-bins a fan-beam sinogram to a parallel-beam sinogram.

    This implementation includes:
    1. Cosine weighting of the fan-beam data.
    2. Correct calculation of parallel-beam coordinates (s, theta).
    3. 2D interpolation using scipy.interpolate.griddata for higher accuracy.

    Parameters:
    -----------
    fan_sinogram_data : ndarray
        2D fan-beam sinogram data (n_angles, n_detectors).
    geometry : object
        An object containing the scanner geometry. Expected attributes:
        - angles: The projection angles (beta) in degrees.
        - SOD: Source-to-Origin Distance (mm).
        - SDD: Source-to-Detector Distance (mm).
        - n_detectors: Number of detector pixels.
        - pixel_size: Size of a detector pixel (mm).

    Returns:
    --------
    parallel_sinogram : ndarray
        The rebinned parallel-beam sinogram.
    parallel_angles_deg : ndarray
        The new projection angles (theta) in degrees for the parallel sinogram.
    """
    # 1. Extract Data and Geometry
    # Ensure input is a numpy array
    fan_sinogram = fan_sinogram_data.as_array()
    if fan_sinogram.ndim != 2:
        raise ValueError(f"Input sinogram must be 2D, but got shape {fan_sinogram.shape}")

    n_beta, n_detectors_fan = fan_sinogram.shape
    
    # Get geometry parameters
    # beta_deg = geometry.angles
    beta_deg = np.linspace(0, 356.6666564941406, 108, endpoint=False)
    beta_rad = np.deg2rad(beta_deg)
    SOD = geometry.dist_source_center
    SDD = geometry.dist_center_detector + SOD
    
    # 2. Define Fan-Beam Coordinates (beta, gamma)
    
    # Detector coordinates (t)
    det_indices = np.arange(n_detectors_fan) - (n_detectors_fan - 1) / 2
    t = det_indices * geometry.pixel_size_h
    
    # Fan angles (gamma)
    gamma = np.arctan(t / SDD)


    #DEBUG prints
    print(f"Beta (degrees): {beta_deg[0]}, {beta_deg[-1]}")
    print(f"Gamma (degrees): {np.rad2deg(gamma[n_detectors_fan//2])}")
    print(f"theta (degrees): {np.rad2deg(beta_rad[0] + gamma[n_detectors_fan//2])}")
    print(f"s (mm): {SOD * np.sin(gamma[n_detectors_fan//2])}")
    print(f"s_gamma_max (mm): {SOD * np.sin(gamma.max())}")


    
    # 3. Apply Cosine Weighting to Fan-Beam Data
    # The weight is cos(gamma), broadcast across all projection angles.
    cosine_weights = np.cos(gamma)
    weighted_fan_sinogram = fan_sinogram * cosine_weights[np.newaxis, :]

    # 4. Calculate Corresponding Parallel-Beam Coordinates (s, theta) for each fan-beam ray
    
    # Create a meshgrid of all fan-beam coordinates
    beta_grid, gamma_grid = np.meshgrid(beta_rad, gamma, indexing='ij')
    
    # Calculate the new parallel coordinates for every point
    s_parallel = SOD * np.sin(gamma_grid)

    #DEBUG prints
    print("Detector t (mm):", t[:5], "...", t[-5:])  # Should be symmetric around 0
    print("Gamma (deg):", np.rad2deg(gamma[:5]), "...", np.rad2deg(gamma[-5:]))
    print("s_parallel (mm):", s_parallel[0, :5], "...", s_parallel[-1, -5:])

    theta_parallel_rad = np.mod(beta_grid + gamma_grid, 2*np.pi)
    # theta_parallel_rad = np.mod(beta_grid + gamma_grid, np.deg2rad(beta_deg[-1]))
    # theta_parallel_rad = beta_grid + gamma_grid  # No wrapping
    # sorted_order = np.argsort(theta_parallel_rad.ravel())

    
    # We now have a list of irregularly spaced points:
    # (theta_parallel_rad, s_parallel) are the coordinates, 
    # and weighted_fan_sinogram contains the values at these points.
    
    # Flatten the arrays to be used as input for griddata
    points = np.vstack((theta_parallel_rad.ravel(), s_parallel.ravel())).T
    values = weighted_fan_sinogram.ravel()
    # points = points[sorted_order]  # Sort points by theta
    # values = values[sorted_order]
    
    # 5. Define the Output Parallel-Beam Grid
    
    # Create the new regular grid where we want to interpolate the data
    n_theta_out = n_beta  # Typically keep the same number of angles
    n_s_out = n_detectors_fan  # And the same number of detector channels

    # The new theta angles should span the full range of the transformed angles
    theta_min, theta_max = np.min(points[:, 0]), np.max(points[:, 0])
    parallel_angles_rad = np.linspace(theta_min, theta_max, n_theta_out)
    
    # The new s-positions should span the range of the transformed positions
    s_min, s_max = np.min(points[:, 1]), np.max(points[:, 1])
    # s_max = SOD * np.sin(np.abs(gamma).max())
    parallel_s = np.linspace(s_min, s_max, n_s_out)
    # parallel_s = np.linspace(-s_max, s_max, n_s_out)
    
    # Create the meshgrid for the output (target)
    grid_theta, grid_s = np.meshgrid(parallel_angles_rad, parallel_s, indexing='ij')

    # 6. Perform 2D Interpolation
    
    print("Performing 2D interpolation... (this may take a moment)")
    parallel_sinogram = griddata(
        points,          # Irregular (theta, s) points from fan-beam
        values,          # Values at those points
        (grid_theta, grid_s), # Regular grid to interpolate onto
        method='linear',
        fill_value=0.0
    )

    print("Interpolation complete.")
    
    parallel_angles_deg = np.rad2deg(parallel_angles_rad)

    return parallel_sinogram, parallel_angles_deg


    # linear_interp = griddata(
    # points, values, (grid_theta, grid_s),
    # method='linear',
    # fill_value=np.nan
    # )
    # nan_mask = np.isnan(linear_interp)
    # if np.any(nan_mask):
    #     nearest_interp = griddata(
    #         points, values, (grid_theta, grid_s),
    #         method='nearest'
    #     )
    #     linear_interp[nan_mask] = nearest_interp[nan_mask]

    # parallel_sinogram = linear_interp

from scipy.ndimage import map_coordinates # type: ignore

def rebin_fan_to_parallel_map_coords(fan_sinogram_data, geometry):
    """
    Accurately re-bins a fan-beam sinogram using a backwards-mapping approach
    with scipy.ndimage.map_coordinates for high-quality interpolation.

    This method is often more accurate and computationally efficient than griddata.

    Parameters are the same as the previous function.
    """
    # 1. Extract Data and Geometry (same as before)
    fan_sinogram = fan_sinogram_data.as_array()
    n_beta, n_detectors_fan = fan_sinogram.shape
    
    beta_rad = np.deg2rad(geometry.angles)
    SOD = geometry.dist_source_center
    SDD = geometry.dist_center_detector + SOD
    
    # 2. Define Fan-Beam Detector/Angle Coordinates
    det_indices = np.arange(n_detectors_fan) - (n_detectors_fan - 1) / 2
    t = det_indices * geometry.pixel_size_h
    gamma = np.arctan(t / SDD)

    # 3. Apply Cosine Weighting
    cosine_weights = np.cos(gamma)
    weighted_fan_sinogram = fan_sinogram * cosine_weights[np.newaxis, :]

    # 4. Define the Output Parallel-Beam Grid
    n_theta_out = n_beta
    n_s_out = n_detectors_fan
    
    # Calculate the range of output angles and positions
    min_theta = np.min(beta_rad) + np.min(gamma)
    max_theta = np.max(beta_rad) + np.max(gamma)
    parallel_angles_rad = np.linspace(min_theta, max_theta, n_theta_out)
    
    s_max = SOD * np.sin(np.max(np.abs(gamma)))
    parallel_s = np.linspace(-s_max, s_max, n_s_out)
    
    # Create the meshgrid for the output (target) grid
    grid_theta, grid_s = np.meshgrid(parallel_angles_rad, parallel_s, indexing='ij')

    # 5. Backwards-Map from Parallel to Fan Coordinates
    # For each point in the parallel grid (grid_theta, grid_s), calculate
    # the corresponding coordinates in the original fan-beam grid.
    
    # Inverse transformation:
    # s_p = SOD * sin(gamma_f)  =>  gamma_f = asin(s_p / SOD)
    # theta_p = beta_f + gamma_f =>  beta_f = theta_p - gamma_f
    
    gamma_fan = np.arcsin(grid_s / SOD)
    beta_fan_rad = grid_theta - gamma_fan
    
    # 6. Convert Fan Coordinates to Pixel Indices
    # We need to map the physical coordinates (beta_fan_rad, gamma_fan) to the
    # array indices of `weighted_fan_sinogram`.
    
    # Beta indices (rows)
    # Assumes beta_rad is linearly spaced
    beta_start = beta_rad[0]
    beta_step = beta_rad[1] - beta_rad[0]
    beta_indices = (beta_fan_rad - beta_start) / beta_step
    
    # Gamma indices (columns)
    # Assumes gamma is linearly spaced (which it is)
    gamma_start = gamma[0]
    gamma_step = gamma[1] - gamma[0]
    gamma_indices = (gamma_fan - gamma_start) / gamma_step
    
    # Stack indices for map_coordinates. The format is [[row_coords], [col_coords]]
    coords = np.array([beta_indices, gamma_indices])

    # 7. Perform Interpolation using map_coordinates
    print("Performing interpolation with map_coordinates...")
    # map_coordinates handles interpolation and boundary conditions efficiently.
    # order=3 is cubic interpolation.
    # cval=0.0 sets the value for points outside the input sinogram boundaries.
    parallel_sinogram = map_coordinates(
        weighted_fan_sinogram,
        coords,
        order=3,
        mode='constant',
        cval=0.0,
        prefilter=True # Recommended for cubic interpolation
    )
    print("Interpolation complete.")

    parallel_angles_deg = np.rad2deg(parallel_angles_rad)
    
    return parallel_sinogram, parallel_angles_deg

from scipy.interpolate import interp1d # type: ignore

def fan2para(
    F,
    D_pixels,
    FanSensorGeometry='line',
    FanSensorSpacing=1.0,
    FanRotationIncrement=1.0,
    FanCoverage='cycle', # Added this parameter
    Interpolation='linear',
    ParallelSensorSpacing=None,
    ParallelCoverage='halfcycle',
    ParallelRotationIncrement=None
):
    
    # --- 1. Setup Fan-Beam Geometry (same as before) ---
    num_fan_angles, m = F.shape
    
    # ... (The rest of the initial setup is identical to the previous version)
    
    theta_deg_orig = np.arange(num_fan_angles) * FanRotationIncrement
    
    m2cn = (m - 1) // 2
    g = (np.arange(m) - m2cn) * FanSensorSpacing
    if FanSensorGeometry == 'line':
        gamma_deg = np.rad2deg(np.arctan(g / D_pixels))
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
    ploc_range = D_pixels * np.sin(gamma_range_rad)
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
        print("Using 'cycle' coverage. Padding sinogram data to fill corners.")
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
    t = D_pixels * np.sin(np.deg2rad(gamma_deg))
    P = np.zeros((len(ploc), len(ptheta_deg)))
    for i in range(len(ptheta_deg)):
        interp_func = interp1d(t, Fsh[:, i], kind=interp_kind,
                               bounds_error=False, fill_value=0.0)
        P[:, i] = interp_func(ploc)

    return P, ploc, ptheta_deg


def rebin_fan_to_parallel_accurate(
    F,
    sod_pixels,
    sdd_pixels,
    fan_sensor_spacing,
    fan_rotation_increment,
    fan_coverage='cycle'  # Added this parameter
):
    """
    A physically accurate fan-to-parallel rebinning function.
    
    V2: Now includes 'cycle' coverage logic to fill missing corners by
    wrapping data from a 360-degree scan.

    Args:
        F (ndarray): Fan-beam sinogram (angles x detectors).
        sod_pixels (float): Source-to-Origin Distance in pixels.
        sdd_pixels (float): Source-to-Detector Distance in pixels.
        fan_sensor_spacing (float): Physical spacing between detectors.
        fan_rotation_increment (float): Gantry rotation increment in degrees.
        fan_coverage (str): Set to 'cycle' to pad data for 360° scans.
    
    Returns:
        tuple: (P, ploc, ptheta_deg) - The rebinned data and its coordinates.
    """
    print("Running physically accurate rebinning with separate SOD/SDD and 'cycle' coverage...")
    num_fan_angles, m = F.shape
    
    # --- 1. Accurate Fan-Beam Geometry ---
    theta_deg_orig = np.arange(num_fan_angles) * fan_rotation_increment
    
    detector_indices = np.arange(m) - (m - 1) / 2
    detector_positions = detector_indices * fan_sensor_spacing
    gamma_rad = np.arctan(detector_positions / sdd_pixels)

    # --- 2. Setup Parallel-Beam Grid ---
    ptheta_deg = np.arange(0, 180, fan_rotation_increment)
    s_max = sod_pixels * np.sin(np.abs(gamma_rad).max())
    ploc = np.linspace(-s_max, s_max, m)

    # --- 3. Handle 'cycle' Coverage and Perform Interpolation ---
    F_interp = F
    theta_deg_interp = theta_deg_orig

    # **FIX**: Implement circular padding for 'cycle' coverage
    if fan_coverage == 'cycle':
        n4 = int(np.ceil(num_fan_angles / 4))
        
        # Pad the fan-beam data array
        F_pad_start = F[-n4:, :]
        F_pad_end = F[:n4, :]
        F_interp = np.vstack([F_pad_start, F, F_pad_end])
        
        # Pad the corresponding angles, wrapping around 360 degrees
        theta_pad_start = theta_deg_orig[-n4:] - 360
        theta_pad_end = theta_deg_orig[:n4] + 360
        theta_deg_interp = np.hstack([theta_pad_start, theta_deg_orig, theta_pad_end])

    # --- 4. Two-Step Interpolation ---
    Fsh = np.zeros((m, len(ptheta_deg)))
    for i in range(m):
        shifted_theta = theta_deg_interp - np.rad2deg(gamma_rad[i])
        interp_func = interp1d(shifted_theta, F_interp[:, i], kind='linear',
                               bounds_error=False, fill_value=0.0)
        Fsh[i, :] = interp_func(ptheta_deg)
        
    t = sod_pixels * np.sin(gamma_rad)
    P = np.zeros((len(ploc), len(ptheta_deg)))
    for i in range(len(ptheta_deg)):
        interp_func = interp1d(t, Fsh[:, i], kind='linear',
                               bounds_error=False, fill_value=0.0)
        P[:, i] = interp_func(ploc)

    return P, ploc, ptheta_deg
