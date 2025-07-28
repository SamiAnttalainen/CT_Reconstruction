# A library to read the .tif CT-scan data as Acquistion data.
from cil.io import TIFFStackReader # type: ignore
from cil.utilities.display import show2D # type: ignore
from cil.framework import AcquisitionGeometry # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore

def read_ct_data(
        filename: str,
        acquistion_geometry: AcquisitionGeometry,
        num_projections: int,
        skip: int = 10
        ) -> AcquisitionData:
    
    """
    Reads .tif files and converts to CIL's
    acquisation data.

    Parameters
    ----------
    filename : str
        Folder path name to the .tif files
    acquistion_geometry : AcquistionGeometry
        CIL's acquistion geometry data class
    num_projections : int
        Number of projections used in the scan
    skip : int, default 10
        Downsampling parameter for the acquistion data
    
    Returns
    -------
    acquistion_data : AcquistionData
        Converted acquistion data from the .tif files
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string, got {type(filename)}.")
    if len(filename) == 0:
        raise ValueError(f"Invalid filename, got {filename}.")
    if not isinstance(acquistion_geometry, AcquisitionGeometry):
        raise TypeError(f"acquisition_geometry must be CIL's AcquistionGeometry data class, got {type(acquistion_geometry)}.")
    if not isinstance(num_projections, int):
        raise TypeError(f"num_projections must be an integer, got {type(num_projections)}.")
    if num_projections <= 0:
        raise ValueError(f"num_projections must be a positive integer, got {num_projections}.")
    if not isinstance(skip, int):
        raise TypeError(f"skip must be an integer, got {type(skip)}.")
    if skip <= 0:
        raise ValueError(f"skip must be a positive integer, got {skip}.")
    
    # Region of interest
    roi = {'axis_0': (0, num_projections, skip), 'axis_1': -1, 'axis_2': (0, 1000,1)}

    # .tif file reader setup
    reader = TIFFStackReader(file_name=filename, transpose=False, roi=roi)

    # Acquistion data from the .tif files
    acquistion_data = reader.read_as_AcquisitionData(acquistion_geometry)

    # Deleting the reader to save up memory
    del reader

    # Printing the raw data information and plotting the image slice of the transmission data
    print(acquistion_data)
    show2D(datacontainers=acquistion_data, origin='upper-right');
    return acquistion_data