# A library to read the .tif CT-scan data as Acquistion data.
from cil.io import TIFFStackReader # type: ignore
from cil.utilities.display import show2D # type: ignore

def read_ct_data(
        filename,
        acquistion_geometry,
        num_projections,
        skip=10
        ):
    
    """
    A function reads the data from the .tif files and makes returns the data in CIL's acquistion data class format.

    Args:
        filename: Folder path name to the .tif files
        acquistion_geometry: CIL's acquistion geometry variable
        num_projections: Number of projections used in the scan
        skip: Downsampling parameter for the acquistion data
    
    Returns:
        acquistion_data: Acquistion data from the .tif files
    """
    
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