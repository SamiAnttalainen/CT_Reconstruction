from util.validateParameter import validate_parameter

def read_pca_file(filepath : str, verbose: int = 0) -> dict:
    """
    Function for reading pca file metadata about 
    the geometry and turning it into a dict.

    Parameters
    ----------
    filepath : str
        Filepath of the PCA file
    verbose : int, default 0
        Verbose parameter. If verbose > 0, then prints conversions. If verbose > 1,
        then prints all lines
    
    Returns
    -------
    out : dict
        Parameter dictionary for the CT-scan parameters
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    
    T H   2025,
    Edited: S A 2025
    """

    # Parameter validation
    validate_parameter(filepath, 'filepath', str)
    if len(filepath) == 0:
        raise ValueError(f"Invalid file path, got {filepath}.")
    validate_parameter(verbose, 'verbose', int)
    if verbose < 0:
        raise ValueError(f"verbose must be a non-negative integer, got {verbose}.")


    out = {}
    # Important floating point values
    floatKeys = ["FDD", "FOD", "cx", "cy", "Tilt", "Oblique", "CalibValue", 
                 "DetectorRot", "RotationSector", "PlanarCTRotCenter",
                 "PlanarCTObjectHeight", "ROILowerHeight", "ROIUpperHeight",
                 "ROIWidthX", "ROIWidthY", "ROIOffsetX", "ROIOffsetY",
                 "DimX", "DimY", "PixelsizeX", "PixelsizeY", "Voltage", "Current",
                 "CenterX", "CenterY"]
    # Important integer values
    intKeys = ["NumberImages", "StartImg", "FreeRay"]
    # Important string values
    strKeys = ["Version-pca"]
    nLines  = 0
    keyVals = 0
    useless = 0
    header = ""
    with open(file=filepath, mode='r') as file:
        for line in file: # Read each line
            nLines += 1
            keyValue = line.split("=", 1)
            if len(line) < 2:
                continue
            elif len(keyValue) == 1: # No "=" to break in to "key = value" pair
                header = line[1:-2]
                if verbose > 0:
                    print(f"== {header} ==")
                continue
            if verbose > 1:
                print(f"Line {nLines}: {line}")
            keyVals += 1
            key = keyValue[0]
            value = keyValue[1]
            
            if key in floatKeys:
                fVal = float(value)
                if verbose > 0:
                    print(f"{key}: {value} converted to {fVal}")
                out.update({key : fVal})
            elif key in intKeys:
                iVal = int(value)
                if verbose > 0:
                    print(f"{key}: {value} converted to {iVal}")
                if header == "CalibValue":
                    key = key+"Calib"
                out.update({key : iVal})
            elif key in strKeys:
                out.update({key : value})
            else:
                useless += 1
        # All lines read
        print(f"{nLines} lines, found {keyVals} values (discarded {useless})")
    return out