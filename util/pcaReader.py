def read_pca_file(filepath, verbose = 0):
    """Function for reading pca file metadata about 
    the geometry and turning it into a dict.
    
    T H   2025"""
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