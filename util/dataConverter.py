# A library to convert transmission data to the absoprtion data.
from cil.processors import Binner, TransmissionAbsorptionConverter # type: ignore
from cil.utilities.display import show2D # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore

def convert_data(
        acquistion_data: AcquisitionData,
        white_level: int,
        binning_parameter: int = 4
        ) -> AcquisitionData: # type: ignore

    """
    Transmission data to the absorption data converter.

    Parameters
    ----------
    acquistion_data : cil.framework.acquisition_data.AcquisitionData
        CIL's acquistion data
    white_level : int or float
        Intensity of the white level in the CT scan
    binning_parameter : int
        Data binning parameter

    Returns
    -------
    data : cil.framework.acquisition_data.AcquisitionData
        Absorption data in CIL's acquistion data class format.

    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """

    # Parameter validation
    if not isinstance(acquistion_data, AcquisitionData):
        raise TypeError(f"acquistion_data must be CIL's AcquisitionData class, got {type(acquistion_data)}.")
    if not isinstance(white_level, int):
        raise TypeError(f"white_level must be an integer, got {type(white_level)}.")
    if white_level<=0:
        raise ValueError(f"white_level must be a positive integer, {white_level}.")
    if not isinstance(binning_parameter, int):
        raise TypeError(f"binning_parameter must be an integer, got {type(binning_parameter)}.")
    if binning_parameter <= 0:
        raise ValueError(f"binning_parameter must be a positive integer, got {binning_parameter}.")

    binner_processor = Binner(roi={'horizontal': (None, None, binning_parameter), 'vertical': (None, None, binning_parameter)})
    binner_processor.set_input(acquistion_data)

    # Binned data
    data = binner_processor.get_output()

    print(f'{white_level = }, about {100*white_level / 2**14:.3f} of the maximum (of 14 bit int)')
    transmission_processor = TransmissionAbsorptionConverter(white_level=white_level)
    # transmission_processor = TransmissionAbsorptionConverter()
    transmission_processor.set_input(data)
    transmission_processor.get_output(out=data)

    show2D(data, origin='upper-right')
    print(data)

    # Freeing up the memory
    del acquistion_data
    del binner_processor
    del transmission_processor

    return data