# A library to convert transmission data to the absoprtion data.
from cil.processors import Binner, TransmissionAbsorptionConverter # type: ignore
from cil.utilities.display import show2D # type: ignore

def convert_data(acquistion_data, white_level, binning=4):

    """
    A function converts CIL's transmission data to the absorption data.

    Args:
        acquistion_data: CIL's acquistion data.
        white_level: Intensity of the white level in the CT scan.

    Returns:
        data: Absorption data in CIL's acquistion data class format.
    """


    binner_processor = Binner(roi={'horizontal': (None, None, binning), 'vertical': (None, None, binning)})
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