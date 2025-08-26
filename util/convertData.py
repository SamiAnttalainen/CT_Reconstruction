# A library to convert transmission data to the absoprtion data.
from cil.processors import Binner, TransmissionAbsorptionConverter # type: ignore
from cil.utilities.display import show2D # type: ignore
from cil.framework.acquisition_data import AcquisitionData # type: ignore
from util.validateParameter import validate_parameter

def convert_ct_data(
        acquisition_data: AcquisitionData,
        white_level: int,
        binning_parameter: int = 4
        ) -> AcquisitionData: # type: ignore

    """
    CT transmission data to absorption data converter.

    Parameters
    ----------
    acquisition_data : cil.framework.acquisition_data.AcquisitionData
        CIL's CT acquistion data
    white_level : int
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
    validate_parameter(acquisition_data, 'acquisition_data', AcquisitionData)
    validate_parameter(white_level, 'white_level', int, must_be_positive=True)
    validate_parameter(binning_parameter, 'binning_parameter', int, must_be_positive=True)

    # Region of interest
    roi={
        'horizontal': (None, None, binning_parameter),
        'vertical': (None, None, binning_parameter)
        }

    binner_processor = Binner(roi=roi)
    binner_processor.set_input(acquisition_data)

    # Binned acquisition data
    binned_data = binner_processor.get_output()

    print(f'{white_level = }, about {100*white_level / 2**14:.3f} of the maximum (of 14 bit int)')
    transmission_processor = TransmissionAbsorptionConverter(white_level=white_level)
    transmission_processor.set_input(binned_data)
    transmission_processor.get_output(out=binned_data)

    show2D(binned_data, origin='upper-right')
    print(binned_data)

    # Freeing up the memory
    del acquisition_data
    del binner_processor
    del transmission_processor

    return binned_data


def convert_cl_data(
        acquisition_data: AcquisitionData,
        white_level: int,
        binning_parameter: int = 4
        ) -> AcquisitionData: # type: ignore

    """
    CL transmission data to absorption data converter.

    Parameters
    ----------
    acquisition_data : cil.framework.acquisition_data.AcquisitionData
        CIL's CL acquistion data
    white_level : int
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
    validate_parameter(acquisition_data, 'acquisition_data', AcquisitionData)
    validate_parameter(white_level, 'white_level', int, must_be_positive=True)
    validate_parameter(binning_parameter, 'binning_parameter', int, must_be_positive=True)

    print(f'{white_level = }, about {100*white_level / 2**14:.3f} of the maximum (of 14 bit int)')
    transmission_processor = TransmissionAbsorptionConverter(white_level=white_level)
    attenuation_data = transmission_processor(acquisition_data)

    # Region of interest
    roi={
        'horizontal': (None, None, binning_parameter),
        'vertical': (None, None, binning_parameter)
        }

    # Data binning
    binner_processor = Binner(roi=roi)
    binner_processor.set_input(attenuation_data)
    binned_acq_data = binner_processor.get_output()
    show2D(binned_acq_data, origin='upper-left')
    print(binned_acq_data)

    # Freeing up the memory
    del acquisition_data
    del binner_processor
    del transmission_processor

    return binned_acq_data