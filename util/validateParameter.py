"""A library to validate function parameters."""


def validate_parameter(value, name: str, expected_types, allow_none: bool = False, 
                       must_be_positive: bool = False) -> None:
    """
    Helper function for parameter validation.
    
    Parameters
    ----------
    value : Any
        A parameter that is validated
    name : str
        The name of the parameter as a string
    expected_types : Any
        An expected parameter type
    allow_none : bool
        Allows None values for the parameter if True, else disallows
    must_be_positive : bool
        Allows non-positive values if False, else disallows
    
    Returns
    -------
    None
    
    Raises
    ------
    TypeError
        If parameters are not of correct type
    ValueError
        If parameters are invalid or required parameters are missing
    """
    if allow_none and value is None:
        return
    
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = [t.__name__ for t in expected_types]
        else:
            type_names = [expected_types.__name__]
        
        if allow_none:
            type_names.append("None")
        
        type_str = " or ".join(type_names)
        raise TypeError(f"{name} must be {type_str}, got {type(value)}.")
    
    if must_be_positive and value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
