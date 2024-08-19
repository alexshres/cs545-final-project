''' Utility Methods '''

def parse_data_line (data, normalization: float = 1.0) -> tuple:
    '''
    Parse the data splitting it into targets and features.  Entering a normalization value
      will normalize the features using the value provided.  If a zero is entered for the 
      normalization a ZeroDivisionError is raised.

    Input
    array-like:         Data to be parsed.
    float:              Normalization value.

    Output
    Tuple:              A tuple of (targets, features)
    '''
    if normalization == 0.0:
        raise ZeroDivisionError

    return (data[:, 0], data[:, 1:] / normalization)
