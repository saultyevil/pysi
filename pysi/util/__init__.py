"""Basic utility functions.

Functions which offer utility and are used project-wide belong in this
file.
"""

def remove_suffix_from_string(string: str, suffix: str) -> str:
    """Remove the provided suffix from a string.

    The string is only updated if the suffix is at the end of the string.

    Parameters
    ----------
    string : str
        The string to remove the suffix from.
    suffix : str
        The suffix to remove.

    Returns
    -------
    str
        The updated string.

    """
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string
