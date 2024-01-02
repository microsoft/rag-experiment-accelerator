import os
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

def _mask_string(s: str, start: int = 2, end: int = 2, mask_char: str = "*") -> str:
    """
    Masks a string by replacing some of its characters with a mask character.

    Args:
        s (str): The string to be masked.
        start (int): The number of characters to keep at the beginning of the string.
        end (int): The number of characters to keep at the end of the string.
        mask_char (str): The character to use for masking.

    Returns:
        str: The masked string.

    Raises:
        None
    """
    if s is None or s == "":
        return ""

    if len(s) <= start + end:
        return s[0] + mask_char * (len(s) - 1)

    return (
        s[:start] + mask_char * (len(s) - start - end) + s[-end:]
        if end > 0
        else s[:start] + mask_char * (len(s) - start)
    )


def _get_env_var(var_name: str, critical: bool, mask: bool) -> str:
    """
    Get the value of an environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        critical (bool): Whether or not the function should raise an error if the variable is not set.
        mask (bool): Whether or not to mask the value of the variable in the logs.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the `critical` parameter is True and the environment variable is not set.
    """
    var = os.getenv(var_name, None)
    if var is None:
        logger.critical(f"{var_name} environment variable not set.")
        if critical:
            raise ValueError(f"{var_name} environment variable not set.")
    else:
        text = var if not mask else _mask_string(var)
        logger.info(f"{var_name} set to {text}")
    return var