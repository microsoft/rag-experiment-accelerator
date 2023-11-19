from unittest.mock import patch
import pytest
from rag_experiment_accelerator.utils.utils import get_env_var, mask_string


@pytest.mark.parametrize(
    "input_string, start, end, mask_char, expected",
    [
        ("1234567890", 2, 2, "*", "12******90"),
        ("", 2, 2, "*", ""),
        ("123", 1, 1, "*", "1*3"),
        ("1234", 2, 2, "*", "1***"),
        ("12", 1, 1, "*", "1*"),
        ("1234", 0, 0, "*", "****"),
        ("abcd", 2, 2, "#", "a###"),
    ],
)
def test_mask_string(input_string, start, end, mask_char, expected):
    result = mask_string(input_string, start, end, mask_char)
    assert result == expected

@patch("rag_experiment_accelerator.utils.utils.logger")  # Replace with the actual import path to logger
@patch("os.getenv")
@pytest.mark.parametrize(
    "var_name, critical, mask, env_value, expected_value, expected_exception, expected_log",
    [
        ("TEST_VAR", True, False, "value", "value", None, "TEST_VAR set to value"),
        (
            "TEST_VAR",
            True,
            False,
            None,
            None,
            ValueError,
            "TEST_VAR environment variable not set.",
        ),
        ("TEST_VAR", True, True, "value", "value", None, "TEST_VAR set to va*ue"),
    ],
)
def test_get_env_var(
    mock_getenv,
    mock_logger,
    var_name,
    critical,
    mask,
    env_value,
    expected_value,
    expected_exception,
    expected_log,
):
    mock_getenv.return_value = env_value
    if expected_exception:
        with pytest.raises(expected_exception):
            get_env_var(var_name, critical, mask)
    else:
        assert get_env_var(var_name, critical, mask) == expected_value
        mock_logger.info.assert_called_with(expected_log)
