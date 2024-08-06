from copy import deepcopy
from typing import get_type_hints
from itertools import product

import random


from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class BaseConfig:
    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseConfig":
        """
        Create a config object from a dictionary.
        """
        config = cls()

        type_hints = get_type_hints(cls)

        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = type_hints[key].from_dict(value)
            elif isinstance(value, list):
                if len(value) and isinstance(value[0], dict):
                    value = [type_hints[key].from_dict(item) for item in value]

            setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """
        Convert the config to a dictionary.
        """
        config_dict = {}
        for key in get_type_hints(self):
            attr_value = getattr(self, key)
            if isinstance(attr_value, BaseConfig):
                attr_value = attr_value.to_dict()
            elif isinstance(attr_value, list):
                if len(attr_value) and isinstance(attr_value[0], BaseConfig):
                    attr_value = [item.to_dict() for item in attr_value]

            config_dict[key] = attr_value

        return config_dict

    def flatten(self, randomize: bool = False):
        """
        Flattens the configuration object by generating all possible combinations of attribute values.

        Args:
            randomize (bool, optional): Flag indicating whether to randomize the order of the generated combinations. Defaults to False.

        Yields:
            BaseConfig: A flattened configuration object with attribute values set to each combination.

        """
        key_values = {key: getattr(self, key) for key in get_type_hints(self)}
        sampling_key_values = {
            key: value
            for key, value in key_values.items()
            if isinstance(value, BaseConfig) or isinstance(value, list)
        }

        attribute_variations = {}
        for key, value in sampling_key_values.items():
            if isinstance(value, list):
                if len(value) and isinstance(value[0], BaseConfig):
                    list_of_lists = [list(value_item.flatten()) for value_item in value]
                    attribute_variations[key] = [
                        item for sublist in list_of_lists for item in sublist
                    ]
                else:
                    attribute_variations[key] = value
            else:
                attribute_variations[key] = list(value.flatten())

        attribute_names = list(attribute_variations.keys())

        combination_tuples = list(product(*(attribute_variations.values())))
        if randomize:
            random.shuffle(combination_tuples)

        for values in combination_tuples:
            obj_copy = deepcopy(self)

            for idx, attr in enumerate(attribute_names):
                setattr(obj_copy, attr, values[idx])

            yield obj_copy

    def sample(self) -> list["BaseConfig"]:
        """
        Returns one randomly selected flattened configuration from the base config.

        Returns:
            BaseConfig: A randomly selected flattened configuration.
        """
        return [next(self.flatten(randomize=True))]
