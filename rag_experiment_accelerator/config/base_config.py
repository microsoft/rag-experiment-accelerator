from copy import deepcopy
from typing import get_type_hints
from itertools import product

import random


from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class BaseConfig:
    def __init__(self) -> None:
        self.__sampling_attributes: list[str] = []
        self.__static_atributes: list[str] = []
        self.__nested_configs: list[str] = []

        self.__gather_sampling_attributes()

    def __gather_sampling_attributes(self) -> None:
        """
        Gather all sampling attributes from the config.
        """
        sampling_attributes = []
        for key, value in get_type_hints(self).items():
            if key == "__sampling_attributes":
                continue

            attr_value = getattr(self, key)
            if type(attr_value) is value and isinstance(attr_value, list):
                if isinstance(attr_value[0], BaseConfig):
                    error_message = f"Nested config found in attribute {key} that is list. This is not allowed."
                    logger.critical(error_message)
                    raise ValueError(error_message)

                self.__sampling_attributes.append(key)
            else:
                if type(attr_value) is value and not isinstance(attr_value, list):
                    logger.warning(
                        f"Attribute {key}={attr_value} is not a list. Attribute will be treated as static."
                    )
                if isinstance(attr_value, BaseConfig):
                    self.__nested_configs.append(key)
                else:
                    self.__static_atributes.append(key)

        return sampling_attributes

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseConfig":
        """
        Create a config object from a dictionary.
        """
        config = cls()

        type_hints = get_type_hints(cls)

        for key, value in config_dict.items():
            if issubclass(type_hints[key], BaseConfig):
                if isinstance(value, list):
                    vals = [type_hints[key].from_dict(item) for item in value]
                else:
                    vals = type_hints[key].from_dict(value)
                setattr(config, key, vals)
            else:
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
                config_dict[key] = attr_value.to_dict()
            else:
                config_dict[key] = getattr(self, key)

        return config_dict

    def flatten(self):
        obj_copy = deepcopy(self)

        static_iterations = [getattr(self, attr) for attr in self.__sampling_attributes]
        nested_iterations = [
            list(getattr(self, attr).flatten()) for attr in self.__nested_configs
        ]

        vals_to_iterate = static_iterations + nested_iterations

        for values in product(*vals_to_iterate):
            for idx, attr in enumerate(self.__sampling_attributes):
                setattr(obj_copy, attr, values[idx])
            for idx, attr in enumerate(self.__nested_configs):
                set_idx = len(static_iterations)
                setattr(obj_copy, attr, values[set_idx + idx])

            yield obj_copy

    def sample(self):
        obj_copy = deepcopy(self)

        vals = {}
        for attr in self.__sampling_attributes:
            attrs_list = getattr(self, attr)
            vals[attr] = random.choice(attrs_list)

        for attr in self.__nested_configs:
            nested_config = getattr(self, attr)
            vals[attr] = nested_config.sample()

        for attr in vals:
            setattr(obj_copy, attr, vals[attr])

        return obj_copy
