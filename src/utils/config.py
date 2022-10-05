# src/utils/config.py

import json

class Config:
    """Configuration class to load parameters from a JSON file.
    """
    def __init__(self, config_path):
        """Initializes the Config class.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def __getattr__(self, name):
        """Allows accessing configuration parameters as attributes.

        Args:
            name (str): Name of the configuration parameter.

        Returns:
            The value of the configuration parameter.

        Raises:
            AttributeError: If the configuration parameter does not exist.
        """
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"No such attribute: {name}")
