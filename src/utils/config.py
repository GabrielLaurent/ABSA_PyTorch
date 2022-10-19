import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        else:
            raise AttributeError(f"Attribute '{name}' not found in config.")


    def get(self, section, key):
        try:
            return self.config[section][key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in section '{section}' in config.")