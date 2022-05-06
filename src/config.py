import os
import toml

def get_config():
  """Loads TOML configuration file from a location specified through the CONFIG_FILE environment variable. Defaults to 'config.toml' in the project's root folder.
  
  Returns:
      TYPE: Description
  """
  config_path = os.getenv("CONFIG_FILE", "conf.toml")
  return toml.load(config_path)
