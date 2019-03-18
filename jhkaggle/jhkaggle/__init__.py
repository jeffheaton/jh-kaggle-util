import json
import os
from pathlib import Path

jhkaggle_config = {}

def load_config(profile,filename = None):
  global jhkaggle_config
  if not filename:
    home = str(Path.home())
    filename = os.path.join(home,".jhkaggleConfig.json")
    if not os.path.isfile(filename):
      raise Exception(f"If no 'filename' paramater specifed, assume '.jhkaggleConfig.json' exists at HOME: {home}")

  with open(filename) as f:
      data = json.load(f)
      if profile not in data:
        raise Exception(f"Undefined profile '{profile}' in file '{filename}'")
      jhkaggle_config = data[profile]