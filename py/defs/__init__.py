import glob
import importlib.util
import os

from .captures import CAPTURE_FIELD_LIST
from .samplers import SAMPLERS

# Locate the current node's directory
dir_name = os.path.dirname(os.path.abspath(__file__))

ext_folder = os.path.join(dir_name, "ext")

# load CAPTURE_FIELD_LIST and SAMPLERS in ext folder
for path in glob.glob(os.path.join(ext_folder, "*.py")):
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(mod)

        CAPTURE_FIELD_LIST.update(getattr(mod, "CAPTURE_FIELD_LIST", {}))
        SAMPLERS.update(getattr(mod, "SAMPLERS", {}))
    except Exception as e:
        print(f"Error loading module {module_name}: {e}")
