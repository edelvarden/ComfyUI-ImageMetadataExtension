import glob
import importlib
import os

from .captures import CAPTURE_FIELD_LIST
from .samplers import SAMPLERS

# load CAPTURE_FIELD_LIST and SAMPLERS in ext folder
dir_name = os.path.dirname(os.path.abspath(__file__))
ext_folder = os.path.join(dir_name, "ext")
parent_folder = os.path.basename(os.path.dirname(os.path.dirname(dir_name)))

for module_path in glob.glob(os.path.join(ext_folder, "*.py")):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    package_name = f"custom_nodes.{parent_folder}.py.defs.ext.{module_name}"
    module = importlib.import_module(package_name)
    CAPTURE_FIELD_LIST.update(getattr(module, "CAPTURE_FIELD_LIST", {}))
    SAMPLERS.update(getattr(module, "SAMPLERS", {}))
