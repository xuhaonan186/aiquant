# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/22


import importlib.util
import sys
import os

def import_module_from_path(path):
    module_name = str(os.path.basename(path)).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
