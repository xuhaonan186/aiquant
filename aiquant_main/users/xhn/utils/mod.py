# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/17

"""
    实例化workflow得config
"""

import importlib

def init_instance_by_config(config):
    # 动态导入模块和类
    if config["module_path"] == "":
        module = importlib.import_module()
    else:
        module = importlib.import_module(config["module_path"])
    cls = getattr(module, config["class"])

    # 处理 kwargs 中的嵌套类实例化
    if "kwargs" in config:
        kwargs = config["kwargs"]
        for key, val in kwargs.items():
            if isinstance(val, dict) and "class" in val:
                # 递归调用 init_instance_by_config 来实例化嵌套类
                kwargs[key] = init_instance_by_config(val)
    else:
        kwargs = {}

    # 实例化并返回类的实例
    instance = cls(**kwargs)
    return instance
