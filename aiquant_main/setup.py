# -*- coding:utf-8 -*-
# Author: ranpeng
# Date: 2023/11/23


from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    name = 'aiquant',
    version = '0.0.1',
    author = 'xhn',
    description= 'test',
    packages = find_packages(),
    package_data = {"":["*"]}, #打包全部数据
    include_package_data = True,
    zip_safe = False,
    install_requires = [
    ],
    # ext_modules=cythonize(["cscdata/ckClient.py", "cscdata/utils.py"]),

)
