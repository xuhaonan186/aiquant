# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/20


import time

def timer(func):
    """装饰器，用于计时函数的执行时间"""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数结束执行的时间
        print(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result

    return wrapper