# -*- coding:utf-8 -*-
# Author: ranpeng
# Date: 2023/11/20




def run(ctx):
    on_init(ctx)
    for dt in ctx.ann_dts:
        on_ann(ctx, dt)