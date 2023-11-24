# -*- coding:utf-8 -*-
# Author: xhn
# Date: 2023/11/22
import numpy as np
from aiquant.utils import init_instance_by_config


if __name__ == '__main__':
    task = {
        "model": {
            "class": "MLP",
            "module_path": "users.xhn.models.pytorch_mlp",
            "kwargs": {
                "model_name": "MLP_v1",
                "d_feat": 16,
                "loss": "mse",
                "lr": 0.001,
                "batch_size": 200,
                "early_stop": 5,
                "optimizer": "adam",
                "GPU": 0
                },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "users.xhn.datasets.ai_dataset_xhn_001",
            "kwargs": {
                "n_lag": 30,
                "tv_split": 0.8
            },
        },
        "scheduler": {
            "class": "XhnScheduler",
            "module_path": "users.xhn.schedulers.scheduler_xhn_001",
            "kwargs": {
                'start_date': 20151231,
                'end_date': 20230630,
                'ntrain_days': 1500,
                'rolling_days': 120
            }
        }

    }

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    scheduler = init_instance_by_config(task["scheduler"])
    model.rolling_fit(dataset, scheduler)
