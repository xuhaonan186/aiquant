# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/19



import lightgbm as lgb

class LGBModel:
    def __init__(self, loss, colsample_bytree, learning_rate, subsample, lambda_l1, lambda_l2, max_depth, num_leaves, num_threads):
        self.loss = loss
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.num_threads = num_threads
        self.model = None

    def fit(self, dataset):
        # 此处添加用于训练模型的代码
        # 例如:
        train_data = lgb.Dataset(dataset['train'][0], label=dataset['train'][1])
        valid_data = lgb.Dataset(dataset['valid'][0], label=dataset['valid'][1])

        params = {
            'objective': self.loss,
            'colsample_bytree': self.colsample_bytree,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'num_threads': self.num_threads
        }

        self.model = lgb.train(params, train_data, valid_sets=[valid_data])

    def predict(self, dataset):
        # 此处添加用于预测的代码
        # 例如:
        return self.model.predict(dataset['test'][0])