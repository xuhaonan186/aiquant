import os.path

import torch
import pandas as pd
import torch.nn as nn
from aiquant.model.base import Model
from users.xhn.utils.log import logger
from users.xhn.utils.config import cache_path
import torch.optim as optim
import copy
import numpy as np


class MLP(Model):
    def __init__(self,
                 model_name,
                 d_feat=6,
                 hidden_size=64,
                 dropout=0.0,
                 n_epochs=200,
                 lr=0.001,
                 metric="",
                 batch_size=2000,
                 early_stop=20,
                 loss="mse",
                 optimizer="adam",
                 GPU=0,
                 **kwargs):

        # Set logger.
        self.logger = logger
        self.logger.info("GRU pytorch version...")

        self.model_name = model_name
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

        self.mlp_model = MLPModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size
        )
        self.logger.info("model:\n{:}".format(self.mlp_model))
        self.train_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.lr)
        self.fitted = False
        self.mlp_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def save_model(self, param, date):
        save_path = f"{cache_path}/saved_models/{self.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(param, f"{save_path}/{date}.pt")

    def load_model(self, date):
        load_path = f"{cache_path}/saved_models/{self.model_name}"
        self.mlp_model.load_state_dict(torch.load(f"{load_path}/{date}.pt"))

    def save_yhats(self, df, date):
        save_path = f"{cache_path}/yhats/{self.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_parquet(f"{save_path}/{date}.parquet")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.mlp_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i: i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i: i + self.batch_size]]).float().to(self.device)

            pred = self.mlp_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.mlp_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.mlp_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i: i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i: i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.mlp_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, segment):
        df_train, df_valid = dataset.prepare(segment, train=True)
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train.iloc[:, :-1], df_train["label"]
        x_valid, y_valid = df_valid.iloc[:, :-1], df_valid["label"]

        # save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.mlp_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.mlp_model.load_state_dict(best_param)
        self.save_model(best_param, segment[-1])

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, segment, save=True):
        if not self.fitted:
            self.load_model(segment[-1])
            # raise ValueError("model is not fitted yet!")

        index, x_test = dataset.prepare(segment, train=False)
        self.mlp_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.mlp_model(x_batch).detach().cpu().numpy()

            preds.append(pred)
        # result = pd.DataFrame(np.concatenate(preds), index=index)
        result = pd.concat([index, pd.DataFrame(np.concatenate(preds))], axis=1)
        result.columns = ['symbol', 'trade_dt', 'value']
        if save:
            self.save_yhats(result, segment[-1])
        return result

    def rolling_fit(self, dataset, scheduler):
        sche = scheduler.get_scheduler()
        for segment in sche:
            self.fit(dataset, segment)
            # result = self.predict(dataset, segment)


class MLPModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64):
        super().__init__()
        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.ac = nn.Sigmoid()
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        x = self.ac(self.fc_in(x))
        out = self.fc_out(x)
        return out
