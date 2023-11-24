import torch
import pandas as pd
import torch.nn as nn
from aiquant.model.base import Model
from ..utils.log import logger
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader
from ..utils.config import cache_path
import os


class GRU(Model):
    def __init__(self,
                 model_name,
                 d_feat=6,
                 hidden_size=64,
                 num_layers=2,
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

        self.model_name = model_name
        # Set logger.
        self.logger = logger
        self.logger.info("GRU pytorch version...")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

        self.gru_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.gru_model))
        self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        self.fitted = False
        self.gru_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def save_model(self, param, date):
        save_path = f"{cache_path}/saved_models/{self.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(param, f"{save_path}/{date}.pt")

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

    def train_epoch(self, data_loader):

        self.gru_model.train()

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            feature = feature.float()
            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.gru_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            feature = feature.float()
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, segment, save_path=None):
        df_train = dataset.prepare(segment, train=True)
        df_test = dataset.prepare(segment, train=False)
        train_loader = DataLoader(
            df_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last = True
        )
        test_loader = DataLoader(
            df_test,
            batch_size=self.batch_size
        )

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(test_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.gru_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.gru_model.load_state_dict(best_param)
        self.save_model(best_param, segment[-1])

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, **kwargs):
        pass

    def rolling_fit(self, dataset, scheduler):
        sche = scheduler.get_scheduler()
        for segment in sche:
            self.fit(dataset, segment)


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
