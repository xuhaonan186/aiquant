import pandas as pd
import numpy as np
from users.xhn.datasets.ai_dataset_xhn_001 import DatasetH


class TSDataSampler:
    def __init__(self, data, step_len):
        self.data = data
        self.step_len = step_len
        self.data_arr = None
        self.data_index = None
        # self.idx_df, self.idx_map = self.build_index(data)
        self.idx_df = self.build_fut_index(data)
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)
        self.nan_idx = 0  # nan补数

        self.data_arr = np.append(
            np.array(data),
            np.full((1, self.data.shape[1]), np.nan),
            axis=0,
        )

    @staticmethod
    def build_fut_index(data):
        idx_df = data.reset_index()
        idx_df = idx_df['index']
        return idx_df

    @staticmethod
    def build_index(data):
        """index(fut+trade_dt)+feature， index提取"""
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = idx_df.unstack()
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = idx_df.T

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    def _get_indices(self, row):
        """
        :param row: the row in self.idx_df
        :return: The indices of data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])
        # 需增加数据处理操作
        return indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            indices = [self._get_indices(i) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(idx)
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        if (np.diff(indices) == 1).all():  # slicing instead of indexing for speeding up.
            data = self.data_arr[indices[0]: indices[-1] + 1]
        else:
            data = self.data_arr[indices]
        return data

    def __len__(self):
        return self.data.shape[0]


class TSDatasetH(DatasetH):
    def __init__(self, step_len, **kwargs):
        self.step_len = step_len
        super().__init__(**kwargs)

    def _prepare_seg(self, start_dt, end_dt):
        data_df = self.data[(self.data.trade_dt >= start_dt) & (self.data.trade_dt <= end_dt)]
        data_df = data_df.iloc[:, 5:].reset_index(drop=True)
        tsds = TSDataSampler(
            data=data_df,
            step_len=self.step_len,
        )
        return tsds

    def prepare(self, segment, train,  **kwargs):
        train_sdt, train_edt, test_sdt, test_edt = segment
        if train:
            start_dt, end_dt = train_sdt, train_edt
        else:
            start_dt, end_dt = test_sdt, test_edt
        return self._prepare_seg(start_dt, end_dt)
