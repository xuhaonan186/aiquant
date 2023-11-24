import pandas as pd


class Dataset:
    """
    Preparing data for model training and inferencing.
    """
    def __init__(self, **kwargs):
        """init is designed to finish following steps:"""
        super().__init__()

    def setup_data(self, **kwargs):
        pass

    def prepare(self, **kwargs):
        pass


class DatasetH(Dataset):
    def __init__(self, tv_split, **kwargs):
        self.tv_split = tv_split
        self.setup_data(**kwargs)
        super().__init__(**kwargs)

    def setup_data(self, **kwargs):
        self.data = pd.read_parquet(r"K:\cta\cache\zn_daily.parquet").reset_index(drop=True)
        self.data['label'] = self.data['close'].shift(1)
        self.data['label'] = self.data['label']/1000
        self.data.drop(['sec_id'], axis=1, inplace=True)

    def prepare(self, segment, train,  **kwargs):
        train_sdt, train_edt, test_sdt, test_edt = segment
        if train:
            data_df = self.data[(self.data.trade_dt >= train_sdt) & (self.data.trade_dt <= train_edt)]
            data_df = data_df.iloc[:, 5:].reset_index(drop=True)
            train_df = data_df.iloc[:int(data_df.shape[0]*self.tv_split), :]
            valid_df = data_df.iloc[int(data_df.shape[0]*self.tv_split):, :].reset_index(drop=True)
            return train_df, valid_df
        else:
            data_df = self.data[(self.data.trade_dt >= test_sdt) & (self.data.trade_dt <= test_edt)]
            data_df = data_df.reset_index(drop=True)
            return data_df[['symbol', 'trade_dt']], data_df.iloc[:, 5:-1]
