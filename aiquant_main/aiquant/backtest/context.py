# -*- coding:utf-8 -*-
# Author: ranpeng
# Date: 2023/11/20

import datetime
from datetime import timedelta
from functools import cached_property, reduce

import pandas as pd
import numpy as np
from cscdata.auth import Connect
from ..utils.tools import import_module_from_path
from ..utils.decorator import timer

yesterday = int((datetime.date.today() - timedelta(days = 1)).strftime("%Y%m%d"))

# 金融数据处理和回测平台, 用于回测或者运维
class FinContext:
    """@DynamicAttrs"""
    @timer
    def __init__(self, start_dt=20070101, end_dt=yesterday, **kwargs):
        self.start_dt = start_dt
        self.end_dt = end_dt
        # self.mode = mode

        self.full_tds = None  # 全部交易日
        self.full_dts = None # 全部日历日
        self.trade_dts = None
        self.ann_dts = None

        self.nd = 0  # num of 交易日
        self.na = 0  # num of 日历日
        self.ns = 0  # num of 股票数量

        self.nd_dict = None  # trade_dts反查表
        self.na_dict = None  # ann_dts反查表
        self.ns_dict = None  # instruments反查表

        self.instruments = None

        self.warmup_days = kwargs.get('warmup_days', 0)
        self.con = None
        self.db = None

        # connection
        self.connect(**kwargs)
        self._init_from_wind()

        # env
        self.env_reset()

    def connect(self, **kwargs):
        self.con = Connect(kwargs)

    @cached_property
    def stock_cal(self):
        """stock calendar"""
        return self.ds('dsets.asharecalendar').get_df().reset_index()

    @cached_property
    def stock_des(self):
        return self.ds('dsets.asharedescription').get_df()

    def _init_from_wind(self):
        """init wind params"""
        self._init_dt()
        self._init_instrument()

    def _init_instrument(self):
        """init wind stkcode"""
        _stkcode_df = self.stock_des
        self.instruments = _stkcode_df[(_stkcode_df.delist_dt >= self.start_dt) & (_stkcode_df.list_dt <= self.end_dt)].reset_index().stockcode.values
        self.ns = len(self.instruments)
        self.ns_dict = pd.Series(np.arange(self.ns), self.instruments)

    def _init_dt(self):
        """init date"""
        _cal_df  = self.stock_cal
        self.full_tds = _cal_df.loc[_cal_df.exch == 'SSE'].sort_values('trade_dt')['trade_dt'].values
        self.full_dts = pd.date_range('19900101', str(self.end_dt)).strftime("%Y%m%d").astype(int).values
        self.trade_dts = self.full_tds.astype(int)[(self.full_tds >= self.start_dt) & (self.full_tds <= self.end_dt)]
        self.ann_dts = pd.date_range(str(self.start_dt), str(self.end_dt)).strftime("%Y%m%d").astype(int).values

        self.nd, self.na = len(self.trade_dts), len(self.ann_dts)
        self.nd_dict = pd.Series(np.arange(self.nd), self.trade_dts)
        self.na_dict = pd.Series(np.arange(self.na), self.ann_dts)

    @timer
    def get_listed_instruments(self, dt):
        """ get instruments """
        return self.ds("dsets.asharedescription").get_df(filters = [("delist_dt" , ">", dt)]).reset_index().stockcode.values

    @timer
    def get_listed_univ(self):
        """ 获取所有上市股票记录 """
        all_dates_df = pd.DataFrame({'trade_dt': self.trade_dts})
        df = self.ds("dsets.asharedescription").get_df().reset_index()[["stockcode", "list_dt", "delist_dt"]]
        # 将stockcode列与每个日期进行笛卡尔积
        df['key'] = 1
        all_dates_df['key'] = 1
        result = pd.merge(df, all_dates_df, on='key').drop('key', axis=1)

        result = result[(result['trade_dt'] >= result['list_dt']) & (result['trade_dt'] < result['delist_dt'])]
        result['value'] = 1
        result = result[['stockcode', 'trade_dt', 'value']]
        result = result.reset_index(drop=True)

        st_df = self.ds("dsets.asharest").get_df().reset_index()
        st_df['key'] =1
        st_df_result = pd.merge(st_df, all_dates_df, on = 'key').drop('key', axis= 1)
        st_df_result = st_df_result[(st_df_result['trade_dt'] >= st_df_result['entry_dt']) & (st_df_result["trade_dt"] < st_df_result['remove_dt'])]
        st_df_result = st_df_result[["stockcode", 'trade_dt']]

        result = pd.merge(result, st_df_result, on=["stockcode", "trade_dt"], how = 'left', indicator=True)
        result = (result[result['_merge'] == 'left_only']).drop(['_merge'],axis =1)

        return result

    def use_db(self, db_name):
        """use db"""
        self.db = self.con.repos.use_db(db_name)
        return self.db

    def ds(self, name):
        """return parquet table"""
        s_name = name.split('.')
        if len(s_name) == 1:
            table = self.db.use_table(s_name[0])
        elif len(s_name) == 2:
            table = self.con.repos.use_db(s_name[0]).use_table(s_name[1])
        else:
            raise KeyError
        return table

    def h5(self, name):
        """return h5 table"""
        return self.ds(name)

    def hist_df(self, name, n_day, day_col = "trade_dt", **kwargs):
        """get hist df
        kwargs: pd.read_parquet的参数
        """
        dfs_list = []
        _date_list = sorted(self.trade_dts, reverse=True)

        for _i in range(n_day):
            df = self.ds(name).get_df(filter = [(f"{day_col}", "=", _date_list[_i])], **kwargs)
            dfs_list.append(df)
        # total day
        return pd.concat(dfs_list, axis=0, ignore_index=True)

    def hist_sdf(self, name, n_day, spark_session, day_col = "trade_dt", ):
        """get hist sdf"""
        _date_list = sorted(self.trade_dts, reverse=True)
        df = self.ds(name).get_sdf(spark_session=spark_session)
        conditions = []

        from pyspark.sql.functions import col
        for _i in range(n_day):
            conditions.append(col[day_col] == _date_list[_i])

        return df.filter(reduce(lambda x, y: x & y, conditions))

    def is_trade_dt(self, dt):
        """ ann_dt in trade_dts"""
        if dt in self.trade_dts:
            return True
        return False

    def step(self):
        """更新环境"""
        self.cur_dt_i += 1
        self.cur_td_i += 1
        self.cur_dt = self.ann_dts[self.cur_dt_i]
        self.cur_td = self.trade_dts[self.cur_td_i]

    def env_reset(self):
        """环境重置"""
        self.cur_dt = None
        self.cur_td = None
        self.cur_dt_i = -1
        self.cur_td_i = -1

    def run(self, file):
        """run .py file
        """
        module = import_module_from_path(file)
        # functions
        module.on_init(self)
        for i, dt in enumerate(self.ann_dts):
            self.step()
            if i < self.warmup_days:
                continue
            module.on_ann_dt(self)
            if self.is_trade_dt(dt):
                module.on_trade_dt(self)
        self.env_reset()

    def intraday_run(self, file):
        """ tick和bar线上 """
        pass

def demo1(ctx):
    """全量的parquet读写"""
    df = ctx.ds('dsets.asharest').get_df()
    print(df)

    ctx.ds('rp.test_context').to_parquet(df)


def demo2(ctx):
    """全量的parquet读写"""
    pass

if __name__ == '__main__':
    ctx = FinContext()
    config = {
        "admin_repo": r"Q:\data_to_now",
        "researcher_repo": r"K:\qtData\cscdata_repo",
        "mode": 'researcher'
    }
    ctx._connect_from_config(**config)

    demo1(ctx)

