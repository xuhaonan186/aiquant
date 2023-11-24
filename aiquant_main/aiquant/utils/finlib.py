# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/21

import numpy as np
import pandas as pd
from functools import lru_cache

######################### 股票ctx相关常规函数 #########################
def shift_tds(ctx, dts, n):
    '''把日期变成，前后n个交易日， n不能为0
    n = -1： 向前找最近的一个交易日，
    n = 0： 向后找最近的一个交易日
    如果不在交易日为首日 越界返回0和99999999
    np.searchsorted会找到最近日期靠后的一天的位置
    '''
    dts = dts.values
    tds = ctx.full_tds.copy()
    loc = np.searchsorted(tds, dts) + n
    overupper = loc > len(tds) - 1
    belowlower = loc < 0
    normal = ~(overupper | belowlower)

    result = np.ones_like(dts)
    result[normal] = tds[loc[normal]]
    result[overupper] = 99999999
    result[belowlower] = 0
    return result

def cal2td(ctx, dts):
    ''' 日历日期转换为trade dt '''
    return ctx.trade_dts[ctx.dt_loc(dts)]

def dt_loc(ctx, dts):
    ''' 获取 dts 在 trade_dts 上的位置，如果找不到取后值位置
    nan的位置变nan
    '''
    loc = np.searchsorted(ctx.trade_dts, dts)
    return np.clip(loc, a_min=0, a_max=ctx.nd - 1)

def pivot_dup(ctx, df, aggfunc='sum'):
    ''' pivot: with duplicates '''
    df['trade_dt'] = ctx.cal2td(df['trade_dt'])
    name = df.columns.drop(['stockcode', 'trade_dt'])[0]
    ndns_df = df.pivot_table(index='trade_dt', columns='stockcode', values=name, aggfunc=aggfunc) \
        .reindex(index=ctx.trade_dts, columns=ctx.instruments)
    return ndns_df

def is_tds(ctx, dts: np.ndarray, exclude=(0, 99999999)):
    '''是否日期序列均为交易日'''
    exclude_mask = np.isin(dts, exclude)
    return np.isin(dts[~exclude_mask], ctx.full_tds).all()

def isin_instrument(ctx, stklist):
    ''' 是否在股票池中 '''
    return np.isin(stklist, ctx.instruments)

def get_univ(ctx, df, right_close=True):
    ''' df中包含stockcode, indate, outdate; 日期nan的位置填充9999999
    right_close: outdate是否是右闭的
    '''
    assert (np.isin(['stockcode', 'indate', 'outdate'], df.columns).all())
    if not right_close:
        df['outdate'] = shift_tds(df['outdate'], -1)
        assert (is_tds(df['outdate']))
    else:
        assert (is_tds(df['outdate']))
    df.indate = df.indate.clip(lower=ctx.trade_dts[0])
    df.outdate = df.outdate.clip(upper=ctx.trade_dts[-1])
    df = df[df['outdate'] >= df['indate']]
    df = df[isin_instrument(df['stockcode'])]  # 退市后依旧在

    indate_df = df[['stockcode', 'indate']].rename(columns={'indate': 'trade_dt'})
    indate_df['value'] = 1.0
    indate_mat = pivot_dup(indate_df)
    outdate_df = df[['stockcode', 'outdate']].rename(columns={'outdate': 'trade_dt'})
    outdate_df['value'] = 1.0
    outdate_mat = pivot_dup(outdate_df)
    indate_mat.fillna(0, inplace=True)
    outdate_mat.fillna(0, inplace=True)

    result = indate_mat.cumsum() - outdate_mat.cumsum()
    result += outdate_mat

    result.replace(0, np.nan, inplace=True)
    result[result > 0] = 1.0
    return result

@lru_cache
def listed_univ(ctx):
    '''
    上市股票集合
    :return: ndns_df listed_univ
    '''
    df = ctx.wind_des.copy()
    df.rename(columns={'list_dt': 'indate', 'delist_dt': 'outdate'}, inplace=True)
    ndns_df = ctx.get_univ(df[['stockcode', 'indate', 'outdate']], right_close=False)
    df = pd.DataFrame({'univ_ipo': ndns_df.unstack(level=1)})
    return df

######################### 股票初始化特征 #########################

######################### 基础股票特征 #########################
@lru_cache
def univ_st(ctx):
    '''ST股票univ'''
    df = ctx.ds('dsets.asharest').get_df()
    df.reset_index(inplace=True)
    df.rename(columns={'entry_dt': 'indate', 'remove_dt': 'outdate'}, inplace=True)
    ndns_df = get_univ(df[['stockcode', 'indate', 'outdate']], right_close=False)
    ndns_df = ndns_df * listed_univ
    return df


