"""
用于生成主力合约专属的dataset
本文件以某一品种为例
"""
import pandas as pd



def read_h5(pth):
    df = pd.read_hdf(pth, 'data')
    return df


def select_by_mc(df):
    mc = read_h5(r'K:\cta\fut_repo\main_contracts.h5')
    result = pd.merge(df, mc, how='left', on=['symbol', 'trade_dt'])
    result = result[(result['active_flag_xw'] == 1)]
    return result.reset_index(drop=True)


def dz2symbol(df):
    """命名上有点混乱，tick中用symbol标识，所以1m也是"""
    instrument = read_h5(r'K:\cta\fut_repo\instrument.h5')
    instrument['futcode'] = instrument['futcode'].apply(lambda x: x[:-4])
    instrument = instrument[['symbol', 'futcode']]
    df.rename(columns={'symbol': 'futcode'}, inplace=True)
    df = pd.merge(df, instrument, how='left', on='futcode')
    return df


def gen_daily_datasets(name):
    eodprice = read_h5(r'K:\cta\fut_repo\eodprice.h5')
    result = select_by_mc(eodprice)
    result = result[result.symbol.map(lambda x: x[1:3] == name)]
    return result


def gen_bar_datasets(name):
    bar_base = pd.read_parquet(r'K:\cta\fut_repo\intraday\dz_lev1_1m_base')
    # 需要symbol映射(发现品种名没改，检验文件名日期已补全）
    bar_base = dz2symbol(bar_base)
    bar_base['trade_dt'] = bar_base['trade_dt'].astype(int)
    result = select_by_mc(bar_base)
    result = result[result.symbol.map(lambda x: x[1:3] == name)]
    return result


if __name__ == '__main__':
    prod_name = 'zn'
    gen_bar_datasets(prod_name)
    pass
