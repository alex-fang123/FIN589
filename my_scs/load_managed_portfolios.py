import pandas as pd
import numpy as np
from datetime import datetime


def datenum2(dates, date_format):
    """
    将日期字符串转换为时间戳
    """
    if date_format == 'mm/dd/yyyy':
        return pd.to_datetime(dates, format='%m/%d/%Y').map(lambda x: x.timestamp())
    else:  # mm/yyyy
        return pd.to_datetime(dates, format='%m/%Y').map(lambda x: x.timestamp())


def load_managed_portfolios(filename, daily, drop_perc=1, omit_prefixes=None, keeponly=None):
    """
    加载管理组合数据

    参数:
    filename : str, 数据文件路径
    daily : bool, 是否为日频数据
    drop_perc : float, 缺失值比例阈值，默认为1
    omit_prefixes : list, 需要删除的特征前缀列表
    keeponly : list, 仅保留的特征列表

    返回:
    dates : numpy array, 日期序列
    re : numpy array, 超额收益率
    mkt : numpy array, 市场收益率
    names : list, 特征名称
    DATA : pandas DataFrame, 完整数据
    """

    if omit_prefixes is None:
        omit_prefixes = []
    if keeponly is None:
        keeponly = []

    # 设置日期格式
    date_format = 'mm/dd/yyyy' if daily else 'mm/yyyy'

    # 读取数据
    DATA = pd.read_csv(filename)
    DATA['date'] = datenum2(DATA['date'], date_format)

    if keeponly:
        # 如果指定了keeponly，只保留这些列
        keep_cols = ['date', 'rme'] + keeponly
        DATA = DATA[keep_cols]
    else:
        # 删除指定前缀的列
        for prefix in omit_prefixes:
            cols_to_drop = [col for col in DATA.columns if col.startswith(prefix)]
            DATA = DATA.drop(columns=cols_to_drop)

    # 删除缺失值比例超过阈值的特征
    missing_ratio = DATA.isna().sum() / len(DATA)
    cols_to_keep = missing_ratio[missing_ratio <= drop_perc].index
    DATA = DATA[cols_to_keep]

    # 删除包含缺失值的观测
    idx2keep = DATA.iloc[:, 2:].isna().sum(axis=1) == 0
    assert sum(idx2keep) > 0.75 * len(DATA), 'More than 25% of obs. need to be dropped!'
    DATA = DATA[idx2keep]

    # 提取所需数据
    dates = DATA['date'].values
    mkt = DATA['rme'].values
    re = DATA.iloc[:, 3:].values
    names = DATA.columns[3:].tolist()

    return dates, re, mkt, names, DATA