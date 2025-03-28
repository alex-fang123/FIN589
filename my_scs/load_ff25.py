import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_ff25(datapath, daily, t0=0, tN=float('inf')):
    """
    加载 Fama-French 25 个投资组合数据

    参数:
    datapath : str, 数据文件路径
    daily : bool, 是否使用日频数据
    t0 : float, 起始时间
    tN : float, 结束时间

    返回:
    dates : numpy array, 日期序列
    ret : numpy array, 投资组合收益率
    mkt : numpy array, 市场收益率
    DATA : pandas DataFrame, 完整数据
    labels : list, 投资组合标签
    """

    if daily:
        ffact5 = 'F-F_Research_Data_Factors_daily.csv'
        ff25 = '25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv'
    else:
        ffact5 = 'F-F_Research_Data_Factors.csv'
        ff25 = '25_Portfolios_5x5_average_value_weighted_returns_monthly.csv'

    # 读取因子数据
    DATA = pd.read_csv(os.path.join(datapath, ffact5))
    # 转换日期格式
    DATA['Date'] = pd.to_datetime(DATA['Date']).map(lambda x: x.timestamp())

    # 筛选时间范围
    DATA = DATA[(DATA['Date'] >= t0) & (DATA['Date'] <= tN)]

    # 读取25个投资组合数据
    RET = pd.read_csv(os.path.join(datapath, ff25))
    RET['Date'] = pd.to_datetime(RET['Date']).map(lambda x: x.timestamp())

    # 合并数据
    DATA = pd.merge(DATA, RET, on='Date', how='inner')

    # 提取所需数据
    dates = DATA['Date'].values
    mkt = DATA['Mkt-RF'].values / 100

    # 计算超额收益率
    # 假设列6-30是投资组合收益率
    ret = DATA.iloc[:, 6:31].values / 100 - DATA['RF'].values[:, np.newaxis] / 100

    # 获取投资组合标签
    labels = RET.columns[1:].tolist()

    return dates, ret, mkt, DATA, labels