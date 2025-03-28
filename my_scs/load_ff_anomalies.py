import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_ff_anomalies(datapath, daily, t0=0, tN=float('inf')):
    """
    加载 Fama-French 因子数据

    参数:
    datapath : str, 数据文件路径
    daily : bool, 是否使用日频数据
    t0 : float, 起始时间
    tN : float, 结束时间

    返回:
    dates : numpy array, 日期序列
    ret : numpy array, 因子收益率
    mkt : numpy array, 市场收益率
    DATA : pandas DataFrame, 完整数据
    """

    if daily:
        ffact5 = 'F-F_Research_Data_5_Factors_2x3_daily.csv'
        fmom = 'F-F_Momentum_Factor_daily.csv'
    else:
        ffact5 = 'F-F_Research_Data_5_Factors_2x3.csv'
        fmom = 'F-F_Momentum_Factor.csv'

    # 读取五因子数据
    DATA = pd.read_csv(os.path.join(datapath, ffact5))
    # 转换日期格式
    DATA['date'] = pd.to_datetime(DATA['Date']).map(lambda x: x.timestamp())

    # 筛选时间范围
    DATA = DATA[(DATA['date'] >= t0) & (DATA['date'] <= tN)]

    # 读取动量因子数据
    MOM = pd.read_csv(os.path.join(datapath, fmom))
    MOM['date'] = pd.to_datetime(MOM['Date']).map(lambda x: x.timestamp())

    # 合并数据
    DATA = pd.merge(DATA, MOM, on='date', how='inner')

    # 提取所需数据
    dates = DATA['date'].values
    ret = DATA[['SMB', 'HML', 'Mom   ', 'RMW', 'CMA']].values / 100
    mkt = DATA['Mkt-RF'].values / 100

    return dates, ret, mkt, DATA