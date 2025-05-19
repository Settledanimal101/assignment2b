import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path, site_id=2000):
    """
    加载并清洗交通数据，生成监督学习格式的输入和输出。

    参数:
        path (str): CSV 数据文件路径。
        site_id (int): 要提取的 SiteID。

    返回:
        scaler (MinMaxScaler): 用于数据反归一化的缩放器。
        X (np.ndarray): 输入特征，形状为 (样本数, 24, 1)。
        y (np.ndarray): 目标值，形状为 (样本数, 1)。
    """
    df = pd.read_csv(path)

    if 'SiteID' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("CSV 文件缺少 'SiteID' 或 'Volume' 列")

    # 选出特定站点的 Volume 数据
    df_site = df[df['SiteID'] == site_id]['Volume'].values.reshape(-1, 1)

    if len(df_site) < 25:
        raise ValueError("数据太少，无法构造时间序列")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_site)

    X, y = [], []
    for i in range(24, len(scaled)):
        X.append(scaled[i-24:i])  # 前24小时作为输入
        y.append(scaled[i])       # 当前时间作为输出

    return scaler, np.array(X), np.array(y)
