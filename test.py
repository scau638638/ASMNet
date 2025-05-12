import os
from os.path import join

import pandas as pd



if __name__ == '__main__':
    # 读取CSV文件
    performance_csv_path = join(os.path.abspath('.'), 'exps/BONet-CFD/csv/test.csv')
    df = pd.read_csv(performance_csv_path)
    metric = df['flops'].values[0]
    print(metric)