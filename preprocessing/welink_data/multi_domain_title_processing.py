import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 定义处理函数
def get_titles(user_hist, doc_id_title_map):
    if user_hist is None or isinstance(user_hist, str):
        return None
    else:
        ids = [int(s.split('_')[-1]) for s in user_hist.split('|') if s]
        return '|'.join([doc_id_title_map.get(i, '') for i in ids])


def process_data(data_path: list, save_path="", mode="id"):
    print(pd.__version__)

    df = pd.read_csv(data_path)

    # 创建doc_id到text_title的映射
    doc_id_title_map = dict(zip(df['doc_id'], df['text_title']))

    # 应用处理函数
    df['user_hist_text'] = df['user_hist'].apply(lambda x: get_titles(x, doc_id_title_map))

    df.to_csv(save_path, index=None, encoding='utf-8', sep=',')

if __name__ == '__main__':
    '''scope
    0
    1
    2
    '''
    process_data(
        ['welink_with_time_title.csv'],
        'fzkuji_welink_with_time_title_text.csv',
    )
