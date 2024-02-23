import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def process_nan(data):
    nan_list = ['brand', 'feature', 'description', 'similar_item', 'tech1']
    for i in range(len(nan_list)):
        data[nan_list[i]] = data[nan_list[i]].fillna('unknown')
    return data


def price_process(data):
    x = data['price']
    if pd.isna(x) or x == "":
        pass
    elif '-' in x:
        x_list = re.findall(r"\d+\.?\d*", x)
        if x_list:
            s = (float(x_list[0]) + float(x_list[1])) / 2
            return s
        else:
            pass
    else:
        found_price = re.findall(r"\d+\.?\d*", x)
        if found_price:
            s = float(found_price[0])
            return s
        else:
            pass


def get_user_history_feature(data, time_window, mode="id"):
    data = data.sort_values(by=['user_id', 'unixReviewTime'], na_position='first').reset_index()
    data_group = data[data['rating'] == 1]
    data_group = data_group[['user_id', 'asin']].groupby('user_id').agg(list).reset_index()
    data['user_hist'] = ''
    i = 0
    print(len(data))
    while i <= len(data) - 1:
        # print("i:", i)
        user_id = data.iat[i, 4]
        user_history_index = data_group[data_group['user_id'] == user_id].index.tolist()
        if len(user_history_index) == 0:
            i += 1
            continue
        else:
            user_history_index = user_history_index[0]
        user_history = data_group['asin'][user_history_index]  # 得到用户正样本点击序列
        len_history = len(user_history)
        pos, j = 0, 0
        i_history = []
        user_group = data[data['user_id'] == user_id]
        while pos <= len_history and i <= len(data) - 1 and j < len(user_group):
            # print("i:", i)
            if data.iat[i - 1, -2] == 0 and j != 0:
                data.iat[i, -1] = data.iat[i - 1, -1]
            elif data.iat[i - 1, -2] == 1 and j != 0:
                pos += 1
                if pos < time_window:
                    i_history = user_history[0:pos]
                else:
                    i_history = user_history[pos - time_window:pos]

                if mode == "id":
                    data.iat[i, -1] = ' & '.join(['product_' + str(num) for num in i_history])
                elif mode == "title":
                    data.iat[i, -1] = ' & '.join(
                        ['product \'' + str(data[data['asin'] == num]['title'].values[0]) + '\'' for num in i_history]
                    )
            j += 1
            i += 1
    return data


def process_data(review_data_path: list, meta_data_path: list, save_path="", mode="id"):
    print(pd.__version__)

    review_data_list = []
    meta_data_list = []

    for i in range(len(review_data_path)):
        review_data = pd.read_json(review_data_path[i], lines=True)
        review_data['scenario'] = i
        review_data_list.append(review_data)
        meta_data = pd.read_json(meta_data_path[i], lines=True)
        meta_data_list.append(meta_data)

    review_data = pd.concat(review_data_list)
    meta_data = pd.concat(meta_data_list)

    meta_data['brand'] = meta_data['brand'].fillna('unknown')
    meta_data = process_nan(meta_data)
    meta_data = meta_data.drop_duplicates(subset=['asin'])

    join_data = pd.merge(review_data, meta_data, on='asin')

    print(join_data.columns.tolist())

    join_data['new_price'] = join_data.apply(price_process, axis=1)
    join_data['new_price'] = join_data['new_price'].fillna(join_data['new_price'].mean())
    sparse_features = ['reviewerID', 'asin']
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(join_data[feat])
        join_data[feat] = lbe.transform(join_data[feat])

    join_data['rating'] = join_data['overall'].apply(lambda x: 1 if x > 3 else 0)
    join_data = join_data.rename(columns={'reviewerID': 'user_id'})

    join_data = get_user_history_feature(join_data, 5, mode=mode)

    print(join_data.columns.tolist())

    filter_data = join_data[[
        "scenario", "user_id", "asin", "brand", "new_price", "rating", "unixReviewTime", "title", "user_hist", "overall"
    ]]
    filter_data = filter_data.replace(r'\s+', ' ', regex=True)

    filter_data.to_csv(save_path, index=None)


if __name__ == '__main__':
    '''
    Amazon Fashion
    Digital Music
    Musical Instruments
    Gift Cards
    All Beauty
    '''
    process_data(
        [
            '../../datasets/amazon_review_data/raw_data/review_data/AMAZON_FASHION.json',
            '../../datasets/amazon_review_data/raw_data/review_data/Digital_Music.json',
            '../../datasets/amazon_review_data/raw_data/review_data/Musical_Instruments.json',
            '../../datasets/amazon_review_data/raw_data/review_data/Gift_Cards.json',
            '../../datasets/amazon_review_data/raw_data/review_data/All_Beauty.json',
        ],
        [
            '../../datasets/amazon_review_data/raw_data/meta_data/meta_AMAZON_FASHION.json',
            '../../datasets/amazon_review_data/raw_data/meta_data/meta_Digital_Music.json',
            '../../datasets/amazon_review_data/raw_data/meta_data/meta_Musical_Instruments.json',
            '../../datasets/amazon_review_data/raw_data/meta_data/meta_Gift_Cards.json',
            '../../datasets/amazon_review_data/raw_data/meta_data/meta_All_Beauty.json',
        ],
        save_path='../../datasets/amazon_review_data/filtered_data/filtered_5_id.csv',
        mode="id"
    )
