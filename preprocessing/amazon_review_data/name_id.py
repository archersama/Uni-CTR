import pandas as pd
import re
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import concurrent.futures
import swifter


def print_elapsed_time(start_time, segment_name):
    elapsed_time = time.time() - start_time
    print(f"Time taken for {segment_name}: {elapsed_time / 60:.2f} minutes")
    return time.time()  # 返回当前时间，以便于下次计时


def process_nan(data, nan_list, fill_text="unknown"):
    for col in nan_list:
        # Fill NaN values with 'unknown'
        data[col].fillna(fill_text, inplace=True)
        # Replace empty strings with 'unknown'
        data[col] = data[col].apply(lambda x: fill_text if x == "" else x)
    return data


def process_price(data):
    x = data['price']
    if pd.isna(x) or x == "":
        return None
    else:
        found_price = re.match(r'^\$\d+(\.\d+)?$', x)
        if found_price:
            price = float(found_price.group()[1:])  # 提取数字部分并转换为浮点数
            return price
        else:
            return None


def get_user_history_feature_optimized(data, time_window, mode="id"):
    # Sort the data by 'user_id' and 'unixReviewTime'
    data = data.sort_values(by=['user_id', 'unixReviewTime']).reset_index(drop=True)

    # Create an empty 'user_hist' column
    data['user_hist'] = ''

    # Create a dictionary to hold the user's history
    user_histories = {}

    # Create a dictionary to map asin to title for fast lookup
    if mode == "title":
        asin_to_title = {row['asin']: str(row['title']) if pd.notna(row['title']) else "unknown"
                         for _, row in data.iterrows()}

    # Using tqdm to show the progress bar
    for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing user history"):
        user_id = row['user_id']
        if user_id not in user_histories:
            user_histories[user_id] = []

        # Get the recent history based on the time_window
        recent_history = user_histories[user_id][-time_window:]

        # Convert the recent history to the desired format
        if mode == "id":
            history_str = ' & '.join(['product_' + str(asin) for asin in recent_history])
        elif mode == "title":
            # Use the asin_to_title mapping for fast lookup
            titles = [asin_to_title[asin] for asin in recent_history]
            history_str = ' & '.join(["product '" + title + "'" for title in titles])
        else:
            history_str = ''

        # Assign the history string to the 'user_hist' column
        data.at[idx, 'user_hist'] = history_str

        # If the rating is positive, add the product to the user's history AFTER updating the user_hist column
        if row['rating'] == 1:
            user_histories[user_id].append(row['asin'])

    return data


def click_process(user_hist_str):
    if pd.isna(user_hist_str):
        return "nothing"
    else:
        return str(user_hist_str)


def process_text(row):
    if row['scenario'] == 0:
        scenario_info = "Amazon Fashion: "
    elif row['scenario'] == 1:
        scenario_info = "Digital Music: "
    elif row['scenario'] == 2:
        scenario_info = "Musical Instruments: "
    elif row['scenario'] == 3:
        scenario_info = "Gift Cards: "
    elif row['scenario'] == 4:
        scenario_info = "All Beauty: "
    else:
        scenario_info = "Unknown Scenario: "

    user_info = "user_" + str(row['user_id']) + ', '
    user_info += click_process(row['user_hist']) + ", "

    item_info = 'The ID of current product is product_' + str(row['asin']) + ", "
    item_info += "the title is \'" + str(row['title']) + "\', "
    item_info += "the brand is \'" + str(row['brand']) + "\', "
    item_info += "the price is " + str(row['new_price']) + "$. "

    return scenario_info + user_info + item_info


def process_data(review_data_path: list, meta_data_path: list, save_path="", mode="id"):
    print(pd.__version__)
    start_time = time.time()

    review_data_list = []
    meta_data_list = []
    start_time = print_elapsed_time(start_time, "Import data")

    for i in range(len(review_data_path)):
        review_data = pd.read_json(review_data_path[i], lines=True)
        review_data['scenario'] = i
        review_data_list.append(review_data)
        meta_data = pd.read_json(meta_data_path[i], lines=True)
        meta_data_list.append(meta_data)
    start_time = print_elapsed_time(start_time, "Combine data")

    review_data = pd.concat(review_data_list)
    meta_data = pd.concat(meta_data_list)

    meta_data = process_nan(meta_data, ['brand', 'feature', 'description', 'similar_item', 'tech1', 'title'])
    meta_data = meta_data.drop_duplicates(subset=['asin'])

    join_data = pd.merge(review_data, meta_data, on='asin')
    start_time = print_elapsed_time(start_time, "Merge data")

    join_data['new_price'] = join_data.apply(process_price, axis=1)
    join_data['new_price'] = join_data['new_price'].fillna(join_data['new_price'].mean())
    start_time = print_elapsed_time(start_time, "Process price")

    sparse_features = ['reviewerID', 'asin']
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(join_data[feat])
        join_data[feat] = lbe.transform(join_data[feat])

    join_data['rating'] = join_data['overall'].apply(lambda x: 1 if x > 3 else 0)
    join_data = join_data.rename(columns={'reviewerID': 'user_id'})
    start_time = print_elapsed_time(start_time, "Process sparse features")

    join_data = get_user_history_feature_optimized(join_data, 5, mode=mode)
    start_time = print_elapsed_time(start_time, "Process user history")

    # 处理原始数据
    filter_data = join_data[["scenario", "user_id", "asin", "brand", "new_price",
                             "rating", "unixReviewTime", "title", "user_hist", "overall"]]
    filter_data = filter_data.replace(r'\s+', ' ', regex=True)

    # 生成文本数据
    filter_data = process_nan(filter_data, ['user_hist'], fill_text="nothing")
    filter_data['content'] = filter_data.swifter.apply(lambda row: process_text(row), axis=1)

    # 过滤过长的文本数据
    filter_data = filter_data[filter_data['content'].str.len() < 10000]

    # 保存文本数据
    print("Saving text data...")
    filter_data.to_csv(save_path, sep='\t', index=None, encoding='utf-8')

    print("Done!")


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
        save_path='../../datasets/amazon_review_data/hybrid_data/hybrid_5_id.csv',
        mode="id"
    )
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
        save_path='../../datasets/amazon_review_data/hybrid_data/hybrid_5_title.csv',
        mode="title"
    )
