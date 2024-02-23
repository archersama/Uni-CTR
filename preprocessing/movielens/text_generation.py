import pandas as pd
import re
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import concurrent.futures
import swifter


def get_user_history_feature_optimized(data, time_window, mode="id"):
    # Sort the data by 'user_id' and 'unixReviewTime'
    data = data.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)

    # Create an empty 'user_hist' column
    data['user_hist'] = ''

    # Create a dictionary to hold the user's history
    user_histories = {}

    # Create a dictionary to map asin to title for fast lookup
    if mode == "title":
        asin_to_title = {row['MovieID']: str(row['Title']) if pd.notna(row['Title']) else "unknown"
                         for _, row in data.iterrows()}

    # Using tqdm to show the progress bar
    for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing user history"):
        user_id = row['UserID']
        if user_id not in user_histories:
            user_histories[user_id] = []

        # Get the recent history based on the time_window
        recent_history = user_histories[user_id][-time_window:]

        # Convert the recent history to the desired format
        if mode == "id":
            history_str = ' & '.join(['movie_' + str(asin) for asin in recent_history])
        elif mode == "title":
            # Use the asin_to_title mapping for fast lookup
            titles = [asin_to_title[asin] for asin in recent_history]
            history_str = ' & '.join(["movie '" + title + "'" for title in titles])
        else:
            history_str = ''

        # Assign the history string to the 'user_hist' column
        data.at[idx, 'user_hist'] = history_str

        # If the rating is positive, add the product to the user's history AFTER updating the user_hist column
        if row['CTR'] == 1:
            user_histories[user_id].append(row['MovieID'])

    return data


def click_process(user_hist_str):
    if pd.isna(user_hist_str):
        return "nothing"
    else:
        return str(user_hist_str)


def process_text(row):
    scenario_info = row['Genres'] + ": "

    user_info = "The user ID is user_" + str(row['UserID'])
    user_info += ', who watched ' + click_process(row['user_hist']) + " recently"
    user_info += ', whose gender is ' + str(row['Gender'])
    user_info += ', whose age is ' + str(row['Age'])
    user_info += ', whose occupation is ' + str(row['Occupation'])
    user_info += ', whose zip-code is ' + str(row['Zip-code']) + ". "

    item_info = 'The ID of current movie is movie_' + str(row['MovieID']) + ", "
    item_info += "the title is \'" + str(row['Title']) + "\'. "

    return scenario_info + user_info + item_info



# merged_movielens_data.dat
join_data = pd.read_csv('../../datasets/movielens/ml-1m/merged_movielens_text_prep.csv', sep='\t', engine='python')
# UserID	MovieID	Rating	Timestamp	Gender	Age	Occupation	Zip-code	Title	Genres

join_data['CTR'] = join_data['Rating'].apply(lambda x: 1 if x > 3 else 0)

join_data = get_user_history_feature_optimized(join_data, 5, mode="id")

join_data['content'] = join_data.swifter.apply(lambda row: process_text(row), axis=1)

join_data.to_csv('../../datasets/movielens/ml-1m/merged_movielens_text_final.csv', sep='\t', index=False)
