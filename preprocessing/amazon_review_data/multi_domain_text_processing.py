import pandas as pd
import csv
import sys

'''
Amazon Fashion
Digital Music
Musical Instruments
Gift Cards
All Beauty
'''


def click_process(user_hist_str):
    if pd.isna(user_hist_str):
        return "nothing"
    else:
        return str(user_hist_str)


def data_process_amazon(data_path, data_source):
    data = pd.read_csv(data_path)
    text_file = open(data_source + '.txt', 'w+', encoding='utf-8')
    print(len(data))
    for i in range(len(data)):
        if data['scenario'][i] == 0:
            scenario_info = "Amazon Fashion: "
        elif data['scenario'][i] == 1:
            scenario_info = "Digital Music: "
        elif data['scenario'][i] == 2:
            scenario_info = "Musical Instruments: "
        elif data['scenario'][i] == 3:
            scenario_info = "Gift Cards: "
        elif data['scenario'][i] == 4:
            scenario_info = "All Beauty: "

        user_info = "The user ID is user_" + str(data['user_id'][i])
        user_info += ', who clicked ' + click_process(data['user_hist'][i]) + " recently. "

        item_info = 'The ID of current product is product_' + str(data['asin'][i]) + ", "
        item_info += "the title is \'" + str(data['title'][i]) + "\', "
        item_info += "the brand is \'" + str(data['brand'][i]) + "\', "
        item_info += "the price is \'" + str(data['new_price'][i]) + "$. "

        text_file.write(scenario_info + user_info + item_info + '\n')

    text_file.close()
    print("Done")


if __name__ == '__main__':
    # csv.field_size_limit(sys.maxsize)
    data_process_amazon(
        '../../datasets/amazon_review_data/filtered_data/filtered_5_id.csv',
        '../../datasets/amazon_review_data/text_data/text_5_id'
    )

