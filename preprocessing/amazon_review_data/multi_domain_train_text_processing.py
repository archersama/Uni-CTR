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

    # 处理数据
    data = pd.read_csv(data_path)
    data['label'] = data['rating']
    print("Num of Original Samples:", len(data))

    # 创建一个空的 DataFrame
    dataset = pd.DataFrame(columns=['content', 'label', 'scenario'])

    for i in range(len(data)):
        print(i)
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
        else:
            scenario_info = "Unknown Scenario: "

        user_info = "The user ID is user_" + str(data['user_id'][i])
        user_info += ', who clicked ' + click_process(data['user_hist'][i]) + " recently. "

        item_info = "The ID of current product is product_" + str(data['asin'][i]) + ", "
        item_info += "the title is \'" + str(data['title'][i]) + "\', "
        item_info += "the brand is \'" + str(data['brand'][i]) + "\', "
        item_info += "the price is \'" + str(data['new_price'][i]) + "$. "

        text = scenario_info + user_info + item_info
        text = text.replace("|", "")
        label = data['label'][i]

        if len(text) < 10000:
            new_row = pd.DataFrame({'content': [text], 'label': [label], 'scenario': data['scenario'][i]})
            dataset = pd.concat([dataset, new_row], ignore_index=True)

    # 保存 DataFrame 到 CSV 文件
    dataset.to_csv(data_source, sep='|', index=False)
    print("Done")


if __name__ == '__main__':
    # csv.field_size_limit(sys.maxsize)
    data_process_amazon(
        '../../datasets/amazon_review_data/filtered_data/filtered_5_title.csv',
        '../../datasets/amazon_review_data/hybrid_data/hybrid_5_title.csv'
    )

