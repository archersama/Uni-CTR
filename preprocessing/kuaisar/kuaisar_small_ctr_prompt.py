# import pandas as pd
# from tqdm import tqdm
#
# id = [i for i in range(6979158 + 2763248)]
# a = pd.DataFrame({'id': id, 'prompt': '', 'label': ''})
#
# src = pd.read_csv('kuaisar_small_src_user_history_full.csv')
# rec = pd.read_csv('kuaisar_small_rec_user_history_full.csv')
#
#
# for i in tqdm(range(6979158)):
#
#     prompt = ('Ask: On a short-video platform, suppose there is a search scenario and a recommendation scenario. Predict whether user_'
#               + str(rec.iloc[i, 0]) + 'will click the ' + rec.iloc[i, 1] + ' in '
#               + rec.iloc[i, 2] + ' category in the recommendation scenario. Answer:')
#     a.iloc[i, 1] = prompt
#     a.iloc[i, 2] = rec.iloc[i, 3]
#
#
# for i in tqdm(range(2763248)):
#     prompt = ('Ask: On a short-video platform, suppose there is a search scenario and a recommendation scenario. Predict whether user_'
#               + str(src.iloc[i, 0]) + 'will click the ' + src.iloc[i, 1] + ' in '
#               + src.iloc[i, 2] + ' category in the search scenario. Answer:')
#     a.iloc[i+6979158, 1] = prompt
#     a.iloc[i+6979158, 2] = src.iloc[i, 2]
#
# a.to_csv('kuaisar_small_ctr_prompt.csv', index=False)


import pandas as pd
from tqdm import tqdm

# 创建 id 列
id = list(range(6979158 + 2763248))
a = pd.DataFrame({'id': id, 'prompt': '', 'label': ''})

# 读取 CSV 文件
src = pd.read_csv('kuaisar_small_src_user_history_full.csv')
rec = pd.read_csv('kuaisar_small_rec_user_history_full.csv')

# 为 rec DataFrame 创建提示和标签
rec_prompts = [
    'Ask: On a short-video platform, suppose there is a search scenario and a recommendation scenario. '
    'Predict whether user_' + str(user_id) + ' will click the ' + video + ' in ' + category +
    ' category in the recommendation scenario. Answer:'
    for user_id, video, category, _ in tqdm(rec.itertuples(index=False), total=len(rec))
]
rec_labels = rec.iloc[:, 3].tolist()

# 为 src DataFrame 创建提示和标签
src_prompts = [
    'Ask: On a short-video platform, suppose there is a search scenario and a recommendation scenario. '
    'Predict whether user_' + str(user_id) + ' will click the ' + video + ' in ' + category +
    ' category in the search scenario. Answer:'
    for user_id, video, category, _ in tqdm(src.itertuples(index=False), total=len(src))
]
src_labels = src.iloc[:, 3].tolist()

# 更新 DataFrame
a.loc[:6979157, 'prompt'] = rec_prompts
a.loc[:6979157, 'label'] = rec_labels
a.loc[6979158:, 'prompt'] = src_prompts
a.loc[6979158:, 'label'] = src_labels

# 保存到 CSV
a.to_csv('kuaisar_small_ctr_prompt.csv', index=False)
