import pandas as pd

# 读取ratings.dat
ratings = pd.read_csv('../../datasets/movielens/ml-1m/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# 读取users.dat
users = pd.read_csv('../../datasets/movielens/ml-1m/users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

# 读取movies.dat
movies = pd.read_csv('../../datasets/movielens/ml-1m/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

# 合并ratings和users
ratings_users = pd.merge(ratings, users, on='UserID')

# 最后合并movies
merged_data = pd.merge(ratings_users, movies, on='MovieID')

# 显示合并后的数据
print(merged_data.head())

# 使用双冒号(::)作为分隔符保存到CSV文件
merged_data.to_csv('../../datasets/movielens/ml-1m/merged_movielens_data.dat', sep='\t', index=False)

