import numpy as np
import pandas as pd
import os

home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "train.csv")
train = pd.read_csv(train_path)
test_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "test.csv")
test = pd.read_csv(test_path)

# 欠損値の取得
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns

print(kesson_table(train))
print(kesson_table(test))

# 欠損値の修復
train['positiveness'] = train['positiveness'].fillna(train['positiveness'].median())
train['danceability'] = train['danceability'].fillna(train['danceability'].median())
train['liveness'] = train['liveness'].fillna(train['liveness'].median())
train['speechiness'] = train['speechiness'].fillna(train['speechiness'].median())
train['instrumentalness'] = train['instrumentalness'].fillna(train['instrumentalness'].median())

test['acousticness'] = test['acousticness'].fillna(test['acousticness'].median())
test['positiveness'] = test['positiveness'].fillna(test['positiveness'].median())
test['danceability'] = test['danceability'].fillna(test['danceability'].median())
test['energy'] = test['energy'].fillna(test['energy'].median())
test['liveness'] = test['liveness'].fillna(test['liveness'].median())
test['speechiness'] = test['speechiness'].fillna(test['speechiness'].median())
test['instrumentalness'] = test['instrumentalness'].fillna(test['instrumentalness'].median())

print(kesson_table(train))
print(kesson_table(test))

# 平均値を計算する関数
def convert_tempo_to_avg(tempo_range):
    low, high = map(int, tempo_range.split('-'))
    return (low + high) / 2  # 平均を返す

train['mean_tempo'] = train['tempo'].apply(convert_tempo_to_avg)
print(train)

# ダミー変数を作成
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=False)  # drop_first=Falseですべてのカテゴリーを保持
    dummies = dummies.applymap(lambda x: 1 if x > 0 else 0)  # 値を1か0に変換
    df = pd.concat([df, dummies], axis=1)
    return df

# train と test にダミー変数を追加
train = create_dummies(train, 'region')
test = create_dummies(test, 'region')

print(train)

# 各 region の出現回数をカウント
region_counts = train['region'].value_counts()
print(region_counts)

# region
unique_categories = pd.concat([train["region"], test["region"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["region"] = train["region"].map(category_map)
test["region"] = test["region"].map(category_map)

print(train['region'])

# 各 genre の出現回数をカウント
genre_counts = train['region'].value_counts()
print(genre_counts)
