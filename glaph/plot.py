import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 訓練データとテストデータの作成
home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "train.csv")
train = pd.read_csv(train_path)

# 平均値を計算する関数
def convert_tempo_to_avg(tempo_range):
    low, high = map(int, tempo_range.split('-'))
    return (low + high) / 2  # 平均を返す

# データの前処理
train['positiveness'] = train['positiveness'].fillna(train['positiveness'].median())
train['danceability'] = train['danceability'].fillna(train['danceability'].median())
train['liveness'] = train['liveness'].fillna(train['liveness'].median())
train['speechiness'] = train['speechiness'].fillna(train['speechiness'].median())
train['instrumentalness'] = train['instrumentalness'].fillna(train['instrumentalness'].median())

# 文字データの変換
train['tempo'] = train['tempo'].apply(convert_tempo_to_avg)
unique_categories = pd.Series(train["region"].explode().unique())
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["region"] = train["region"].map(category_map)

# 説明変数と予測変数の決定
X = train[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]
target = train["genre"]

for column in X.columns:
    plt.scatter(X[column], target, color="blue", label=f"{column}")
    plt.xlabel(column)
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
