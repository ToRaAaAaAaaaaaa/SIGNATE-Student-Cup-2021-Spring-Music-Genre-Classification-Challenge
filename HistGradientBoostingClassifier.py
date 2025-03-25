import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 訓練データとテストデータの作成
home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "train.csv")
train = pd.read_csv(train_path)
test_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "test.csv")
test = pd.read_csv(test_path)

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

test['acousticness'] = test['acousticness'].fillna(test['acousticness'].median())
test['positiveness'] = test['positiveness'].fillna(test['positiveness'].median())
test['danceability'] = test['danceability'].fillna(test['danceability'].median())
test['energy'] = test['energy'].fillna(test['energy'].median())
test['liveness'] = test['liveness'].fillna(test['liveness'].median())
test['speechiness'] = test['speechiness'].fillna(test['speechiness'].median())
test['instrumentalness'] = test['instrumentalness'].fillna(test['instrumentalness'].median())

# 文字データの変換
train['tempo'] = train['tempo'].apply(convert_tempo_to_avg)
test['tempo'] = test['tempo'].apply(convert_tempo_to_avg)

unique_categories = pd.concat([train["region"], test["region"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["region"] = train["region"].map(category_map)
test["region"] = test["region"].map(category_map)


# 説明変数と予測変数の決定
features = train[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]
target = train["genre"]
test_features = test[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]

# 変数の標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
test_features_scaled = scaler.transform(test_features)

# モデルの学習
hgb = HistGradientBoostingClassifier(
    loss='multiclass', 
    max_iter=1000, 
    max_depth=None, 
    learning_rate=0.01, 
    random_state=42
    )
hgb.fit(features_scaled, target)
my_prediction = hgb.predict(test_features_scaled)

# csvの作成
index = np.array(test["index"]).astype(int)
data = list(zip(index, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)