"""
제작한 NLP, CNN 데이터셋을 결합해
조회수 예측 모델을 학습시키는 소스입니다.
"""

import pickle

import keras
import numpy as np
import pandas as pd
import psycopg2 as pg2
from keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 기추출한 cnn/nlp 피쳐 불러오기
with open("cnn_feature_a_b.pickle", "rb") as f:
    cnn_feature = pickle.load(f)
with open("nlp_feature_a_b.pickle", "rb") as f:
    nlp_feature = pickle.load(f)
with open("2_14_videos.pickle", "rb") as f:
    a_b = pickle.load(f)


# 입력한 데이터프레임의 각 row의 영상의 cnn, nlp를 concat하여 하나의 벡터로 array화
def make_ins_outs(df):
    ins = []
    outs = []

    for i, r in tqdm(df.iterrows()):
        video_id = r["video_id"]
        try:
            ins.append(
                np.concatenate(
                    (
                        cnn_feature[video_id],
                        nlp_feature[video_id],
                        r["subscriber_num_log"],
                        r["date_delta"],
                    ),
                    axis=None,
                )
            )
            outs.append(r["views_log"])
        except:
            continue

    return np.array(ins), np.array(outs)


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(4096, activation="relu", input_shape=[2818]),
            layers.Dense(2048, activation="relu"),
            layers.Dense(1520, activation="relu"),
            layers.Dense(1024, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=["mse", "mae"])
    return model


# 구독자수가 공개된 채널에 대한 영상만 수집
conn = pg2.connect(
    database=None,
    user=None,
    password=None,
    host=None,
    port=None,
)
df = pd.read_sql(
    f"""
SELECT v.idx, v.video_id, v.upload_time, vv.check_time, vv.views, c.subscriber_num FROM video v 
JOIN video_views vv on v.idx = vv.video_idx JOIN channel c on v.channel_idx = c.idx
WHERE c.subscriber_num != 0 AND v.idx IN ({",".join(map(str, a_b))})""",
    con=conn,
)
# df = pd.read_csv(r'D:\all_ai.csv', error_bad_lines=False)
df = df.drop(df.loc[df["views"] <= 1000].index)  # 조회수가 1000회 미만인 영상 삭제

# df.sort_values('views').tail(8)
df = df.drop(df.loc[df["video_id"].isin(["MZ4JGye4dQU", "HPQ5mqovXHo", "wTowEKjDGkU"])].index)
# df = df.drop(df.sort_values('views').tail(11).index)  # 조회수가 특이하게 높은 이상치 데이터 삭제

# 영상이 업로드 된 시점까지를 0~1의 값으로 normalize
df["upload_time"] = pd.to_datetime(df["upload_time"])
df["date_delta"] = (df["check_time"] - df["upload_time"]) / np.timedelta64(1, "D")
date_delta_min = df["date_delta"].min()
df["date_delta"] = df["date_delta"] - date_delta_min
date_delta_max = df["date_delta"].max()
df["date_delta"] = df["date_delta"] / date_delta_max

# 영상의 조회수에 log 함수 적용, 0~1로 normalize
df["views_log"] = np.log(df["views"])

df["views_log"] = df["views_log"] - df["views_log"].min()
df["views_log"] = df["views_log"] / df["views_log"].max()

# log를 씌우지 않은 조회수 normalize
# df['views'] = df['views'] - df['views'].min()
# df['views'] = df['views'] / df['views'].max()

# 영상이 속한 채널의 구독자수에 log 적용, normalize
df["subscriber_num_log"] = np.log(df["subscriber_num"])
df["subscriber_num_log"] = df["subscriber_num_log"] - df["subscriber_num_log"].min()
df["subscriber_num_log"] = df["subscriber_num_log"] / df["subscriber_num_log"].max()

# log를 씌우지 않은 구독자수 normalize
df["subscriber_num"] = df["subscriber_num"] - df["subscriber_num"].min()
df["subscriber_num"] = df["subscriber_num"] / df["subscriber_num"].max()

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

train_all_ins, train_all_outs = make_ins_outs(df)
train_ins, train_outs = make_ins_outs(df_train)
test_ins, test_outs = make_ins_outs(df_test)

# 제작한 학습용 데이터 저장
with open("data_prepare.pickle_2_14", "wb") as f:
    pickle.dump(
        [train_all_ins, train_all_outs, train_ins, train_outs, test_ins, test_outs], f, protocol=4
    )

model = build_model()
model.summary()

his = model.fit(
    x=train_ins,
    y=train_outs,
    validation_data=(test_ins, test_outs),
    epochs=200,
    batch_size=256,
    verbose=1,
)

model.save("third_train_model")

with open("third_train_his.pickle", "wb") as f:
    pickle.dump(model.history.history, f)
#
# with open('second_train_his.pickle', 'wb') as f:
#     pickle.dump(model.history.history, f)

his = model.history.history
with open("second_train_his.pickle", "rb") as f:
    his = pickle.load(f)

import matplotlib.pyplot as plt

# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
plt.plot(his["mae"])
plt.plot(his["val_mae"])
plt.title("Second Model MAE")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

# 학습 손실 값과 검증 손실 값을 플롯팅 합니다.
plt.plot(his["loss"])
plt.plot(his["val_loss"])
plt.title("Second Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()
