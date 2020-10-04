import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from tqdm import tqdm

# roc curve,

def build_model():
    model = keras.Sequential([
        layers.Dense(2048, activation='relu', input_shape=[2818]),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    opt = keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=opt, loss='mse')
    return model


with open('7000_thumbnail_feature.pickle', 'rb') as f:
    cnn_feature = pickle.load(f)
with open('7000_nlp_feature.pickle', 'rb') as f:
    nlp_feature = pickle.load(f)

df = pd.read_csv(r'D:\movie_channel_7000_with_subscriber.csv', error_bad_lines=False)
df.drop(columns=['idx', 'all_keyword', 'popularity', 'logo', 'check_time', 'keywords', 'video_description', 'names'],
        inplace=True)
df_target = df.loc[df['subscriber_num'] != 0]
df_target = df_target.dropna()
df_target.info()

df_target['upload_time'] = pd.to_datetime(df_target['upload_time'])
df_target['date_delta'] = (df_target['upload_time'] - df_target['upload_time'].min()) / np.timedelta64(1, 'D')
df_target['date_delta'] = df_target['date_delta'] / df_target['date_delta'].max()

df_target['views_log'] = np.log(df_target['views'])
df_target['views_log'] = df_target['views_log'] - df_target['views_log'].min()
df_target['views_log'] = df_target['views_log'] / df_target['views_log'].max()

df_target['views'] = df_target['views'] - df_target['views'].min()
df_target['views'] = df_target['views'] / df_target['views'].max()

df_target['subscriber_num_log'] = np.log(df_target['subscriber_num'])
df_target['subscriber_num_log'] = df_target['subscriber_num_log'] - df_target['subscriber_num_log'].min()
df_target['subscriber_num_log'] = df_target['subscriber_num_log'] / df_target['subscriber_num_log'].max()

df_target['subscriber_num'] = df_target['subscriber_num'] - df_target['subscriber_num'].min()
df_target['subscriber_num'] = df_target['subscriber_num'] / df_target['subscriber_num'].max()

model = build_model()
model.summary()

df_train, df_test = train_test_split(df_target, test_size=0.2, shuffle=True, random_state=100)

ins = []
outs = []
for i, r in tqdm(df_target.iterrows()):
    video_id = r['video_id']
    ins.append(np.concatenate((cnn_feature[video_id], nlp_feature[video_id],
                               r['subscriber_num_log'], r['date_delta']), axis=None))
    outs.append(r['views_log'])
train_all_ins = np.array(ins)
train_all_outs = np.array(outs)

ins = []
outs = []
for i, r in tqdm(df_train.iterrows()):
    video_id = r['video_id']
    ins.append(np.concatenate((cnn_feature[video_id], nlp_feature[video_id],
                               r['subscriber_num_log'], r['date_delta']), axis=None))
    outs.append(r['views_log'])
train_ins = np.array(ins)
train_outs = np.array(outs)

ins = []
outs = []
for i, r in tqdm(df_test.iterrows()):
    video_id = r['video_id']
    ins.append(np.concatenate((cnn_feature[video_id], nlp_feature[video_id],
                               r['subscriber_num_log'], r['date_delta']), axis=None))
    outs.append(r['views_log'])
test_ins = np.array(ins)
test_outs = np.array(outs)

test_outs.max()
plt.show()



model.fit(x=train_ins, y=train_outs,
          validation_data=(test_ins, test_outs),
          epochs=300, batch_size=128, verbose=1)

pred_outs = model.predict(test_ins)
r2_score(test_outs, pred_outs)
pred_outs = model.predict(train_ins)
r2_score(train_outs, pred_outs)
pred_outs = model.predict(train_all_ins)
r2_score(train_all_outs, pred_outs)

tt_1 = np.array([1, 2, 3])
tt_1 = np.array([1, 2, 3])
r2_score(tt_1, [1, 2, 3, ])

svr_rbf = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=.1)
svr_rbf.fit(train_ins, train_outs)
print(svr_rbf.score(train_ins, train_outs))
print(svr_rbf.score(test_ins, test_outs))

param = {'C': [100, 10, 1, 0.1, 0.01, 0.001],
         'epsilon': [100, 10, 1, 0.1, 0.01, 0.001],
         'gamma': [100, 10, 1, 0.1, 0.01, 0.001]}

modelsvr = SVR(kernel='rbf')

grids = GridSearchCV(modelsvr, param, cv=3, verbose=2, n_jobs=5, scoring='r2')
grids.fit(train_all_ins, train_all_outs)

svr = SVR(C=10, gamma=0.001, kernel='rbf', epsilon=0.001)
svr.fit(train_ins, train_outs)
print(svr.score(train_ins, train_outs))
print(svr.score(test_ins, test_outs))
print(svr.score(train_all_ins, train_all_outs))


GridSearchCV(cv=3, estimator=SVR(), n_jobs=5,
             param_grid={'C': [100, 10, 1, 0.1, 0.01, 0.001],
                         'epsilon': [100, 10, 1, 0.1, 0.01, 0.001],
                         'gamma': [100, 10, 1, 0.1, 0.01, 0.001]},
             scoring='r2', verbose=2)
grids.best_estimator_




### aiai


with open('7000_thumbnail_feature.pickle', 'rb') as f:
    cnn_feature = pickle.load(f)
with open('7000_nlp_feature.pickle', 'rb') as f:
    nlp_feature = pickle.load(f)