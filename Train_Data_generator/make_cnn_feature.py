"""
조회수 예측 AI 모델 학습 데이터를 생성하는 소스입니다.
CNN 네트워크를 통해 썸네일 피쳐를 추출해 저장합니다.
"""

import io
import pickle

import PIL
import numpy as np
import pandas as pd
import psycopg2 as pg2
import requests
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image


def load_image_from_url(url):
    """Returns an image with shape [1, height, width, num_channels]."""
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    image = preprocess_image(image)
    return image


# 구글 Big Transfer m-r101x1 모델 로드 (pre trained)
module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x1/1")

storage = {}

# 기존 추출 데이터가 존재할 경우
# with open('cnn_feature.pickle', 'rb') as f:
#     storage = pickle.load(f)

with open("../2_14_videos.pickle", "rb") as f:
    a_b = pickle.load(f)
# while True:
conn = pg2.connect(
    database=None,
    user=None,
    password=None,
    host=None,
    port=None,
)
cur = conn.cursor()

# DB에서 1000개 row 가져오기
df = pd.read_sql(
    f"""
    SELECT idx, video_id, thumbnail_url FROM video WHERE idx IN ({",".join(map(str, a_b))})
      """,
    con=conn,
)

for i, r in tqdm(df.iterrows()):
    try:
        image = load_image_from_url(r["thumbnail_url"])
        features = module(image)
        # dict 타입으로 저장
        storage[r["video_id"]] = features.numpy()
        # DB에 cnn 피쳐 추출로 컬럼 업데이트
    except Exception as e:
        print(e)
        pass

conn.commit()
conn.close()

# 1000개 마다 pickle로 저장
with open("../cnn_feature_a_b.pickle", "wb") as f:
    pickle.dump(storage, f)
