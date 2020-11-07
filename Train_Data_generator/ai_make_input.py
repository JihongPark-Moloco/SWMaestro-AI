"""
조회수 예측 모델 AI 학습을 위한 데이터셋을 생성하는 소스입니다.
CNN으로 썸네일을 NLP로 영상 제목을 벡터화해 pickle 파일로 저장합니다.
"""

import torch


def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    # 배치에서 각 line들의 mask를 valid length 만큼 1로 치환
    # ex) BERT의 입력은 64길이이고 실제 입력되는 문장이 30길이면 마스크는 '1'로 되어진 30개의 요소 + '0'으로 되어진 34개의 요소로 이루어진다.
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.long().cuda()


def gen_input_ids(tokenizer, sentence):
    target = []
    valid_length = []
    for s in sentence:
        s = "[CLS] " + s
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
        valid_length.append(len(input_ids))
        target_ids = [1] * 32
        target_ids[: len(input_ids)] = input_ids
        target.append(target_ids)
    return target, valid_length


from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()
input_ids, valid_length = gen_input_ids(
    tokenizer=tokenizer, sentence=["한국어 모델을 공유합니다.", "두번째 문장입니다."]
)

model = get_kobert_model()
model.eval()

input_ids = torch.LongTensor(input_ids)
attention_mask = gen_attention_mask(input_ids, valid_length)
attention_mask = torch.LongTensor(attention_mask)
token_type_ids = torch.zeros_like(input_ids)
sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)

pooled_output

import tensorflow as tf
import tensorflow_hub as hub
import requests
import io
import PIL
import numpy as np

module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x1/1")


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


# import pandas as pd
# from tqdm import tqdm
#
# df = pd.read_csv(r'D:\movie_channel_target_7000.csv')
# storage = {}
#
# for i, r in tqdm(df.iterrows()):
#     image = load_image_from_url(r['thumbnail_url'])
#     features = module(image)
#     storage[r['video_id']] = features.numpy()
#
#
# import pickle
#
# with open('7000.pickle', 'wb') as f:
#     pickle.dump(storage, f)

import psycopg2 as pg2
import pandas as pd
import pickle
from tqdm import tqdm

storage = {}

with open("../cnn_feature.pickle", "rb") as f:
    storage = pickle.load(f)

conn = pg2.connect(
    database="createtrend",
    user="muna",
    password="muna112358!",
    host="ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com",
    port="5432",
)
cur = conn.cursor()
for k in tqdm(storage.keys()):
    cur.execute(f"UPDATE video SET processed_cnn = true WHERE video_id = '{k}'")
conn.commit()
conn.close()

while True:
    conn = pg2.connect(
        database="createtrend",
        user="muna",
        password="muna112358!",
        host="ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com",
        port="5432",
    )
    cur = conn.cursor()
    df = pd.read_sql(
        """
    SELECT video_id, v.thumbnail_url
    FROM video v
             JOIN channel c on c.idx = v.channel_idx
    WHERE v.processed_cnn = false
      AND v.status = True
      AND v.forbidden = false AND c.subscriber_num is not null AND c.subscriber_num != 0 LIMIT 1000;
      """,
        con=conn,
    )

    for i, r in tqdm(df.iterrows()):
        try:
            image = load_image_from_url(r["thumbnail_url"])
            features = module(image)
            storage[r["video_id"]] = features.numpy()
            cur.execute(f"UPDATE video SET processed_cnn = true WHERE video_id = '{r['video_id']}'")
        except Exception as e:
            print(e)
            pass

    conn.commit()
    conn.close()

    with open("../cnn_feature.pickle", "wb") as f:
        pickle.dump(storage, f)

import pandas as pd
from tqdm import tqdm

df = pd.read_csv(r"D:\movie_channel_target_7000.csv")
storage = {}
tokenizer = get_tokenizer()
model = get_kobert_model()
model = model.cuda()
model.eval()

for i, r in tqdm(df.iterrows()):
    input_ids, valid_length = gen_input_ids(tokenizer=tokenizer, sentence=[r["video_name"]])
    input_ids = torch.LongTensor(input_ids).cuda()
    attention_mask = gen_attention_mask(input_ids, valid_length)
    token_type_ids = torch.zeros_like(input_ids).cuda()
    sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
    storage[r["video_id"]] = pooled_output.cpu().detach().numpy()
    del input_ids, attention_mask, token_type_ids

with open("../7000_nlp_feature.pickle", "wb") as f:
    pickle.dump(storage, f)

### NLP 준비

import psycopg2 as pg2
import pandas as pd
import pickle
from tqdm import tqdm

storage = {}

with open("../nlp_feature.pickle", "rb") as f:
    storage = pickle.load(f)

while True:
    conn = pg2.connect(
        database="createtrend",
        user="muna",
        password="muna112358!",
        host="ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com",
        port="5432",
    )
    cur = conn.cursor()
    df = pd.read_sql(
        """
    SELECT video_id, v.video_name
    FROM video v
             JOIN channel c on c.idx = v.channel_idx
    WHERE v.processed_nlp = false
      AND v.status = True
      AND v.forbidden = false AND c.subscriber_num is not null AND c.subscriber_num != 0 LIMIT 1000;
      """,
        con=conn,
    )

    for i, r in tqdm(df.iterrows()):
        input_ids, valid_length = gen_input_ids(tokenizer=tokenizer, sentence=[r["video_name"]])
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = gen_attention_mask(input_ids, valid_length)
        token_type_ids = torch.zeros_like(input_ids).cuda()
        sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
        storage[r["video_id"]] = pooled_output.cpu().detach().numpy()
        del input_ids, attention_mask, token_type_ids
        cur.execute(f"UPDATE video SET processed_nlp = true WHERE video_id = '{r['video_id']}'")

    conn.commit()
    conn.close()

    with open("../nlp_feature.pickle", "wb") as f:
        pickle.dump(storage, f)
