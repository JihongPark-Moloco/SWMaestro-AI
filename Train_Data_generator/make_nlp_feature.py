"""
조회수 예측 AI 모델 학습 데이터를 생성하는 소스입니다.
NLP 네트워크를 통해 영상 제목 피쳐를 추출해 저장합니다.
"""

import pickle

import pandas as pd
import psycopg2 as pg2
import torch
from tqdm import tqdm

from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer


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


storage = {}

# 기존 추출 데이터가 존재할 경우
# with open('nlp_feature.pickle', 'rb') as f:
#     storage = pickle.load(f)

with open("../2_14_videos.pickle", "rb") as f:
    a_b = pickle.load(f)

tokenizer = get_tokenizer()
model = get_kobert_model()
model.cuda()
model.eval()

# while True:
conn = pg2.connect(
    database="createtrend",
    user="muna",
    password="muna112358!",
    host="ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com",
    port="5432",
)
cur = conn.cursor()

# DB에서 1000개 row 가져오기
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
    # 32개 길이의 토큰 생성
    input_ids, valid_length = gen_input_ids(tokenizer=tokenizer, sentence=[r["video_name"]])
    input_ids = torch.LongTensor(input_ids).cuda()

    # attention mask 생성, 토큰 길이만큼 1 입력 그 이외에는 0
    attention_mask = gen_attention_mask(input_ids, valid_length)

    # 문장의 유형을 구분 ex) context:0, question:1 , 여기는 제목뿐이므로 0
    token_type_ids = torch.zeros_like(input_ids).cuda()

    sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
    storage[r["video_id"]] = pooled_output.cpu().detach().numpy()
    del input_ids, attention_mask, token_type_ids

    # DB에 nlp 처리 컬럼 업데이트
    cur.execute(f"UPDATE video SET processed_nlp = true WHERE video_id = '{r['video_id']}'")

conn.commit()
conn.close()

# 1000개 단위로 저장
with open("../nlp_feature_a_b.pickle", "wb") as f:
    pickle.dump(storage, f)
