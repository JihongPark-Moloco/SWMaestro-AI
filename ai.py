import io

import PIL
import keras
import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
import torch

from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer

cnn_model = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x1/1")

tokenizer = get_tokenizer()
nlp_model = get_kobert_model()
nlp_model.eval()

model = keras.models.load_model('first_train_model')


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


def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    # 배치에서 각 line들의 mask를 valid length 만큼 1로 치환
    # ex) BERT의 입력은 64길이이고 실제 입력되는 문장이 30길이면 마스크는 '1'로 되어진 30개의 요소 + '0'으로 되어진 34개의 요소로 이루어진다.
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.long()


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


def do(data):
    print(data)
    thumbnail_url, video_name, channel_subscriber, upload_date = data
    ## CNN do
    # image = load_image_from_url(thumbnail_url)
    image = load_image_from_url(r"https://i.ytimg.com/vi/iv56zLmWENA/hqdefault.jpg")
    features = cnn_model(image)

    ## NLP do
    # input_ids, valid_length = gen_input_ids(
    #     tokenizer=tokenizer, sentence=[video_name]
    # )
    input_ids, valid_length = gen_input_ids(
        tokenizer=tokenizer, sentence=["헐 스시 존맛탱"]
    )
    input_ids = torch.LongTensor(input_ids)

    # attention mask 생성, 토큰 길이만큼 1 입력 그 이외에는 0
    attention_mask = gen_attention_mask(input_ids, valid_length)

    # 문장의 유형을 구분 ex) context:0, question:1 , 여기는 제목뿐이므로 0
    token_type_ids = torch.zeros_like(input_ids)

    _, pooled_output = nlp_model(input_ids, attention_mask, token_type_ids)
    pooled_output = pooled_output.cpu().detach().numpy()

    ## etc do
    processed_channel_subscriber = np.log(int(channel_subscriber)) - 8.573573524852344
    if processed_channel_subscriber < 0:
        print("Warning: processed_channel_subscriber is minus!!")

    def do_predict(processed_upload_date):
        model_input = np.concatenate((features, pooled_output,
                                      processed_channel_subscriber, processed_upload_date), axis=None)

        predicted_views_log = model.predict(np.array([model_input]))
        views_log = 2 ** (predicted_views_log[0][0] * 17.977546369374885 + 6.90875477931522)
        return views_log

    # 1.0 == 1000일
    views = []

    views.append(do_predict(30.0))
    views.append(do_predict(25.0))
    views.append(do_predict(20.0))
    views.append(do_predict(15.0))
    views.append(do_predict(10.0))
    views.append(do_predict(5.0))
    views.append(do_predict(0.0))

    return views
