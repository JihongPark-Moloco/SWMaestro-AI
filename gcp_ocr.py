"""
GCP Vision API의 OCR API를 호출해 썸네일로부터 글귀를 추출해 DB에 저장하는 소스입니다.
"""

import io
import re
import time
from io import BytesIO

import psycopg2 as pg2
import requests
import tqdm
from PIL import Image


def detect_text(content):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # 추출 텍스트 표시 코드
    # for text in texts:
    #     print('\n"{}"'.format(text.description))
    #
    #     vertices = [
    #         "({},{})".format(vertex.x, vertex.y)
    #         for vertex in text.bounding_poly.vertices
    #     ]
    #
    #     print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return texts


def print_texts(texts):
    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        print("bounds: {}".format(",".join(vertices)))


# 430장의 이미지를 합치는 함수
def concat_h(imgs):
    num_rows = 86
    dst = Image.new("RGB", (5 * 480, num_rows * 360))

    for index, img in enumerate(imgs):
        if index >= 5 * num_rows:
            break
        row = index // 5
        col = index - (row * 5)
        dst.paste(img, (col * 480, row * 360))

    return dst

def get_image_from_url(url):
    # 가로로 최대 136
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).resize((480, 360))


# 이미지를 합쳐서 OCR API를 요청하고 이를 다시 나누어 DB에 저장합니다.
while True:
    conn = pg2.connect(
        database="createtrend",
        user="muna",
        password="muna112358!",
        host="ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com",
        port="5432",
    )
    cur = conn.cursor()
    print("13760")
    cur.execute(
        """SELECT idx, thumbnail_url from video where status = true AND thumbnail_processed = false AND forbidden = false AND channel_idx IN
                   (66, 98, 218, 211, 181, 197, 30, 17, 169, 171, 14, 138, 99, 60, 206, 122, 94, 43, 38, 115, 55, 85, 200, 182, 134, 112,
 52, 157, 123, 124, 119, 207, 156, 107, 162, 89, 71, 51, 252, 1588, 76, 220, 263, 26, 130, 12, 163, 172, 148, 153, 164,
 103, 95, 145, 133, 161, 228, 117, 70, 149, 116, 80, 137, 158, 9, 170, 147, 23, 11, 215, 83, 72, 168, 192, 29, 195, 91,
 79, 201, 106, 113, 104, 28, 64, 2) LIMIT 430"""
    )
    rows = cur.fetchall()
    thumbnail_indexs = [row[0] for row in rows]
    thumbnail_urls = [row[1] for row in rows]

    # 이미지 합치기 진행
    imgs = [get_image_from_url(thumbnail_url) for thumbnail_url in tqdm.tqdm(thumbnail_urls)]
    img = concat_h(imgs)
    img_bytes_arr = io.BytesIO()
    img.save(img_bytes_arr, format="JPEG")
    img_bytes_arr = img_bytes_arr.getvalue()
    texts = detect_text(img_bytes_arr)

    # 영역별로 나누어 DB에 저장
    for i, thumbnail_index in enumerate(tqdm.tqdm(thumbnail_indexs)):
        row = i // 5
        col = i - row * 5

        words = " ".join(
            [
                t.description
                for t in texts[1:]
                if col * 480 <= t.bounding_poly.vertices[0].x <= (col + 1) * 480
                and row * 360 <= t.bounding_poly.vertices[0].y <= (row + 1) * 360
            ]
        )

        words = re.sub("'", "", words)

        sql = f"""INSERT INTO thumbnail_logo (video_idx, logo)
                VALUES ('{thumbnail_index}', '{words[:499]}') ON CONFLICT DO NOTHING;"""
        cur.execute(sql)

    conn.commit()
    conn.close()
    time.sleep(2)
