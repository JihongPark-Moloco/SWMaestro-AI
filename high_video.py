import psycopg2 as pg2
import numpy as np
import pandas as pd

conn = pg2.connect(database="createtrend", user="muna", password="muna112358!", host="222.112.206.190",
                   port="5432")
cur = conn.cursor()

cur.execute(f'SELECT idx, channel_id FROM channel;')
channel_id_list = cur.fetchall()

for idx, channel_id in channel_id_list:
    idx = 530
    cur.execute(f"""
SELECT A.idx, A.video_name, A.video_description, B.views
FROM video A
         LEFT JOIN (SELECT DISTINCT ON (video_idx) video_idx, check_time, views
                    FROM video_views
                    ORDER BY video_idx, check_time DESC) B
                   ON A.idx = B.video_idx
WHERE A.channel_idx = {idx}
  AND A.forbidden = FALSE;
""")
    video_list = pd.DataFrame(cur.fetchall())
    views = video_list[3]
    np.average(views)