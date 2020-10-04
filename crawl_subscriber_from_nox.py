import requests
import psycopg2 as pg2
import pika
import traceback

credentials = pika.PlainCredentials('muna', 'muna112358!')
connection = pika.BlockingConnection(pika.ConnectionParameters('13.124.107.195', 5672, '/',
                                                               credentials, heartbeat=0,
                                                               blocked_connection_timeout=None))
channel = connection.channel()
channel.basic_qos(prefetch_count=1)


def do_sub(channel_id):
    try:
        # channel_id = 'UCVXx89z5j3gkQMGScNqmWjw'
        IP = "ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com"

        conn = pg2.connect(database="createtrend", user="muna", password="muna112358!", host=IP,
                           port="5432")
        conn.autocommit = False
        cur = conn.cursor()

        response = requests.get(
            f'https://kr.noxinfluencer.com/ws/star/trend/{channel_id}?type=total&dimension=sub&interval=daily')
        datas = response.json()['retData']['history']

        for data in datas:
            sql = f"""
        INSERT INTO channel_subscriber (channel_idx, subscriber_num, check_time)
        VALUES (
            (SELECT idx FROM channel WHERE channel_id = '{channel_id}'), 
            '{data['value']}', 
            to_timestamp('{data['date']}', 'YYYY-MM-DD')
        )"""
            cur.execute(sql)
            # print(data['date'], data['value'])

        conn.commit()
        conn.close()
        return True
    except:
        print(traceback.format_exc())
        conn.close()
        return False

def do_view(channel_id):
    try:
        # channel_id = 'UCVXx89z5j3gkQMGScNqmWjw'
        IP = "ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com"

        conn = pg2.connect(database="createtrend", user="muna", password="muna112358!", host=IP,
                           port="5432")
        conn.autocommit = False
        cur = conn.cursor()

        response = requests.get(
            f'https://kr.noxinfluencer.com/ws/star/trend/{channel_id}?type=total&dimension=view&interval=daily')
        datas = response.json()['retData']['history']

        for data in datas:
            sql = f"""
        INSERT INTO channel_views (channel_idx, view_count, check_time)
        VALUES (
            (SELECT idx FROM channel WHERE channel_id = '{channel_id}'), 
            '{data['value']}', 
            to_timestamp('{data['date']}', 'YYYY-MM-DD')
        )"""
            cur.execute(sql)
            # print(data['date'], data['value'])

        conn.commit()
        conn.close()
        return True
    except:
        print(traceback.format_exc())
        conn.close()
        return False

# do_view('UCLkAepWjdylmXSltofFvsYQ')


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body.decode())
    if do_view(body.decode()):
        channel.basic_ack(delivery_tag=method.delivery_tag, multiple=False)
    else:
        channel.basic_nack(delivery_tag=method.delivery_tag, multiple=False, requeue=False)
    # if YouTube_Crawler.main(body.decode()):
    #     return
    # else:
    #     channel.basic_publish(exchange ='', routing_key = 'URL_dead', body=body.decode())


# auto_ack를 False로 수정했습니다.
# auto_ack가 True일 경우 메세지를 꺼내오는 순간에 메세지 큐에서 해당 메세지는 삭제됩니다.
# 만약 해당 주소를 크롤러가 받아와서 도는 도중에 크롤러가 중간에 에러를 띄우고
# 프로세스가 중간에 죽어버릴 경우 정상적으로 처리가 안되었지만 메세지 큐에는 해당 주소가 없는 상황이 발생합니다.
channel.basic_consume(queue='URL2', on_message_callback=callback, auto_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
