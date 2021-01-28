"""
서버에서 오는 요청을 처리하기 위한 AI 서버 구동 코드입니다.
RabbitMQ를 이용해서 통신합니다.
"""

import json

import pika

import ai


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body.decode())
    try:
        method_frame, header_frame, data = channel.basic_get(queue=body.decode())
        data = json.loads(data)
        channel.queue_delete(queue=body.decode())
        views = ai.do(data)
        channel.basic_publish(exchange="", routing_key=body.decode() + "_r", body=json.dumps(views))
    except Exception as e:
        print(e)
        channel.queue_delete(queue=body.decode())
        channel.queue_delete(queue=body.decode() + "_r")


credentials = pika.PlainCredentials(None, None)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        None, None, "/", credentials, heartbeat=10, blocked_connection_timeout=10,
    )
)

channel = connection.channel()

channel.basic_consume(queue="request_ai_process", on_message_callback=callback, auto_ack=True)

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
