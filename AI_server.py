import json
import pika
import ai




def callback(ch, method, properties, body):
    print(" [x] Received %r" % body.decode())
    method_frame, header_frame, data = channel.basic_get(queue=body.decode())
    data = json.loads(data)
    channel.queue_delete(queue=body.decode())
    # data = [thumbnail_url, video_name, channel_subscriber, upload_date]
    views = ai.do(data)
    channel.basic_publish(exchange="", routing_key=body.decode() + "_r", body=json.dumps(views))


credentials = pika.PlainCredentials("muna", "muna112358!")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        "13.124.107.195", 5672, "/", credentials, heartbeat=10, blocked_connection_timeout=10,
    )
)

channel = connection.channel()

channel.basic_consume(queue="request_ai_process", on_message_callback=callback, auto_ack=True)

print(" [*] Waiting for messages. To exit press CTRL+C")
channel.start_consuming()
