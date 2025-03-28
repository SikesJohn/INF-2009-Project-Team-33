import paho.mqtt.client as mqtt
import queue
import ssl
import json
import base64

TOPIC_TEXT = "dashboard/text"
TOPIC_IMAGE = "dashboard/image"
BROKER = "a34z8733tlhxxz-ats.iot.ap-southeast-2.amazonaws.com"


def on_connect(client, userdata, flags, rc):
    """On connect, subscribe to topics"""
    client.subscribe([(TOPIC_TEXT, 0), (TOPIC_IMAGE, 0)])


def on_message(client, userdata, message):
    """On message receive, save data into database and update dashboard"""
    text = json.loads(message.payload.decode("utf-8"))

    if message.topic == TOPIC_TEXT:
        msg_queue.put(("text", text["timestamp"] + " " + text["text"]))
    elif message.topic == TOPIC_IMAGE:
        decoded_content = base64.b64decode(text["text"].encode("utf-8"))
        msg_queue.put(("image", bytearray(decoded_content)))
        

def start_mqtt(message_queue):
    """Start MQTT subscriber service to receive messages from Raspberry Pi"""
    global msg_queue

    msg_queue = message_queue
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.tls_set(
        ca_certs='./rootCA.pem',
        certfile='./aws-certificate.pem.crt',
        keyfile='./aws-private.pem.key',
        tls_version=ssl.PROTOCOL_SSLv23
    )
    client.tls_insecure_set(True)
    client.connect(BROKER, 8883, 60)
    client.loop_forever()
