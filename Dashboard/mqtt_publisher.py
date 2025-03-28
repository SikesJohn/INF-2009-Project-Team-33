import paho.mqtt.client as mqtt
import time
import datetime
import ssl
import json
import base64

TOPIC_TEXT = "dashboard/text"
TOPIC_IMAGE = "dashboard/image"
BROKER = "a34z8733tlhxxz-ats.iot.ap-southeast-2.amazonaws.com"


class mqtt_pub():
    def __init__(self):
        """Start MQTT publisher service"""
        self.client = mqtt.Client()
        self.client.tls_set(ca_certs='./rootCA.pem', certfile='./aws-certificate.pem.crt', keyfile='./aws-private.pem.key', tls_version=ssl.PROTOCOL_SSLv23)
        self.client.tls_insecure_set(True)
        self.client.connect(BROKER, 8883, 60)
        self.client.loop_start()
        time.sleep(1) # Give time for the initialisation

    def send_text(self, text):
        """Send text to dashboard"""
        message = json.dumps({"timestamp": str(datetime.datetime.now()), "text": text, "type": "text"}, indent=2)
        self.client.publish(TOPIC_TEXT, payload=message, qos=0, retain=False)
        time.sleep(0.1) # To space out sending of the content

    def send_image(self, image_path):
        """Send image to dashboard"""
        with open(image_path, "rb") as f:
            file_content = f.read()
            file_content = base64.b64encode(file_content).decode("utf-8")

            message = json.dumps({"timestamp": str(datetime.datetime.now()), "text": file_content, "type": "image"}, indent=2)
            self.client.publish(TOPIC_IMAGE, payload=message, qos=0, retain=False)

        # When image sent, also send intruder log and timestamp
        time.sleep(0.1) # To space out sending of the content
        publish_text = json.dumps({"timestamp": str(datetime.datetime.now()), "text": "Intruder detected!", "type": "text"}, indent=2)
        self.client.publish(TOPIC_TEXT, payload=publish_text, qos=0, retain=False)
        time.sleep(0.1) # To space out sending of the content
