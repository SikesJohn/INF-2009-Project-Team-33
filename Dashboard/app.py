import mqtt_subscriber
import dashboard
import threading
import queue


def main():
    message_queue = queue.Queue()

    mqtt_thread = threading.Thread(target=mqtt_subscriber.start_mqtt, args=(message_queue,))
    mqtt_thread.start()

    dashboard.initialise_dashboard(message_queue)


if __name__ == "__main__":
    main()
