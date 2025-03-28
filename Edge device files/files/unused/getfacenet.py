import tensorflow_hub as hub

# Load a FaceNet model from TensorFlow Hub
module_url = "https://tfhub.dev/google/facenet/1"  # Example URL
model = hub.load(module_url)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("facenet_model.tflite", "wb") as f:
    f.write(tflite_model)