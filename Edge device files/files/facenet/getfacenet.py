import tensorflow as tf
print(tf.__version__)
model = tf.saved_model.load('./')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('./')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Optionally, you can try int8 here
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('facenet_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as facenet_model.tflite")
