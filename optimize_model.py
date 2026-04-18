import tensorflow as tf
import os

# 1. Carregando o modelo treinado
print("Carregando o modelo treinado...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")

# 2. Convertendo para TensorFlow Lite
print("Convertendo modelo para TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Conversão concluída!")
