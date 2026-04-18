import tensorflow as tf
import os

# 1. Carregando o modelo treinado
print("Carregando o modelo treinado...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")
