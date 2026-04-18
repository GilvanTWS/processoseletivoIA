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

# 3. Aplicando otimização Dynamic Range Quantization
print("Aplicando otimização Dynamic Range Quantization...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_optimized = converter.convert()
print("Otimização concluída!")


# 4. Salvando o modelo otimizado
output_path = "model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model_optimized)

print(f"Modelo otimizado salvo em {output_path}")
print(f"Tamanho do modelo original: {os.path.getsize('model.h5') / 1024:.1f} KB")
print(f"Tamanho do modelo otimizado: {os.path.getsize(output_path) / 1024:.1f} KB")