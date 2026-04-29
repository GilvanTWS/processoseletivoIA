import tensorflow as tf
import os

# 1. Carregando o modelo treinado
print("Carregando o modelo treinado...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")

# 2. Aplicando Dynamic Range Quantization
print("\nAplicando Dynamic Range Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_dynamic = converter.convert()

dynamic_path = "model.tflite"
with open(dynamic_path, "wb") as f:
    f.write(tflite_model_dynamic)
print(f"Modelo Dynamic salvo em {dynamic_path}")

# 3. Aplicando Float16 Quantization
print("\nAplicando Float16 Quantization...")
converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.target_spec.supported_types = [tf.float16]
tflite_model_float16 = converter2.convert()

float16_path = "model_float16.tflite"
with open(float16_path, "wb") as f:
    f.write(tflite_model_float16)
print(f"Modelo Float16 salvo em {float16_path}")

# 4. Comparando os resultados
print("\n--- Comparação de Tamanhos ---")
print(f"Modelo original (.h5):            {os.path.getsize('model.h5') / 1024:.1f} KB")
print(f"Dynamic Range Quantization:       {os.path.getsize(dynamic_path) / 1024:.1f} KB")
print(f"Float16 Quantization:             {os.path.getsize(float16_path) / 1024:.1f} KB")