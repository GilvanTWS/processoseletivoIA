import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 1. Carregando o dataset MNIST
print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalizando os dados (valores de 0-255 para 0-1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# 3. Ajustando o formato para a CNN (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# 4. Construindo a CNN
print("Construindo o modelo...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 5. Compilando o modelo
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Treinando o modelo
print("Treinando o modelo...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 7. Avaliando no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no teste: {test_acc:.4f}")

# 8. Salvando o modelo
model.save("model.h5")
print("Modelo salvo em model.h5")