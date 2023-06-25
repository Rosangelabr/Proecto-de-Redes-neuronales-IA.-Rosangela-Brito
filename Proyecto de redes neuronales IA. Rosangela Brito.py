import numpy as np
import tensorflow as tf
from tensorflow import keras

# Definir los datos de entrenamiento
A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
B = np.array([[0], [0], [0], [1]])

# Definir la arquitectura de la red neuronal
Prototipo = keras.Sequential()
Prototipo.add(keras.layers.Dense(units=1, activation='sigmoid', input_dim=2))

# Compilar el Prototipoo
Prototipo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el Modelo
Prototipo.fit(A, B, epochs=10000, verbose=0)

# Probar el Modelo entrenado
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = Prototipo.predict(test_input)

# Mostrar los resultados
print("Valores de entrada:")
print(test_input)
print("\nSalida predicha por la red neuronal:")
print(np.round(predicted_output))
