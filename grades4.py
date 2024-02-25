import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import json

grades_file = open('grades.json')
averages_file = open('averages.json')
grades = np.array(json.load(grades_file))
average = np.array(json.load(averages_file))

print('Files loaded!')

model = Sequential([
    Dense(64, activation='relu', input_shape=[3]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4)
])

print('Compiling model...')

model.compile(
    optimizer=Adam(0.01),
    loss='mean_squared_error'
)

print('Model compiled!')

print('Training model...')

history = model.fit(grades, average, epochs=50)

print('History: ')

plt.xlabel("# epoch")
plt.ylabel("Magnitud de Perdida")
plt.plot(history.history['loss'])

print('Testing prediction.')

input_data = [85, 80, 90]
reshaped_data = tf.reshape(input_data, shape=(-1, 3))
result = model.predict(reshaped_data)

print(f"Predicted: {result[0]}, {result[0].sum() / 4}")

guessed = round(result[0].sum() / 4) == 85

print(f"Prediction is {guessed}")

if guessed:
	print('Model has successfully generated the requested grades!')
	print('Exporting model now...')
	model.save('grades_4.h5')
	print('Model exported as "grades_4.h5"')
else:
	print('The calculation was wrong!')
