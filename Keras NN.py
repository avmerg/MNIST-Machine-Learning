#Build a RNN neural network now
#Import keras
from tensorflow import keras
#Load the data again
mnist = tf.keras.datasets.mnist

#Split into train and test
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Regularize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

#One-hotcode y
y_train_hotcode = pd.get_dummies(y_train)

#Create an RNN with Adam Optimizer and Categorical Crossentropy

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1200, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1200, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
          loss='categorical_crossentropy',
          metrics=['accuracy'])

model.fit(x_train, y_train_hotcode,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_split = 0.2,
          shuffle=True)
#Made with reference to official Keras documentation


#Validate on the test set
y_test_hotcode = pd.get_dummies(y_test)

metrics = model.evaluate(x_test, y_test_hotcode, verbose=1)
print("Test loss:", metrics[0])
print("Test accuracy:", metrics[1])
