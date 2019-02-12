#Make a convolutional neural network
#Data Science Challenge 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train_h = pd.get_dummies(y_train)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(15, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(), 
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model.fit(x_train, y_train_h,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_split = 0.2,
          shuffle=True)

#Made with reference to official Keras documentation