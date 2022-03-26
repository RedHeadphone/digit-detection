import tensorflow as tf

mnist = tf.keras.datasets.mnist

(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

xtrain = tf.keras.utils.normalize(xtrain,axis=1)
xtest = tf.keras.utils.normalize(xtest,axis=1)

# print(xtrain[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=16,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=16,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=3)

loss,acc =model.evaluate(xtest,ytest)
print("accuracy:",acc)
print("loss:",loss)

model.save('model.h5')
