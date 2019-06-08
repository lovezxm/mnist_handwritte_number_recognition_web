import numpy as np
from scipy.misc import imread, imresize
from keras.models import load_model

loaded_model = load_model('model.hdf5')
print("Loaded Model from disk")

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
x = imread('output.png',mode='L')
x = np.invert(x)
x = imresize(x,(28,28))
import matplotlib.pyplot as plt
plt.imshow(np.uint8(x))
plt.show()
x = x.reshape(1,28,28,1)
x = x/255
out = loaded_model.predict(x)
print(out)
print(np.argmax(out,axis=1))
