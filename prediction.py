from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage import transform

IMAGE_FILE_NAME = 'test/cats/cat.0.jpg'


def load(filename: str):
    psf = np.ones((200, 200)) / 25
    _image = image.load_img(filename, color_mode='grayscale', target_size=(200, 200))
    np_image = np.array(_image)
    np_image = transform.resize(np_image, (200, 200, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


image = load(IMAGE_FILE_NAME)
model = models.load_model('model', compile=True)
class_prob = model.predict(image)
print(class_prob)
class_prob =  model.predict_classes(image)
print(class_prob)
