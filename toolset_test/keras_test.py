from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
import warnings

# load image
img = load_img('koala.png')
print(type(img))                # <class 'PIL.Image.Image'>
print(img.format)               # None
print(img.mode)                 # RGB
print(img.size)                 # (259, 194)
img.show()

# convert image to numpy array
img_array = img_to_array(img)
print("Image to NumPy array: {}".format(type(img_array)))   # <class 'numpy.ndarray'>
print("type:",img_array.dtype)                              # float32
print("shape:",img_array.shape)                             # (194, 259, 3)
# save image from numpy array
save_img("keras_koala.png", img_array)

# convert numpy array to image
img_pil = array_to_img(img_array)
print("Numpy array to Image: {}".format(type(img_pil)))     # <class 'PIL.Image.Image'>
img_pil.show()