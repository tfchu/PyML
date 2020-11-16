from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot

def load_show_pil():
    img = Image.open('koala.png')
    print(img.format)           # PNG
    print(img.size)             # (259, 194)
    print(img.mode)             # RGBA (A: Alpha)
    img.show()

def load_show_matplotlib():
    img = image.imread('koala.png')
    print(img.dtype)            # datatype: float32
    print(img.shape)            # (width, height, color channel): (194, 259, 4)
    pyplot.imshow(img)
    pyplot.show()

def img_to_numpy_array_and_reverse():
    # image to numpy array
    img = Image.open('koala.png')
    print(type(img))           # <class 'PIL.Image.Image'>
    print(img.mode)            # RGBA
    print(img.size)            # (259, 194)
    data = asarray(img)         # rgb value of each pixel from [[(0, 0), (0, 1), ...], [(1, 0), (1, 1), ...], ...]
                                # or np.array(img)
    print(type(data))           # <class 'numpy.ndarray'>
    print(data.shape)           # (194, 259, 4)

    # array to Pillow image
    img2 = Image.fromarray(data)
    print(type(img2))           # <class 'PIL.Image.Image'>
    print(img2.mode)            # RGBA
    print(img2.size)            # (259, 194)
    img2.show()

def gray_img_to_numpy_array_and_reverse():
    img = Image.open('handwriting_5.png').convert('L')       # conver to gray-scale (0 ~ 255)
    print(img.mode)             # L
    print(img.size)             # (28, 28)    

    data = asarray(img)         
    print(type(data))           # <class 'numpy.ndarray'>
    print(data.shape)           # (28, 28) <-- RGBA has 1 extra dimension

    img2 = Image.fromarray(data)
    print(type(img2))           # <class 'PIL.Image.Image'>
    print(img2.mode)            # l
    print(img2.size)            # (28, 28)
    img2.show()


# RGB to Gray scale directly with PIL
# can also convert to numpy array then save
def rgb_to_gray_pil():
    Image.open('koala.png').convert('L').save('koala_gray.png')

# resize
def resize_pil():
    img = Image.open('koala.png')
    print(img.size)             # (259, 194)
    img = img.resize((100, 100))
    print(img.size)             # (100, 100)
    img.show()

def main():
    gray_img_to_numpy_array_and_reverse()

if __name__ == '__main__':
    main()