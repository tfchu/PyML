from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot

def load_show_pil():
    img = Image.open('koala.png')
    print(img.format)           # PNG
    print(img.size)             # (259, 194)
    print(img.mode)             # RGBA
    img.show()

def load_show_matplotlib():
    img = image.imread('koala.png')
    print(img.dtype)            # datatype: float32
    print(img.shape)            # (width, height, color channel): (194, 259, 4)
    pyplot.imshow(img)
    pyplot.show()

def img_to_numpy_array():
    # image to numpy array
    img = Image.open('koala.png')
    data = asarray(img)         # rgb value of each pixel from [[(0, 0), (0, 1), ...], [(1, 0), (1, 1), ...], ...]
                                # or np.array(img)
    print(type(data))           # <class 'numpy.ndarray'>
    print(data.shape)           # (194, 259, 4)

    # array to Pillow image
    img2 = Image.fromarray(data)
    print(type(img2))           # <class 'PIL.Image.Image'>
    print(img2.mode)            # RGBA
    print(img2.size)            # (259, 194)


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
    resize_pil()

if __name__ == '__main__':
    main()