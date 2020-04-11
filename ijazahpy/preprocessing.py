import numpy as np
import cv2

def crop_ijazah(img):
    """
    resize & crop gambar ijazah.

    params:
    img::ndarray::~ Ijazah Image in BGR or Grayscale
    
    Returns an image
    """
    
    X = 112
    Y = 250
    W = 650
    H = 450
    
    img = cv2.resize(img, (850, 1100))
    img = img[Y:Y+H, X:X+W]
    return img

def crop(img, dim, X, Y, W, H):
    """
    General fungsi resize & crop gambar.

    params:
    img::ndarray::~ Ijazah Image in BGR or Grayscale
    
    Returns an image
    """
    
    img = cv2.resize(img, dim)
    img = img[Y:Y+H, X:X+W]
    return img

def remove_noise_bin(img_bin, noise_size=3):
    """
    Removes noises

    params:
    img::ndarray: binary image in numpy array.
    noise_size::float: contour area size of noise

    Returns an image in numpy array.
    """

    original = img_bin.copy()
    
    contours, _ = cv2.findContours(img_bin,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        
        area = cv2.contourArea(c)
        if area <= noise_size:

            black = np.zeros((h,w))
            img_bin[y:y+h, x:x+w] = black

    return img_bin

def to_mnist_ar(img_gray):
    """
    Converts image to mnist like format. 28x28 normalized center.

    params:
    img::ndarray::~ Grayscale image
    """
    
    img = img_gray.copy()
    mnist_x = 28
    mnist_y = 28
    center_x, center_y = int(mnist_x/2), int(mnist_y/2)

    # Aspect Ratio
    current_height = img.shape[0]
    current_width = img.shape[1]
        
    new_height = 22
    new_width = int(current_width * (new_height / current_height))

    dimension = (new_width, new_height)
    
    img = cv2.resize(img, dimension, 0, 0, interpolation=cv2.INTER_AREA)
    
    (thresh, img) = cv2.threshold(img,
                                  128,
                                  255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    blank_image = np.zeros(shape=[28, 28], dtype=np.uint8)

    # Calculate offset
    start_x = center_x - int(0.5 * new_width)
    start_y = center_y - int(0.5 * new_height)

    # Place center
    blank_image[start_y:start_y+new_height, start_x:start_x+new_width] = img    
    blank_image = blank_image.reshape(28, 28)
    
    return remove_noise_bin(blank_image, noise_size=2)

def to_mnist(img_gray):
    img = img_gray.copy()
    dimension = (22, 22)
    img = cv2.resize(img, dimension, 0, 0, interpolation=cv2.INTER_AREA)
    
    (thresh, img) = cv2.threshold(img,
                                  128,
                                  255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    blank_image = np.zeros(shape=[28, 28], dtype=np.uint8)
    blank_image[3:25, 3:25] = img
    blank_image = blank_image.reshape(28, 28)
    return remove_noise_bin(blank_image, noise_size=2)

def prepare_ws_image(img, height):
    """
    convert given image to grayscale image (if needed) and
    resize to desired height
    """

    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)

def prepare_text_image(img, thresh=False):
    """
    Converts image to shape (32, 128, 1) & normalize
    params:
        img::ndarray:~ grayscale/binary image
    returns a binary image with shape (32, 128, 1)
    """
    w, h = img.shape
    
    if thresh:
        _, img = cv2.threshold(img, 
                               128, 
                               255, 
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 0)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 0)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img
