import math
import numpy as np
import cv2
from pythonRLSA import rlsa
from ijazahpy.preprocessing import remove_noise_bin as remove_noise

class DotsSegmentation:
    """
    Pendekatan segmentasi baru
    Segmentasi dilakukan dengan mengambil area dots pada ijazah

    methods:
        segment::
        remove_bin_noise::
        get_dots_locs::
        segment_dots::
    
    """

    def __init__(self, rlsa_val=47):
        self.RLSA_VALUE = rlsa_val
    
    def segment(self, img, dot_size=3, min_width=72, min_height=9, imshow=False):
        """
        params:
            img::ndarray::~ Grayscale image
            
        Returns an array of tuples (x,y,w,h)
        """
        og = img.copy()
        dots_loc, dots_loc_bin = self.get_dots_loc(og, dot_size=dot_size)
        
        if imshow:
            cv2.imshow('dots', dots_loc_bin)
            
        rects = self.segment_dots(dots_loc_bin,
                                  min_width=min_width,
                                  min_height=min_height,
                                  imshow=imshow)
        return rects

    def remove_bin_noise(self, img_bin, min_line_width=50):
        """
        Removes noise in binary image.

        params:
            img_bin::ndarray::~ binary image, applied threshold
        """
        
        contours, _ = cv2.findContours(img_bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if w <= min_line_width:
                (x,y,w,h) = cv2.boundingRect(c)
                black = np.zeros((h, w))
                img_bin[y:y+h, x:x+w] = black
                cv2.fillPoly(img_bin, pts=[c], color=(0,0,0))
        return img_bin
    
    @staticmethod
    def get_dots_loc(og, dot_size=3):
        img = og.copy()

        _, img_bin = cv2.threshold(img,
                                   100,
                                   255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img_bin_og = img_bin.copy()

        contours, _ = cv2.findContours(img_bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        
        # Removes except dots
        for c in contours:
            area = cv2.contourArea(c)
            (x,y,w,h) = cv2.boundingRect(c)
            
            if area >= 1 or h > 3:
                white = np.full((h, w), 255)
                img[y:y+h, x:x+w] = white
                cv2.fillPoly(img, pts=[c], color=(255,255,255))

                black = np.zeros((h,w))
                img_bin[y:y+h, x:x+w] = black
        
        # Adds dot in removed segment
        for c in contours:
            area = cv2.contourArea(c)
            (x,y,w,h) = cv2.boundingRect(c)
            if area <= dot_size and h < 5: # dots size
                img[y:y+h, x:x+w] = og[y:y+h, x:x+w]
                img_bin[y:y+h, x:x+w] = img_bin_og[y:y+h, x:x+w]
        
        return img, img_bin
    
    def segment_dots(self, img_bin, field_height=22, min_width=72, min_height=9, imshow=False):
        """Connect dots horizontal & Find contours"""
        
        img_rlsa = self.connect_horizontal(img_bin, self.RLSA_VALUE)
                                         
        if imshow:
            cv2.imshow('connect', img_rlsa)
        
        return self.segment_line(img_rlsa, field_height, min_width, min_height)

    @staticmethod
    def connect_horizontal(img_bin, rlsa_val=47):
        """Connect dots horizontal"""
        
        og = img_bin.copy()
        
        # Setting RLSA
        RLSA_VALUE = rlsa_val
        RLSA_HORIZONTAL = True
        RLSA_VERTICAL = False
        
        img_bin = cv2.subtract(255, img_bin)
        img_rlsa = rlsa.rlsa(img_bin,
                             RLSA_HORIZONTAL,
                             RLSA_VERTICAL,
                             RLSA_VALUE)
        img_rlsa = cv2.subtract(255, img_bin)
        
        return img_rlsa

    @staticmethod
    def segment_line(img_bin, field_height=22, min_width=72, min_height=9):
        """Segment connected dots"""
        
        img_rlsa = img_bin.copy()
        
        (contours, _) = cv2.findContours(img_rlsa,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        rects = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if h < min_height and w > min_width:
                rects.append((x,y-field_height,w,h+field_height))

        rects.sort(key=lambda tup: tup[1])
        
        return rects
    
class WordSegmentation:
    """
    Word Segmentation Object
    """
    
    def segment(self, img, kernelSize=25, sigma=11,
                theta=7, minArea=32, imshow_steps=False):
        """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
        
        params:
            img::ndarray::~ grayscale uint8 image of the text-line to be segmented.
            kernel_size::uint::~ size of filter kernel, must be an odd integer.
            sigma::uint::~ standard deviation of Gaussian function used for filter kernel.
            theta::uint::~ approximated width/height ratio of words, filter function is distorted by this factor.
            minArea::uint::~ ignore word candidates smaller than specified area.
            
        Returns a list of tuples. Each tuple contains the bounding box and the image of the segmented word.
        """
        
        # apply filter kernel
        kernel = self.create_kernel(kernelSize, sigma, theta)
        filtered_img = cv2.filter2D(img,
                                   -1,
                                   kernel,
                                   borderType=cv2.BORDER_REPLICATE).astype(
                                       np.uint8)
        
        (_, bin_img) = cv2.threshold(filtered_img,
                                      0,
                                      255,
                                      cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        
        if imshow_steps:
            cv2.imshow('filtered', filtered_img)
            cv2.imshow('blob', bin_img)
            
        (contours, _) = cv2.findContours(bin_img,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

        # append contours to result
        res = []
        for c in contours:
            # skip small word candidates
            if cv2.contourArea(c) < minArea:
                continue
            # append bounding box and image of word to result list
            box = cv2.boundingRect(c) 
            (x, y, w, h) = box
            cropped_img = img[y:y+h, x:x+w]
            res.append((box, cropped_img))

        # return list of words, sorted by x-coordinate
        return sorted(res, key=lambda entry:entry[0][0])

    @staticmethod
    def prepare_img(img, height):
        """convert given image to grayscale image (if needed) and resize to desired height"""
        assert img.ndim in (2, 3)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img.shape[0]
        factor = height / h
        return cv2.resize(img, dsize=None, fx=factor, fy=factor)

    @staticmethod
    def create_kernel(kernel_size, sigma, theta):
        """
        create anisotropic filter kernel according to given parameters
        """
        
        assert kernel_size % 2 # must be odd size
        half_size = kernel_size // 2
        
        kernel = np.zeros([kernel_size, kernel_size])
        sigma_x = sigma
        sigma_y = sigma * theta
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - half_size
                y = j - half_size
                
                expTerm = np.exp(-x**2 / (2 * sigma_x) - y**2 / (2 * sigma_y))
                xTerm = (x**2 - sigma_x**2) / (2 * math.pi * sigma_x**2 * sigma_y)
                yTerm = (y**2 - sigma_y**2) / (2 * math.pi * sigma_y**2 * sigma_x)
                
                kernel[i, j] = (xTerm + yTerm) * expTerm

        kernel = kernel / np.sum(kernel)
        return kernel

# Fails
def segment_characters(img_gray, walking_kernel=False, noise_remove=False):
    """Segments characters"""
    
    gray = img_gray.copy()
    
    _, img_bin = cv2.threshold(gray,
                           0,
                           255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    if noise_remove:
        img_bin = remove_noise(img_bin,3)
        
    contours, _ = cv2.findContours(img_bin,
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
    res = []
    if walking_kernel:

        heights = [cv2.boundingRect(contour)[3] for contour in contours]
        avgheight = sum(heights)/len(heights) # average height

        widths = [cv2.boundingRect(contour)[2] for contour in contours]
        avgwidth = sum(widths) / len(widths)
        rects = []
        
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if w > avgwidth * 0.8:
                rects.append((x,y,w,h))
                
        # New Algorithm, Walking Kernel.
        for (x,y,w,h) in rects:
            # mid_percent = 0.8
            mid_index = int(h * 0.5)

            # kernel_width = 2
            kernel = (2,h)
            img_temp = img_bin[y:y+h, x:x+w]
            img_temp = remove_noise(img_temp, 6)
            
            # strides = 1
            for step in range(0, w, 1):
                img_target = img_temp[0:kernel[1], step:step+kernel[0]]
                pixels_value = np.sum(img_target)
                # minimum_pixels_value = 1020
                if pixels_value <= 1020:
                    img_target[mid_index:] = 0
                    img_bin[y:y+h, x+step:x+step+kernel[0]] = img_target
                    
                # x offset
                if step >= w * 0.8:
                    break

        contours, _ = cv2.findContours(img_bin,
                         cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_NONE)

        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if w > 3 and h > 3:
                res.append(((x,y,w,h), gray[y:y+h, x:x+w]))
    else:
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if h > 3 and h < 60:
                res.append(((x,y,w,h), gray[y:y+h, x:x+w]))
    
    return sorted(res, key=lambda entry:entry[0][0])

# Fails
def segment_character2(img_gray):
    gray = img_gray.copy()
    _, img_bin = cv2.threshold(gray,
                               0,
                               255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    img_bin = remove_noise(img_bin, 3)
    
    kernel = np.ones((2,1))
    erosion = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)# cv2.erode(img_bin, kernel, iterations=1)
    cv2.imshow('erosion', erosion)

    ero_inv = cv2.subtract(255, erosion)
    
    img_rlsa = rlsa.rlsa(ero_inv,
                         True,
                         True,
                         10)
    res = cv2.subtract(255, img_rlsa)

    cv2.imshow('res', res)

    contours, _ = cv2.findContours(res,
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)

    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        if h > 3:
            cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 0, 0), 1)

    cv2.imshow('final', gray)
    return

def segment_words(img_gray, rlsa_val=7, bin_result=False):
    """ Segment words with RLSA

    params:
        img_gray::ndarray:~ grayscale image
        rlsa_val::integer:~ value for run length smoothing algorithm

    Returns a list of tuple -> ((x,y,w,h), image_array)
    """
    
    gray = img_gray.copy()
    
    _, img_bin = cv2.threshold(gray,
                               0,
                               255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    img_bin = remove_noise(img_bin, 30)

    img_bin_og = img_bin.copy()
    
    img_bin = cv2.subtract(255, img_bin)

    img_rlsa = rlsa.rlsa(img_bin,
                         True,
                         True,
                         rlsa_val)

    res = cv2.subtract(255, img_rlsa)

    contours, _ = cv2.findContours(res,
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)

    res = []
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        if h > 3:
            if bin_result:
                cropped_img = img_bin_og[y:y+h, x:x+w]
            else:
                cropped_img = gray[y:y+h, x:x+w]
                
            zipp = ((x,y,w,h), cropped_img)
            res.append(zipp)
            
    return res

##if __name__ == '__main__':
##    filepath = 'G:\Kuliah\skripsi\github\SimpleApp-master\SimpleApp\media/'.replace('\\', '/')
##    filename = '2ijazah3.jpg'
##    print(filepath+filename)
##    img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
##    
##    res = segment_words(img, bin_result=True)
##    
####    img = cv2.imread('samples/random1.jpg', cv2.IMREAD_GRAYSCALE)
####    wordSegmentation = WordSegmentation()
####    res = wordSegmentation.segment(img, imshow_steps=True)
##    for i, entry, in enumerate(res):
##        (box, img) = entry
##        cv2.imshow(str(i), img)
