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

    def __init__(self, min_h=9, rlsa_val=47):
        self.min_h = min_h
        self.RLSA_VALUE = rlsa_val
    
    def segment(self, img):
        """
        params:
            img::ndarray::~ Grayscale image
            
        Returns an array of tuples (x,y,w,h)
        """
        og = img.copy()
        dots_loc, dots_loc_bin = self.get_dots_loc(og)
        rects = self.segment_dots(dots_loc_bin)
        return rects

    def remove_bin_noise(self, img_bin, min_noise_size=50):
        """
        Removes noise in binary image.

        params:
            img_bin::ndarray::~ binary image, applied threshold
        """
        
        contours, _ = cv2.findContours(img_bin,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area <= min_noise_size:
                (x,y,w,h) = cv2.boundingRect(c)
                black = np.zeros((h, w))
                img_bin[y:y+h, x:x+w] = black
                cv2.fillPoly(img_bin, pts=[c], color=(0,0,0))
        return img_bin
    
    @staticmethod
    def get_dots_loc(og):
        img = og.copy()

        _, img_bin = cv2.threshold(img,
                                   100,
                                   255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = cv2.subtract(255, img_bin)
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
            if area <= 1 and h < 5: # dots size
                img[y:y+h, x:x+w] = og[y:y+h, x:x+w]
                img_bin[y:y+h, x:x+w] = img_bin_og[y:y+h, x:x+w]
        
        return img, img_bin
    
    def segment_dots(self, img_bin):
        og = img_bin.copy()
        
        # Setting RLSA
        RLSA_VALUE = self.RLSA_VALUE
        RLSA_HORIZONTAL = True
        RLSA_VERTICAL = False

        # Heuristics
        WIDTH_MULTIPLIER = 2
        HEIGHT_MULTIPLIER = 1
        
        img_bin = cv2.subtract(255, img_bin)
        
        img_rlsa = rlsa.rlsa(img_bin,
                             RLSA_HORIZONTAL,
                             RLSA_VERTICAL,
                             RLSA_VALUE)
        img_rlsa = cv2.subtract(255, img_bin)
        
        img_rlsa = self.remove_bin_noise(img_rlsa)
        
        (contours, _) = cv2.findContours(img_rlsa,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        rects = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            # Heuristics: self.min_width, self.min_height
            if h < 9 and w > 100:
                rects.append((x,y-21,w,h+21))

        rects.sort(key=lambda tup: tup[1])
        
        return rects

class WordSegmentation:
    """
    Word Segmentation Object
    """
    
    def segment(self, img, kernelSize=25, sigma=11, theta=7, minArea=32):
        """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
        
        Args:
            img: grayscale uint8 image of the text-line to be segmented.
            kernelSize: size of filter kernel, must be an odd integer.
            sigma: standard deviation of Gaussian function used for filter kernel.
            theta: approximated width/height ratio of words, filter function is distorted by this factor.
            minArea: ignore word candidates smaller than specified area.
            
        Returns:
            List of tuples. Each tuple contains the bounding box and the image of the segmented word.
        """
        
        # apply filter kernel
        kernel = self.create_kernel(kernelSize, sigma, theta)
        imgFiltered = cv2.filter2D(img,
                                   -1,
                                   kernel,
                                   borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
        (_, imgThres) = cv2.threshold(imgFiltered,
                                      0,
                                      255,
                                      cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # find connected components. OpenCV: return type differs between OpenCV2 and 3
        if cv2.__version__.startswith('3.'):
            (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # append components to result
        res = []
        for c in components:
            # skip small word candidates
            if cv2.contourArea(c) < minArea:
                continue
            # append bounding box and image of word to result list
            currBox = cv2.boundingRect(c) # returns (x, y, w, h)
            (x, y, w, h) = currBox
            currImg = img[y:y+h, x:x+w]
            res.append((currBox, currImg))

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
    def create_kernel(kernelSize, sigma, theta):
        """
        create anisotropic filter kernel according to given parameters
        """
        
        assert kernelSize % 2 # must be odd size
        halfSize = kernelSize // 2
        
        kernel = np.zeros([kernelSize, kernelSize])
        sigmaX = sigma
        sigmaY = sigma * theta
        
        for i in range(kernelSize):
            for j in range(kernelSize):
                x = i - halfSize
                y = j - halfSize
                
                expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
                xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
                yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
                
                kernel[i, j] = (xTerm + yTerm) * expTerm

        kernel = kernel / np.sum(kernel)
        return kernel

def segment_characters(img_gray, walking_kernel=False):
    """Segments characters"""
    
    gray = img_gray.copy()
    _, img_bin = cv2.threshold(gray,
                               0,
                               255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    img_bin = remove_noise(img_bin,2)

    contours, _ = cv2.findContours(img_bin,
                     cv2.RETR_EXTERNAL,
                     cv2.CHAIN_APPROX_NONE)

    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights) # average height

    widths = [cv2.boundingRect(contour)[2] for contour in contours]
    avgwidth = sum(widths) / len(widths)

    res = []
    if walking_kernel:
        rects = []
        for c in contours:
            (x,y,w,h) = cv2.boundingRect(c)
            if w > avgwidth * 0.8:
                rects.append((x,y,w,h))
                
        # New Algorithm, Walking Kernel.
        for (x,y,w,h) in rects:
            # mid_percent = 0.8
            mid_index = int(h * 0.7)

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
            if h > 3:
                res.append(((x,y,w,h), gray[y:y+h, x:x+w]))
    
    return sorted(res, key=lambda entry:entry[0][0])

if __name__ == '__main__':
    img = cv2.imread('samples/random1.jpg', cv2.IMREAD_GRAYSCALE)
    wordSegmentation = WordSegmentation()
    print(wordSegmentation.segment(img))
