import os

import numpy as np
import cv2

from preprocessing import *
from segmentation import *
from pretrained import TextRecognizer

import pytesseract

def test_recognize():
    filepath = 'samples/'

    files = os.listdir(filepath)

    ijazah_cropped = []
    segmented_ijazah = []

    dots = DotsSegmentation(rlsa_val=50)
    word = WordSegmentation()

    ####for i, filename in enumerate(files):
    filename='random12.jpg'
    img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
    img = crop_ijazah(img)

    ijazah_cropped.append(img)

    rects = dots.segment(img)

    ##recognizer = CharacterRecognizer()
    text_recognizer = TextRecognizer()

    for j, rect in enumerate(rects):
        x,y,w,h = rect
        segmented_image = img[y:y+h, x:x+w]
        segmented_ijazah.append((rect, segmented_image))
        
        prepared_img = prepare_ws_image(segmented_image, 50)
        _, curr_img = cv2.threshold(prepared_img, 
                                            128, 
                                            255, 
                                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        prepared_img = remove_noise_bin(curr_img, 30)
        prepared_img = cv2.subtract(255,prepared_img)
        res = word.segment(prepared_img)
        
    ##        for k, w in enumerate(res):
    ##            (word_box, word_img) = w
    ##            cv2.imshow(filename+str(j)+str(k), word_img)
        
    ##    character_entries = segment_characters(segmented_image, True)
    ##    cv2.imshow(str(j), segmented_image)
        
        letters = ""
        # Load model
        if j == 4:
            cv2.imshow('a', prepared_img)
            for k, entry in enumerate(res):
                box, curr_img = entry
                og = curr_img.copy()
                curr_img = cv2.subtract(255,curr_img)

                if curr_img.shape[0] < 50 and curr_img.shape[1] < 50:
                    continue
                cv2.imshow('b', curr_img)
                
                curr_img = prepare_for_tr(curr_img, thresh=False)
                
                cv2.imshow('test' + str(k), curr_img)
                print(text_recognizer.recognize(curr_img))
##                print(pytesseract.image_to_string(curr_img.reshape(32,128)))
            break
    ##        for k, entry in enumerate(character_entries):
    ##            # converts to mnist like format
    ##            try:
    ##                mnist_like = to_mnist_ar(entry[1])
    ##                pred = recognizer.recognize_char(mnist_like)
    ##                letter = recognizer.prediction_to_char(pred)
    ##                letters+= letter
    ##                cv2.imshow(letter+' '+str(k), mnist_like)
    ##            except Exception as e:
    ##                cv2.imshow(letter+' '+str(k), entry[1])
    ####                mnist_like = to_mnist(entry[1])
    ##                # predict text
            print(letters)
            break
    
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    SZ = img.shape[1]
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,
                         M,
                         (SZ, SZ),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

if __name__ == '__main__':
    test_recognize()
