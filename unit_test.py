import os

import numpy as np
import cv2

from preprocessing import *
from segmentation import *
from pretrained import CharacterRecognizer

filepath = 'samples/'

files = os.listdir(filepath)

ijazah_cropped = []
segmented_ijazah = []

dots = DotsSegmentation(rlsa_val=47)
word = WordSegmentation()

####for i, filename in enumerate(files):
filename='ijazah1.jpg'
img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
img = crop_ijazah(img)

ijazah_cropped.append(img)

rects = dots.segment(img)

recognizer = CharacterRecognizer()

for j, rect in enumerate(rects):
    x,y,w,h = rect
    segmented_image = img[y:y+h, x:x+w]
    segmented_ijazah.append((rect, segmented_image))
    
##        res = word.segment(segmented_image)
##        for k, w in enumerate(res):
##            (word_box, word_img) = w
##            cv2.imshow(filename+str(j)+str(k), word_img)
    
    character_entries = segment_characters(segmented_image, True)
    cv2.imshow(str(j), segmented_image)
    letters = ""
    # Load model
    if j == 7:
        for k, entry in enumerate(character_entries):
            # converts to mnist like format
            try:
                mnist_like = to_mnist_ar(entry[1])
                pred = recognizer.recognize_char(mnist_like)
                letter = recognizer.prediction_to_char(pred)
                letters+= letter
                cv2.imshow(letter+' '+str(k), mnist_like)
            except Exception as e:
                cv2.imshow(letter+' '+str(k), entry[1])
##                mnist_like = to_mnist(entry[1])
                # predict text
                
           
        print(letters)
        break    
