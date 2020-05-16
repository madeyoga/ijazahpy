import Levenshtein as lv

ijazah_dictionary = {
    # SMA/SMK 2013/2014 2015/2016
    'nama': 'Nama',
    'tempatdantanggallahir': 'Tempat dan Tanggal Lahir',
    'namaorangtua/wali': 'Nama Orang Tua/Wali',
    'nomorinduksiswa': 'Nomor Induk Siswa',
    'nomorinduksiswanasional': 'Nomor Induk Siswa Nasional',
    'nomorpesertaujiannasional': 'Nomor Peserta Ujian Nasional',
    'sekolahasal': 'Sekolah Asal',
    'programkeahlian': 'Program Keahlian',
    'paketkeahlian': 'Paket Keahlian',
    # SMA 2008/2009
    'namaorangtua': 'Nama Orang Tua',
    'nomorinduk': 'Nomor Induk',
    'nomorpeserta': 'Nomor Peserta',
    # SMA Tahun < 2011 Paket C dll.
    'kelompokbelajar': 'Kelompok Belajar',
    'desa/kelurahan': 'Desa/Kelurahan',
    'kecamatan': 'Kecamatan',
    }

def process_label(label, metrics='jaro', tolerance=0.5):
    
    if label == '':
        return label
    
    if metrics == 'jaro':
        distance_function = lv.jaro
    elif metrics == 'ratio':
        distance_function = lv.ratio
    else:
        raise Exception('Invalid metrics: {}. valid metrics: jaro, ratio.'.format(metrics))

    # Get the highest similarity
    result = ''
    highest = -1
    for key in ijazah_dictionary.keys():
        current_score = distance_function(key, label)
        if highest < current_score:
            result = ijazah_dictionary[key]
            highest = current_score

    if highest > tolerance:
        return result
    
    return label

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
        if j == 2:
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
    return 0

def test_word_segmentation():
    filepath = "G:\\Kuliah\\skripsi\\github\\SimpleApp-master\\SimpleApp\\media\\"
    filename = "2Ijazah1.jpg"
    
    img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('og', img)

##    prepared = prepare_ws_image(img, 32)
##    cv2.imshow('prepared1', prepared)
    
    ws = WordSegmentation()
    res = ws.segment(img, imshow_steps=True)

    for k, w in enumerate(res):
        (word_box, word_img) = w
        cv2.imshow(str(k), word_img)
        cv2.rectangle(img, (word_box[0], word_box[1]), (word_box[0]+word_box[2], word_box[1]+word_box[3]), (0, 0, 0), 1)
    cv2.imshow('res', img)
    return 0

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

def test_dots():
    dot = DotsSegmentation()
    # Segment 1 with crop5
    dot.RLSA_VALUE = 47

    path = 'samples/'
    files = os.listdir(path)
    for filename in files:        
        img = crop_ijazah(cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE))
        rects = dot.segment(img, min_width=64)
        for box in rects:
            (x,y,w,h) = box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 2)
        cv2.imwrite('G:\\Kuliah\\skripsi\\github\\ijazahpy\\images\\'+filename, img)
    
    # Segment 2 without crop ijazah
##    (img_gray, img_bin) = dot.get_dots_loc(img, 5)
##
##    connected_dots = dot.connect_horizontal(img_bin)
##    cv2.imshow('c', connected_dots)
##    cv2.imwrite('c.png', connected_dots)
####    cv2.imshow('g', img_gray)
####    cv2.imshow('b', img_bin)
####    cv2.imwrite('g.png', img_gray)
####    cv2.imwrite('b.png', img_bin)
    return 0

def test_ijazah_kuliah():
    dot = DotsSegmentation()
    
    img = crop_ijazah(cv2.imread('samples/random21.jpg', cv2.IMREAD_GRAYSCALE))
    cv2.imshow('b', img)
    
    dots_img, img_bin = dot.get_dots_loc(img, 10)
    cv2.imshow('a', img_bin)
    
    rects = dot.segment_dots(img_bin)

    for r in rects:
        (x,y,w,h) = r
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)
    cv2.imshow('c', img)
    

if __name__ == '__main__':
    import os
    import numpy as np
    import cv2
    from preprocessing import *
    from segmentation import *
##    from pretrained import TextRecognizer
##    import pytesseract
    
    test_ijazah_kuliah()
##    test_word_segmentation()
