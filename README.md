# ijazahpy
[![Discord Badge](https://discordapp.com/api/guilds/458296099049046018/embed.png)](https://discord.gg/Y8sB4ay)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/madeyoga/ijazahpy/pulls)
[![CodeFactor](https://www.codefactor.io/repository/github/madeyoga/ijazahpy/badge)](https://www.codefactor.io/repository/github/madeyoga/ijazahpy)

Skripsi 2020 jurusan Informatika. Implementasi algoritma DotSegmentation. Python wrapper untuk segmentasi lokasi data dari gambar scan ijazah sekolah Indonesia.

## Permasalahan
Ijazah merupakan sebuah dokumen yang di berikan kepada siswa yang telah menyelesaikan jenjang pendidikan tertentu. Ijazah memiliki jenis form yang beragam. Hal ini dapat diperhatikan seperti ijazah SMA dan ijazah SMK. Ijazah SMK memiliki baris data lebih banyak dibanding Ijazah SMA. 

| Ijazah SMA             |  Ijazah SMK  |
:-------------------------:|:-------------------------:
<img src="https://blue.kumparan.com/image/upload/fl_progressive,fl_lossy,c_fill,q_auto:best,w_640/v1632364372/znp4baxizfyd63f9kxbu.jpg" width="75%" height="75%" />  |  <img src="https://www.quipper.com/id/blog/wp-content/uploads/2021/09/Ijazah-SMK.webp" width="75%" height="75%" />

Begitu juga dengan perbedaan tahun periode pelajaran. Setiap rilis form ijazah tahun ajaran baru, ada kemungkinan bentuk form tersebut berubah. Dapat diperhatikan Ijazah SMA tahun ajaran 2015/2016 berbeda dengan Ijazah tahun ajaran 2020/2021 dibawah. Ijazah 2016 memiliki jumlah baris data yang lebih sedikit bandingkan dengan Ijazah 2021. 

| Ijazah 2015/2016             |  Ijazah 2020/2021  |
:-------------------------:|:-------------------------:
![](https://data03.123doks.com/thumbv2/123dok/000/162/162082/cover.webp)  |  ![](https://www.imrantululi.net/asset/kcfinder/upload/files/SMP%20Depan1.PNG)

Tidak hanya itu, pengisian data pada Ijazah juga masih dilakukan dengan tulisan tangan. Berbeda dengan tulisan komputer yang konsisten secara penulisan, tulisan tangan oleh manusia memiliki ukuran, ketebalan, dan bentuk yang beragam. Tulisan tangan setiap manusia memiliki keunikan masing-masing. Hal ini tentu menjadi tantangan dalam mendapatkan akurasi pengenalan tulisan tangan yang tinggi.

Oleh karena itu, otomatisasi pembacaan data pada ijazah menjadi sebuah tantangan, terutama banyak dilakukan dalam proses pendataan murid, contoh saat penerimaan murid baru atau mahasiswa baru. Penelitian ini dibuat untuk menyelesaikan permasalahan tersebut. Secara garis besar, tahapan program dalam penelitian ini ialah: Input program berupa sebuah gambar ijazah, diproses, dan menghasilkan output sebuah label dan text.

### Tahapan Proses:
1. Mendeteksi lokasi data pada gambar ijazah.
2. Crop bagian lokasi data ijazah.
3. Mengaplikasikan metode _Handwritten Character Recognition_ atau _Handwritten Text Recognition_ pada hasil crop. 

Repositori ini berisi implementasi dari algoritma `dot segmentation`, yang merupakan hasil penelitian untuk mendeteksi lokasi data pada gambar ijazah (proses 1).
Algoritma dot segmentation berkerja dengan mendeteksi lokasi dot atau titik titik pada gambar, dan menyambungkan semua titik titik tersebut, sehingga membentuk sebuah garis atau line. Garis tersebut lah yang kemudian di grow secara vertical sehingga di temukan data tulisan di atasnya.

Repositori proses ke 3: _[Handwritten Text Recognition](https://github.com/madeyoga/Handwritten-Text-Recognition)_ dan _[Handwritten Character Recognition](https://github.com/madeyoga/EMNIST-CNN)_

[Digital collection buku skripsi](https://dewey.petra.ac.id/catalog/digital/detail?id=48502)

## Install
- Clone project
- Open the project directory in cmd
```
C:\...\ijazahpy-master> pip install -r requirements.txt
C:\...\ijazahpy-master> pip install .
```

## Contoh visual segmentasi lokasi data ijazah
```python
import cv2
from ijazahpy.preprocessing import crop_ijazah
from ijazahpy.segmentation import DotsSegmentation

# Initialize object
dot = DotsSegmentation(rlsa_val=47)

# Load gambar 'input' ijazah
img = crop_ijazah(cv2.imread('replace me', cv2.IMREAD_GRAYSCALE))
rects = dot.segment(img, min_width=64)
for box in rects:
    (x,y,w,h) = box
    # visualize
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 2)
# show output
cv2.imshow('visual', img)
```

### Input 
![input_image](https://github.com/madeyoga/ijazahpy/blob/master/output/Input.jpg)
### Output
![output_image](https://github.com/madeyoga/ijazahpy/blob/master/output/Output.jpg)

## Kesimpulan
Algoritma dot segmentation dapat digunakan untuk mendeteksi lokasi data pada Ijazah dengan benar. Algoritma ini tentu memiliki persyratan untuk dapat menghasilkan akurasi deteksi lokasi data yang tinggi, antara lain: 

1. Gambar Ijazah merupakan gambar scan
2. Gambar Ijazah tidak memiliki perspective

Kekurangan yang dimiliki algoritma ialah:

### Algoritma membutuhkan pre-proses gambar untuk menemukan region of interest. 

Anda dapat lihat pada contoh code visual segmentasi di atas, fungsi `crop_ijazah` di gunakan untuk mengambil region of interest pada ijazah. Jika fungsi ini fail dalam menemukan region of interest, maka kemungkinan lokasi data tidak dapat ditemukan. Sehingga proses _character recognition_ juga akan fail

### Pembacaan pada gambar ijazah yang memiliki perspective

Pada gambar yang memiliki perspective atau bukan hasil scan, titik titik pada gambar kemungkinan tidak membentuk garis horizontal jika di hubungkan.

## Pengembangan selanjutnya

Jika anda ingin melanjutkan penelitian ini (Skripsi berikutnya), anda dapat mulai dari mengatasi permasalahan pada algoritma dot segmentation. 

### Pengembangan 1

Agar algoritma dapat berkerja dengan menghilangkan penggunaan fungsi `crop_ijazah`. Fungsi tersebut merupakan pokok permasalahan yang di alami saat ini. Berikutnya, anda dapat test parameter pada algoritma dot segmentation, sehingga tanpa proses crop ijazah, lokasi data masih dapat di temukan.

### Pengembangan 2

Temukan solusi dari masalah pendeteksian lokasi data pada gambar ijazah yang memiliki perspective atau rotasi. Tidak perlu mengembangkan algoritma dot segmentation, anda juga dapat mencoba membuat algoritma semantik anda sendiri. Atau mencoba algoritma object detection, seperti YoloV3, YoloV4, YoloV5, dan lain lain.


--------------------------------
Repositori proses tahapan ke 3: _[Handwritten Text Recognition](https://github.com/madeyoga/Handwritten-Text-Recognition)_ dan _[Handwritten Character Recognition](https://github.com/madeyoga/EMNIST-CNN)_

Link [Digital collection buku skripsi](https://dewey.petra.ac.id/catalog/digital/detail?id=48502)

Semoga bermanfaat 
