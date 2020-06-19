# ijazahpy
[![Discord Badge](https://discordapp.com/api/guilds/458296099049046018/embed.png)](https://discord.gg/Y8sB4ay)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/madeyoga/ijazahpy/pulls)
[![CodeFactor](https://www.codefactor.io/repository/github/madeyoga/ijazahpy/badge)](https://www.codefactor.io/repository/github/madeyoga/ijazahpy)

Python wrapper untuk segmentasi lokasi data dari gambar scan ijazah sekolah Indonesia.

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
