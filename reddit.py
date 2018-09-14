import cv2 as cv
import numpy as np
import utils as ut

def filtroNegativo(img,colored=False):
    rows,cols = img.shape[:2]
    new_img = np.zeros(img.shape,dtype='uint8')
    for y in xrange(rows):
        for x in xrange(cols):
            if(colored):
                new_img[y][x] = img[y][x]
                new_img[y][x][2] = 255-img[y][x][2]
            else:
                new_img[y][x] = 255-img[y][x]

    return new_img

def gravarArquivo(img,name=None,colorSpace='GRAY'):
    if(name is None):
        name = 'Rtemp.png'

    if(colorSpace == 'HSV'):
        img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    
    return cv.imwrite(name,img)

def prepareImage(filename,color=False):
    img = cv.imread(filename)

    if(color is False):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gravarArquivo(img)
    else:
        img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        gravarArquivo(img,colorSpace='HSV')

    return img

prepareImage('Lenna.png',color=True)
originalImage = cv.imread('Rtemp.png')
hsvConverted = cv.cvtColor(originalImage,cv.COLOR_BGR2HSV)

inverted = ut.filtroNegativo(hsvConverted,colored=True)

import pickle as p
f = file('pic','w')
f.write(p.dumps(inverted))
f.close()

invertedTwice = filtroNegativo(inverted,colored=True)

cv.imwrite('NegativeLenna.png',cv.cvtColor(inverted,cv.COLOR_HSV2BGR))
cv.imwrite('SuposeToBeNormalLenna.jpg',cv.cvtColor(invertedTwice,cv.COLOR_HSV2BGR))
