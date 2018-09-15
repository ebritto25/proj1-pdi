from __future__ import division
import Tkinter as tk
import tkFileDialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def abrirArquivo():
    tk.Tk().withdraw() #ocultar janela raiz
    arq = tkFileDialog.askopenfilename()
    print(arq)
    return arq

def gravarArquivo(img,name=None,colorSpace='GRAY'):
    if(name is None):
        name = 'temp.png'

    if(colorSpace == 'HSV'):
        img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    
    return cv.imwrite(name,img)

def filtroNegativo(img,colored=False):
    if(colored):
        img[:,:,2] = 255-img[:,:,2]
    else:
        img = 255 - img

    return img


def calculaHistograma(img):
    h = [0]*256
    rows,cols = img.shape
    for y in xrange(rows):
        for x in xrange(cols):
            h[img[y][x]] += 1

    return h

def graficoHistograma(hist):
    plt.bar(np.arange(len(hist)),hist)
    plt.show(block = False)

def ajusteContraste(img,g_min,g_max):
    rows, cols = img.shape
    new_img = np.zeros(img.shape,dtype='uint8')
    hist = calculaHistograma(img) 

    for i in xrange(len(hist)-1,-1,-1):
        if hist[i] != 0:
            f_max = i
            break

    for i in xrange(0,len(hist)):
        if hist[i] != 0:
            f_min = i
            break

    delta = (g_max - g_min)/(f_max - f_min)

    print('Divisao: %s' % delta)

    for y in xrange(rows):
        for x in xrange(cols):
            new_img[y][x] = delta*(img[y][x] - f_min) + g_min

    return new_img

def histogramaNormalizado(img):
    hist = calculaHistograma(img)
    hist_n = []
    pixels = img.size

    for i in xrange(len(hist)):
        hist_n.append(hist[i]/pixels)

    return hist_n

def histogramaAcumulado(hist):
    acc = 0
    hist_acc = []
    for i in xrange(len(hist)):
        acc += hist[i]
        hist_acc.append(acc)

    return hist_acc
        
def equalizarHistogramaImagem(img,L=255):
    hist = calculaHistograma(img)
    hist_n = histogramaNormalizado(img)
    hist_acumulado = histogramaAcumulado(hist_n)
    
    s = []
    for i in xrange(len(hist_acumulado)):
        s.append(math.floor((L-1)*hist_acumulado[i]))

    return s


def equalizarImagem(img):
    hist_eq = equalizarHistogramaImagem(img)
    new_img = np.zeros(img.shape,dtype='uint8')
    rows,cols = img.shape

    for y in xrange(rows):
        for x in xrange(cols):
            new_img.itemset((y,x),hist_eq[img.item((y,x))])

    return new_img

def conv2D(img,kernel):
    rows, cols = img.shape[:2]
    krows, kcols = kernel.shape

#   kr_min = krows//2
#   kc_min = kcols//2

    new_img = np.zeros(img.shape,dtype='uint8')

    img = cv.copyMakeBorder(img,1,1,1,1,cv.BORDER_REFLECT_101)

    for r in xrange(rows):
        for c in xrange(cols):
            kernel=np.flip(kernel,axis=0)
            kernel=np.flip(kernel,axis=1)

            value = 0
            value = np.sum(img[r:r+krows,c:c+kcols] * kernel)
            new_img.itemset((r,c),value)


    return new_img


#           for kr in xrange(-kr_min,kr_min+1):
#               for kc in xrange(-kc_min,kc_min+1):
#                   y = r+kr
#                   if(y < 0):
#                       y = kr*(-1)
#                   elif(y >= rows):
#                       y = r - kr

#                   x = c+kc
#                   if(x < 0):
#                       x = kc*(-1)
#                   elif(x >= cols):
#                       x = c - kc

#                    value += img[y][x] * kernel[kr+kr_min][kc+kc_min]
#                   value += img.item(y,x) * kernel.item(kr+kr_min,kc+kc_min)

#            new_img[r][c] = value
#           new_img.itemset((r,c),value)

#    return new_img

def filtroMedia(img):
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    return conv2D(img,kernel)

def limiarizacao_simples(img,limiar):
#   new_img = np.zeros(img.shape,dtype='uint8')
#   for y in xrange(img.shape[0]):
#       for x in xrange(img.shape[1]):
#            if (img[y,x] > limiar):
#                new_img[y,x] = 255
#            else:
#                new_img[y,x] = 0

#           if(img.item(y,x) > limiar):
#               img.itemset((y,x),255)
#           else:
#               img.itemset((y,x),0)

    return ((img > limiar).astype(int)) * 255

def filtroLaplaciano(img):
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    return conv2D(img,kernel)

