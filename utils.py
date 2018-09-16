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
            h[img.item(y,x)] += 1

    return h

def graficoHistograma(hist):
    plt.bar(np.arange(len(hist)),hist)
    plt.show(block = False)

def histogramaImagem(img):
    h = calculaHistograma(img)
    graficoHistograma(h)

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
            new_img.itemset((y,x),delta*(img.item(y,x) - f_min) + g_min)

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

    padding = krows//2

    new_img = np.zeros(img.shape)

    img = cv.copyMakeBorder(img,padding,padding,padding,padding,cv.BORDER_REFLECT_101)

    for r in xrange(rows):
        for c in xrange(cols):
            kernel=np.flip(kernel,axis=0)
            kernel=np.flip(kernel,axis=1)

            value = 0
            value = np.sum(img[r:r+krows,c:c+kcols] * kernel)
            new_img.itemset((r,c),value)


    return new_img

def filtroMedia(img):
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9
    return conv2D(img,kernel)

def limiarizacao_simples(img,limiar):
    return ((img > limiar).astype(int)) * 255

def filtroLaplaciano(img):
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    return conv2D(img,kernel)

def filtroMinimo(img,krows,kcols):
    rows, cols = img.shape[:2]

    padding = krows//2

    new_img = np.zeros(img.shape)

    img = cv.copyMakeBorder(img,padding,padding,padding,padding,cv.BORDER_REFLECT_101)

    for r in xrange(rows):
        for c in xrange(cols):
            value = img[r:r+krows,c:c+kcols].min()
            new_img.itemset((r,c),value)


    return new_img.astype('uint8')

def filtroMaximo(img,krows,kcols):
    rows, cols = img.shape[:2]

    padding = krows//2

    new_img = np.zeros(img.shape)

    img = cv.copyMakeBorder(img,padding,padding,padding,padding,cv.BORDER_REFLECT_101)

    for r in xrange(rows):
        for c in xrange(cols):
            value = img[r:r+krows,c:c+kcols].max()
            new_img.itemset((r,c),value)


    return new_img.astype('uint8')

def filtroMediana(img,krows,kcols):
    rows, cols = img.shape[:2]
    middle = (krows*kcols)//2

    padding = krows//2

    new_img = np.zeros(img.shape)

    img = cv.copyMakeBorder(img,padding,padding,padding,padding,cv.BORDER_REFLECT_101)

    for r in xrange(rows):
        for c in xrange(cols):
            pixelArray = img[r:r+krows,c:c+kcols].reshape(-1)
            pixelArray = np.sort(pixelArray)
            value = pixelArray[middle]
            new_img.itemset((r,c),value)


    return new_img.astype('uint8')
    
def filtroLogaritmico(img,L=255):
    #TODO PERGUNTAR PRO PROFESSOR
    c = L/np.log10(1+img.max())
    return c*np.log10((1+img)).astype('uint8')

def filtroPotencia(img,gamma,C=1):
    #TODO PERGUNTAR PRO PROFESSOR
    mat = np.clip(C*np.power(img,gamma),0,255)
    return np.clip(mat,0,255).astype('uint8')

def filtroGaussiano(img):
    kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    return conv2D(img,kernel)
