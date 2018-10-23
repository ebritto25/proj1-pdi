'''
Nome: Eduardo Britto da Costa
RA: 1633368
'''
from __future__ import division
import Tkinter as tk
import tkFileDialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

def abrirArquivo():
    tk.Tk().withdraw() #ocultar janela raiz
    arq = tkFileDialog.askopenfilename()
    if(arq == ''):
        raise Exception
    print(arq)
    return arq

def gravarArquivo(img,name=None,colorSpace='GRAY'):
    if(name is None):
        name = 'temp.png'

    if(colorSpace == 'HSV'):
        img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    
    return cv.imwrite(name,img)

def filtroNegativo(img):
    return 255 - img


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
    return conv2D(img,kernel).astype('uint8')

def limiarizacao_simples(img,limiar):
    return ((img > limiar).astype(int)) * 255

def filtroLaplaciano(img):
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    img = filtroGaussiano(img)
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
    c = L/np.log10(1+img.max())
    float_img = img.astype(np.float32)
    res = c*np.log10(1+float_img)
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)

    return res

def filtroPotencia(img,gamma,C=1):
    mat = C*np.power(img.astype(float),gamma)
    return np.clip(mat,0,255).astype('uint8')

def filtroGaussiano(img):
    kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    return conv2D(img,kernel)

def limiarOtsu(img):
    threshold,img= cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return (threshold,img)

def crescimentoRegioes(img,threshold,seedCoord):
    _img = np.zeros(img.shape,dtype='uint8')

    def adicionaVizinhos(pixel,fila,visitados):
        vizinhos = [(pixel[0]+1,pixel[1]),(pixel[0]+1,pixel[1]-1),(pixel[0],pixel[1]-1),(pixel[0]-1,pixel[1]-1),(pixel[0]-1,pixel[1]),(pixel[0]-1,pixel[1]+1),(pixel[0],pixel[1]+1),(pixel[0]+1,pixel[1]+1)]
        for viz in vizinhos:
            if viz not in visitados:
                if ((viz[0] >= 0 and viz[1] >= 0) and \
                    (viz[0] < img.shape[0] and viz[1] < img.shape[1])):
                    visitados[viz] = True
                    fila.append(viz)
                

    fila = deque()
    visitados = {}

    fila.append((seedCoord['y'],seedCoord['x']))
    pixel = fila.popleft()
    valor_semente = img.item(pixel)
    adicionaVizinhos(pixel,fila,visitados)
    _img.itemset(pixel,valor_semente)

    while(len(fila) > 0):
        pixel = fila.popleft()
        adicionaVizinhos(pixel,fila,visitados)
        valor_atual = img.item(pixel)
        if(abs(valor_atual - valor_semente) <= threshold):
            _img.itemset(pixel,valor_atual)

    return _img

def deteccaoDeSobel(img):
    img = filtroGaussiano(img)
    
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    g1 = conv2D(img,kernel)

    kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    g2 = conv2D(img,kernel)

    out = g1+g2
    return np.clip(out,0,255).astype('uint8')

def deteccaoDeCanny(img,tInferior,tSuperior):
   return cv.Canny(img,tInferior,tSuperior) 

def transAbertura(img):
    bordas = deteccaoDeCanny(img,100,200)
    posErosao = filtroMinimo(bordas,3,3)
    posDilacao = filtroMaximo(posErosao,3,3)
    return np.clip(posDilacao,0,255).astype('uint8')

def transFechamento(img):
    bordas = deteccaoDeCanny(img,100,200)
    posDilacao = filtroMaximo(bordas,3,3)
    posErosao = filtroMinimo(posDilacao,3,3)
    return np.clip(posErosao,0,255).astype('uint8')
