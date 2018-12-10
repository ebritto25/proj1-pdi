# coding=utf-8
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
import os
import os.path as path
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
    _img.itemset(pixel,255)

    while(len(fila) > 0):
        pixel = fila.popleft()
        adicionaVizinhos(pixel,fila,visitados)
        valor_atual = img.item(pixel)
        if(abs(valor_atual - valor_semente) <= threshold):
            _img.itemset(pixel,255)

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
    bordas = limiarOtsu(img)[1]
    posErosao = filtroMinimo(bordas,3,3)
    posDilacao = filtroMaximo(posErosao,3,3)
    return posErosao.astype('uint8')

def transFechamento(img):
    bordas = deteccaoDeCanny(img,100,200)
    posDilacao = filtroMaximo(bordas,3,3)
    posErosao = filtroMinimo(posDilacao,3,3)
    return np.clip(posErosao,0,255).astype('uint8')

def extracaoCor(img):
    multi_rgb = cv.cvtColor(img,cv.COLOR_HSV2RGB).astype('uint8')

    for i in range(3):
        multi_rgb[:,:,i] = np.bitwise_and(multi_rgb[:,:,i],np.uint8(192))

    rgb = np.zeros(multi_rgb.shape,dtype='uint8')
    rgb = (multi_rgb[:,:,0] >> 2) | (multi_rgb[:,:,1] >> 4) | (multi_rgb[:,:,2] >> 6)
    rows,cols = rgb.shape

    def populaHistogramas(pixel,histoBorda,histoInterior):
        vizinhos = [(pixel[0]+1,pixel[1]),(pixel[0],pixel[1]-1),\
                    (pixel[0]-1,pixel[1]),(pixel[0],pixel[1]+1)]

        cor_centro = rgb[pixel[0],pixel[1]]

        for viz in vizinhos:
            if ((viz[0] >= 0 and viz[1] >= 0) and \
                (viz[0] < rows and viz[1] < cols)):
                
                cor_viz = rgb[viz[0],viz[1]]

                if(cor_centro != cor_viz):
                    histoBorda[cor_viz] += 1
                    return

        histoInterior[cor_centro] += 1
        return
        

    histoBorda = [0]*64
    histoInterior = [0]*64

    for y in xrange(rows):
        for x in xrange(cols):
            populaHistogramas((y,x),histoBorda,histoInterior)

    histos = histoBorda + histoInterior

    '''
    print(histoBorda)
    print(histoInterior)
    print(multi_rgb[0,0])
    '''

    return histos,cv.cvtColor(multi_rgb,cv.COLOR_RGB2HSV)

def arffWriter(filename,names,classes,relation,data):
    arffDump(filename,data,relation,classes,attributes_names=names)

def extForma(img):
    im2, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    def defineCodigoVizinhos(vizinhos):
        codigo = {(1,0)  :'0',
                  (1,-1) :'1',
                  (0,-1) :'2',
                  (-1,-1):'3',
                  (-1,0) :'4',
                  (-1,1) :'5',
                  (0,1)  :'6',
                  (1,1)  :'7'}

        primeiro = vizinhos[0][0]
        segundo = vizinhos[1][0]
        return codigo.get((segundo[0] - primeiro[0],\
                           segundo[1] - primeiro[1]))

    chain = []
    maiorContorno = max(contours,key=lambda x: len(x))

    for i in xrange(len(maiorContorno)-1):
        chain.append(defineCodigoVizinhos(maiorContorno[i:i+2]))

    chain.append(defineCodigoVizinhos([maiorContorno[-1],maiorContorno[0]]))
    
    hist = [0]*8

    for c in chain:
        hist[int(c)] += 1

    return hist

def arffDump(dst_filename,data,relation,classes,attributes_names = None):
    relation = '@RELATION %s\n' % (relation)

    types = {type(0):'numeric',type(0.0):'numeric',type(''):'string'}
    data_sample = data[0]

    if attributes_names is None:
        attributes_names = ['Attribute%s' % (i) for i in range(len(data_sample)-1)]

    attrs = ''
    for i, attr in enumerate(attributes_names):
        attr_type = types[type(data_sample[i])]
        a = '@ATTRIBUTE %s %s\n' % (attr,attr_type)
        attrs += a

    class_ = '@ATTRIBUTE class {%s}\n' % (','.join(classes))

    header = relation + attrs + class_ + '@DATA\n'  

    with open(dst_filename,'w') as file:
        file.write(header)
        for d in data:
            dados = ','.join([str(v) for v in d])
            dados += '\n'
            file.write(dados)

def createFileList(root_directory):
    #root_directory = '~/testedir/'
    root_directory = path.abspath(path.expanduser(root_directory))

    assert path.exists(root_directory) == True, 'Diretorio não existe'

    dir_list = os.listdir(root_directory)
    classes = [] 

    for cl in dir_list:
        class_path = root_directory + os.sep + cl
        if path.isdir(class_path):
            classes.append(cl)

    print('Foram encontradas {} Classes'.format(len(classes)))
    class_files = {}
    for cl in classes:
        class_path = '%s/%s' % (root_directory,cl)
        files = ['%s/%s' % (class_path, f) for f in os.listdir(class_path)]
        class_files[cl] = files

    return class_files

def loadImageBatch(path_dict):
    img_lists = []
    train_images = []
    test_images = []

    for image_class, images_paths in path_dict.items():
        qtd_images = len(images_paths)
        train_qtd = int(round(qtd_images*0.8))

        train_paths = images_paths[:train_qtd]
        test_paths = images_paths[train_qtd:]

        for img_path in train_paths:
            image = cv.imread(img_path)
            train_images.append((image,image_class,))

        for img_path in test_paths:
            image = cv.imread(img_path)
            test_images.append((image,image_class,))

    print('{} imagens de treinamento'.format(len(train_images)))
    print('{} imagens de teste'.format(len(test_images)))
    return (train_images,test_images)

def colorBatchExtraction(base_dir):
    from sets import Set
    base = createFileList(base_dir)
    train_images, test_images = loadImageBatch(base)

    train_data = []
    test_data = []
    classes = Set()
    for image, label in train_images:
        image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        features, _ = extracaoCor(image)
        features.append(label)
        classes.add(label)
        train_data.append(features)

    for image, label in test_images:
        image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        features, _ = extracaoCor(image)
        features.append(label)
        classes.add(label)
        test_data.append(features)

    print('{} Imagens Processadas'.format(len(test_data)+len(train_data)))
    print('Classes: {}'.format(classes))
    return train_data,test_data,classes


def formBatchExtraction(base_dir):
    from sets import Set
    base = createFileList(base_dir)
    train_images, test_images = loadImageBatch(base)

    train_data = []
    test_data = []
    classes = Set()
    for image, label in train_images:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        threshold,_ = limiarOtsu(image)
        image = deteccaoDeCanny(image,threshold,threshold*2)
        features = extForma(image)
        features.append(label)
        classes.add(label)
        train_data.append(features)

    for image, label in test_images:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        threshold,_ = limiarOtsu(image)
        image = deteccaoDeCanny(image,threshold,threshold*2)
        features = extForma(image)
        features.append(label)
        classes.add(label)
        test_data.append(features)

    print('{} Imagens Processadas'.format(len(test_data)+len(train_data)))
    print('Classes: {}'.format(classes))
    return train_data,test_data,classes
