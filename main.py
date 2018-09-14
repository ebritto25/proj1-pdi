#!/usr/bin/python

import atexit
import cv2 as cv
import Tkinter as tk
import tkMessageBox
import utils as ut
from PIL import Image,ImageTk
import numpy as np

class mainWindow:
    
    def __init__(self,master):
        self.img = None
        self.originalFile = None
        self.valorLimiar = 100
        self.tempPath = 'temp.png'
        self.colorVariable = tk.IntVar()

        self.frame = tk.Frame(master)
        self.frame.pack()
        
        self.frameFiltros = tk.Frame(self.frame)
        self.frameFiltros.pack(side=tk.TOP,anchor='nw')

        self.btnOpen = tk.Button(self.frameFiltros,text='Abrir Imagem',command=self.abrir)
        self.btnOpen.pack(side=tk.LEFT)

        #BOTAO NEGATIVO
        self.btnNegativo = tk.Button(self.frameFiltros,text='Negativo',command=self.filtroNegativo)
        self.btnNegativo.pack(side=tk.LEFT)

        #BOTAO MEDIA
        self.btnMedia = tk.Button(self.frameFiltros,text='Filtro Media(Blur)',command=self.filtroMedia)
        self.btnMedia.pack(side=tk.LEFT)

        #BOTAO LIMIARIZACAO
        self.btnLimiarS = tk.Button(self.frameFiltros,text='Filtro Limiar Simples',command=self.filtroLimiarS)
        self.btnLimiarS.pack(side=tk.LEFT)



        #BOTAO SALVAR
        self.btnSair = tk.Button(self.frameFiltros,text='SALVAR',command=self.salvar)
        self.btnSair.pack(side=tk.LEFT)

        #BOTAO IMAGEM ORIGINAL
        self.btnOriginal = tk.Button(self.frameFiltros,text='Imagem Original',command=self.resetImage)
        self.btnOriginal.pack(side=tk.LEFT)

        #BOTAO SAIR
        self.btnSair = tk.Button(self.frameFiltros,text='SAIR',command=self.frame.quit)
        self.btnSair.pack(side=tk.LEFT)
        
        self.canvas = tk.Canvas(self.frame,height=600,width=800)
        self.canvas.pack()

        #FRAME SETTINGS
        self.frameSettings = tk.Frame(self.frame)
        self.frameSettings.pack(side=tk.BOTTOM,anchor='sw')

        #CONTROLE DE COR
        self.colorSelectorFrame = tk.LabelFrame(self.frameSettings,text='Aplicar na foto:')
        self.colorSelectorFrame.pack(side=tk.LEFT,anchor='w')

        self.rBtnPB = tk.Radiobutton(self.colorSelectorFrame,text='Tons de Cinza',variable=self.colorVariable,value=0)
        self.rBtnPB.pack(side=tk.LEFT)

        self.rBtnColorido = tk.Radiobutton(self.colorSelectorFrame,text='Colorida',variable=self.colorVariable,value=1)
        self.rBtnColorido.pack(side=tk.LEFT)

        #SPINNER LIMIAR
        self.limiarSelectorFrame = tk.LabelFrame(self.frameSettings,text='Controle de limiar:')
        self.limiarSelectorFrame.pack(side=tk.LEFT,anchor='w')

        self.spinnerLimiar = tk.Spinbox(self.limiarSelectorFrame,from_=100,to=255,increment=5)
        self.spinnerLimiar.pack(side=tk.LEFT)

    def salvar(self):
        from shutil import copyfile
        copyfile('temp.png','salvo.png')

    def filtroLimiarS(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        self.valorLimiar = int(self.spinnerLimiar.get())
        img = self.prepareImage(self.tempPath)
        new_img = ut.limiarizacao_simples(img,self.valorLimiar)
        ut.gravarArquivo(new_img)
        self.showImageOnCanvas(filename=self.tempPath)

    def filtroMedia(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.prepareImage(self.tempPath)
        new_img = ut.filtroMedia(img)
        ut.gravarArquivo(new_img)
        self.showImageOnCanvas(filename=self.tempPath)

    def filtroNegativo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.tempPath,color=hasColor)

        processed_img = ut.filtroNegativo(img,colored=hasColor)

        if(hasColor):
            ut.gravarArquivo(processed_img,colorSpace='HSV')
        else:
            ut.gravarArquivo(processed_img,colorSpace='GRAY')

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def abrir(self):
        self.showImageOnCanvas(filename=None,colored=True)

    def resetImage(self):
        if(self.originalFile is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return 

        self.showImageOnCanvas(filename=self.originalFile,colored=True)        
        self.colorVariable.set(0)

    def showImageOnCanvas(self,filename=None,colored=False):
        if(filename is None):   
            imgPath = ut.abrirArquivo()
            self.originalFile = imgPath
        else:
            imgPath = filename

        self.prepareImage(imgPath,color=colored)
        self.canvas.destroy()

        self.img = ImageTk.PhotoImage(Image.open('temp.png'))
        self.canvas = tk.Canvas(self.frame,height=self.img.height(),width=self.img.width())
        self.canvas.pack()
        imgID = self.canvas.create_image(0,0,image=self.img,anchor='nw')
        self.canvas.pack()

    def prepareImage(self,filename,color=False):
        img = cv.imread(filename)

        if(color is False):
            img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ut.gravarArquivo(img)
        else:
            img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
            ut.gravarArquivo(img,colorSpace='HSV')

        return img



if (__name__ == '__main__'):
    root = tk.Tk()
    root.title('Projeto Processamento de Imagens')
    window = mainWindow(root)

    root.mainloop()

    def exit_handler():
        from os import remove
        from os.path import isfile
        if(isfile('temp.png')):
            print 'Removendo Temporarios'
            remove('temp.png')

    atexit.register(exit_handler)
