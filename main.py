'''
Nome: Eduardo Britto da Costa
RA: 1633368
'''
import atexit
import cv2 as cv
import Tkinter as tk
import tkMessageBox
import tkFileDialog as tkF
import utils as ut
from PIL import Image,ImageTk
import numpy as np

class ValueEntryDialog:

    def __init__(self,root,dataHolder,**kwargs):

        self.top = tk.Toplevel(root)
        self.frame = tk.Frame(self.top)
        self.frame.pack()

        
        self.entryComponents = []
        self.entryVariables = []
        for key in kwargs:
            self.Label = tk.Label(self.frame,text=kwargs[key])
            self.Label.pack()

            self.entryVariables.append(tk.StringVar())
            entryVar = self.entryVariables[len(self.entryVariables)-1]

            self.entryComponents.append(tk.Entry(self.frame,textvariable=entryVar))
            entry = self.entryComponents[len(self.entryComponents)-1]
            entry.pack()

            validateEntry = entry.register(self.entryValidation)
            entry.configure(validatecommand=validateEntry,validate='focus')

        self.buttonFrame = tk.Frame(self.frame)
        self.buttonFrame.pack(side=tk.BOTTOM)

        self.okButton = tk.Button(self.buttonFrame,text='OK',command=lambda:self.saveData(dataHolder,**kwargs))
        self.okButton.pack(side=tk.RIGHT,anchor='e')

        self.cancelButton = tk.Button(self.buttonFrame,text='Cancelar',command=self.sair)
        self.cancelButton.pack(side=tk.LEFT,anchor='w')

    def sair(self):
        self.top.destroy()

    def saveData(self,dataHolder,**kwargs):
        for i,entry in enumerate(self.entryComponents):
            entryVar = self.entryVariables[i]
            data = float(entryVar.get().replace(',','.'))
            dataHolder[kwargs.keys()[i]] = data

        self.sair()

    def entryValidation(self):
        for i,entry in enumerate(self.entryComponents):
            try:
                entryVar = self.entryVariables[i]
                data = float(entryVar.get().replace(',','.'))
                if(data < 0):
                    entryVar.set('0')
                    return False
                elif(data > 255):
                    entryVar.set('255')
                    return False
            except Exception as ex:
                entryVar.set('0')
                return False

        return True


class MainWindow:
    
    def __init__(self,master):
        self.img = None
        self.originalFile = None
        self.valorLimiar = 100
        self.tempPath = 'temp.png'
        self.colorVariable = tk.IntVar()
        self.root = master

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

        #BOTAO LAPLACE
        self.btnLaplace = tk.Button(self.frameFiltros,text='Filtro Laplace',command=self.filtroLaplace)
        self.btnLaplace.pack(side=tk.LEFT)






        #BOTAO AJUSTE CONTRASTE
        self.btnEqualizar = tk.Button(self.frameFiltros,text='Ajustar Contraste',command=self.ajustarContraste)
        self.btnEqualizar.pack(side=tk.LEFT)

        #BOTAO EQUALIZAR
        self.btnEqualizar = tk.Button(self.frameFiltros,text='Equalizar Imagem',command=self.equalizar)
        self.btnEqualizar.pack(side=tk.LEFT)

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

        self.spinnerLimiar = tk.Spinbox(self.limiarSelectorFrame,from_=50,to=255,increment=5,width=4)
        spinnerValidation = self.spinnerLimiar.register(self.validateSpinner)

        self.spinnerLimiar.pack(side=tk.LEFT)
        self.spinnerLimiar.configure(state=tk.DISABLED,validatecommand=spinnerValidation,validate='focusout')

        #BOTAO MEDIANA
        self.btnMediana = tk.Button(self.frameSettings,text='Mediana',command=self.filtroMediana)
        self.btnMediana.pack(side=tk.LEFT)

        #BOTAO MAXIMO
        self.btnMaximo = tk.Button(self.frameSettings,text='Maximo',command=self.filtroMaximo)
        self.btnMaximo.pack(side=tk.LEFT)

        #BOTAO MINIMO
        self.btnMinimo = tk.Button(self.frameSettings,text='Minimo',command=self.filtroMinimo)
        self.btnMinimo.pack(side=tk.LEFT)

        #BOTAO LOGARITMICO
        self.btnLogaritmico = tk.Button(self.frameSettings,text='Logaritmico',command=self.filtroLogaritmico)
        self.btnLogaritmico.pack(side=tk.LEFT)

        #BOTAO POTENCIA
        self.btnPotencia = tk.Button(self.frameSettings,text='Filtro Potencia',command=self.filtroPotencia)
        self.btnPotencia.pack(side=tk.LEFT)

        #BOTAO HISTOGRAMA
        self.btnHistograma = tk.Button(self.frameSettings,text='Histograma Atual',command=self.exibirHistograma)
        self.btnHistograma.pack(side=tk.LEFT)

        #BOTAO HISTOGRAMA EQUALIZADO
        self.btnHistEqualizado = tk.Button(self.frameSettings,text='Histograma Equalizado',command=self.histEqualizado)
        self.btnHistEqualizado.pack(side=tk.LEFT)

        #BOTAO GAUSSIANO
        self.btnGaussiano = tk.Button(self.frameSettings,text='Desfoque Gaussiano',command=self.filtroGaussiano)
        self.btnGaussiano.pack(side=tk.LEFT)

    ################FILTROS DAS IMAGENS##################

    def filtroGaussiano(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.tempPath,color=hasColor)
        
        if(hasColor):
            mask = ut.filtroGaussiano(img[:,:,2])
            img[:,:,2] = np.clip(mask,0,255).astype(int)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            mask = ut.filtroGaussiano(img)
            img = np.clip(mask,0,255).astype(int)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)
    
    def filtroMediana(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())
        img = self.prepareImage(self.tempPath,color=hasColor)

        if(hasColor):
            img[:,:,2] = ut.filtroMediana(img[:,:,2],3,3)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.filtroMediana(img,3,3)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)


    def filtroMinimo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())
        img = self.prepareImage(self.tempPath,color=hasColor)

        if(hasColor):
            img[:,:,2] = ut.filtroMinimo(img[:,:,2],3,3)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.filtroMinimo(img,3,3)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def filtroMaximo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())
        img = self.prepareImage(self.tempPath,color=hasColor)

        if(hasColor):
            img[:,:,2] = ut.filtroMaximo(img[:,:,2],3,3)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.filtroMaximo(img,3,3)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def filtroLogaritmico(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())
        img = self.prepareImage(self.tempPath,color=hasColor)

        if(hasColor):
            img[:,:,2] = ut.filtroLogaritmico(img[:,:,2])
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.filtroLogaritmico(img)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def filtroPotencia(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return
        
        dialogConfig = {'c':'Digite o valor da Constante','gamma':'Digite o valor de Gamma'}
        dataHolder = {}
        
        d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)

        self.root.wait_window(d.top)

        if(len(dataHolder)>0):
            hasColor = bool(self.colorVariable.get())
            img = self.prepareImage(self.tempPath,color=hasColor)

            if(hasColor):
                img[:,:,2] = ut.filtroPotencia(img[:,:,2],dataHolder['gamma'],C=dataHolder['c'])
                ut.gravarArquivo(img,colorSpace='HSV')
            else:
                img = ut.filtroPotencia(img,dataHolder['gamma'],C=dataHolder['c'])
                ut.gravarArquivo(img)

            self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def filtroLaplace(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.tempPath,color=hasColor)
        
        if(hasColor):
            mask = ut.filtroLaplaciano(img[:,:,2])
            mask = np.clip(mask,0,255).astype(int)
            img[:,:,2] = np.clip((img[:,:,2]+mask),0,255)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            mask = ut.filtroLaplaciano(img)
            mask = np.clip(mask,0,255).astype(int)
            img = np.clip((img+mask),0,255)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)
    
    def ajustarContraste(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return
        
        dialogConfig = {'gMin':'Valor minimo esperado (g_min)','gMax':'Valor maximo esperado (g_max)'}
        dataHolder = {}

        d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)
        
        self.root.wait_window(d.top)

        if(len(dataHolder) > 0):
            hasColor = bool(self.colorVariable.get())

            img = self.prepareImage(self.tempPath,color=hasColor)
            if(hasColor):
                img[:,:,2] = ut.ajusteContraste(img[:,:,2],int(dataHolder['gMin']),int(dataHolder['gMax']))
                ut.gravarArquivo(img,colorSpace='HSV')
            else:
                img = ut.ajusteContraste(img,int(dataHolder['gMin']),int(dataHolder['gMax']))
                ut.gravarArquivo(img)

            self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

    def equalizar(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.tempPath,color=hasColor)
        if(hasColor):
            img[:,:,2] = ut.equalizarImagem(img[:,:,2])
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.equalizarImagem(img)
            ut.gravarArquivo(img)


        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)
        

    def filtroLimiarS(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        self.setRadioState(tk.DISABLED)
        self.spinnerLimiar.configure(state=tk.NORMAL)

        self.valorLimiar = int(self.spinnerLimiar.get())

        img = self.prepareImage(self.tempPath)
        new_img = ut.limiarizacao_simples(img,self.valorLimiar)
        ut.gravarArquivo(new_img)
        self.showImageOnCanvas(filename=self.tempPath)

    def filtroMedia(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())
        
        img = self.prepareImage(self.tempPath,color=hasColor)

        if(hasColor):
            img[:,:,2] = ut.filtroMedia(img[:,:,2])
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            img = ut.filtroMedia(img)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=hasColor)

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

    ################GERENCIAMENTO DE JANELAS##################

    def histEqualizado(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.originalFile,color=hasColor)
         
        if(hasColor):
            hist = ut.equalizarHistogramaImagem(img[:,:,2])
        else:
            hist = ut.equalizarHistogramaImagem(img)


        ut.graficoHistograma(hist)


    def exibirHistograma(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        hasColor = bool(self.colorVariable.get())

        img = self.prepareImage(self.tempPath,color=hasColor)
        
        if(hasColor):
            ut.histogramaImagem(img[:,:,2])
        else:
            ut.histogramaImagem(img)


    def setRadioState(self,rState):
        self.rBtnPB.configure(state=rState)
        self.rBtnColorido.configure(state=rState)

    def resetComponents(self):
        self.setRadioState(tk.NORMAL)
        self.colorVariable.set(0)
        self.spinnerLimiar.configure(state=tk.DISABLED)

    def validateSpinner(self):
        try:
            self.valorLimiar = int(self.spinnerLimiar.get())
            self.prepareImage(filename=self.originalFile)
            self.filtroLimiarS()
            return True
        except Exception as e:
            print e
            tkMessageBox.showerror('Erro','Valor invalido')
            return False

    def salvar(self):
        from shutil import copyfile
        filename = tkF.asksaveasfilename(filetypes=('PNG {*.png}'))
        if(filename != ''):
            copyfile('temp.png',filename)

    def abrir(self):
        self.showImageOnCanvas(filename=None,colored=True)

    def resetImage(self):
        if(self.originalFile is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return 
        

        self.resetComponents()
        self.showImageOnCanvas(filename=self.originalFile,colored=True)        

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
    window = MainWindow(root)

    root.mainloop()

    def exit_handler():
        from os import remove
        from os.path import isfile
        if(isfile('temp.png')):
            print 'Removendo Temporarios'
            remove('temp.png')

    atexit.register(exit_handler)
