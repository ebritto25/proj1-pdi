# coding=utf-8
'''
Nome: Eduardo Britto da Costa
RA: 1633368
'''
import atexit
from multiprocessing.pool import ThreadPool
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

    PROCESS_MULTI_CHANNEL = 1
    PROCESS_SINGLE_CHANNEL = 0
    
    def __init__(self,master):
        self.hasColor = None
        self.img = None
        self.workImg = None
        self.originalFile = None
        self.valorLimiar = 100
        self.tempPath = 'temp.png'
        self.colorVariable = tk.IntVar()
        self.root = master

        self.frame = tk.Frame(master)
        self.frame.pack()

        top = master.winfo_toplevel()
        self.menuBar = tk.Menu(top)
        top['menu'] = self.menuBar
        
        #MENU ARQUIVO
        self.subMenuArquivo = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Arquivo',menu=self.subMenuArquivo)
        self.subMenuArquivo.add_command(label='Abrir Colorido',command=self.openColoredImage)
        self.subMenuArquivo.add_command(label='Abrir Cinza',command=self.openGrayImage)
        self.subMenuArquivo.add_command(label='Imagem Original',command=self.resetImage)
        self.subMenuArquivo.add_command(label='Salvar Imagem',command=self.salvar)

        #MENU FILTROS
        self.subMenuFiltros = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Filtros',menu=self.subMenuFiltros)
        self.subMenuFiltros.add_command(label='Filtro Negativo',command = self.filtroNegativo)
        self.subMenuFiltros.add_command(label='Filtro Media',command = self.filtroMedia)
        self.subMenuFiltros.add_command(label='Filtro Mediana',command = self.filtroMediana)
        self.subMenuFiltros.add_command(label='Filtro Maximo',command = self.filtroMaximo)
        self.subMenuFiltros.add_command(label='Filtro Minimo',command = self.filtroMinimo)
        self.subMenuFiltros.add_command(label='Filtro Laplace',command = self.filtroLaplace)
        self.subMenuFiltros.add_command(label='Filtro Logaritmico',command=self.filtroLogaritmico)
        self.subMenuFiltros.add_command(label='Filtro Potencia',command=self.filtroPotencia)
        self.subMenuFiltros.add_command(label='Filtro Desfoque Gaussiano',command=self.filtroGaussiano)

        #MENU CONTRASTE
        self.subMenuContraste = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Contraste',menu=self.subMenuContraste)
        self.subMenuContraste.add_command(label='Equalizar Contraste',command = self.equalizar)
        self.subMenuContraste.add_command(label='Ajustar Contraste',command = self.ajustarContraste)
       
        #MENU LIMIARIZAÇÃO
        self.subMenuLimiarizacao = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Segmentação',menu=self.subMenuLimiarizacao)
        self.subMenuLimiarizacao.add_command(label='Limiarização Simples',command = self.filtroLimiarS)

        #MENU HISTOGRAMAS
        self.subMenuHistogramas = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Histrogramas',menu=self.subMenuHistogramas)
        self.subMenuHistogramas.add_command(label='Histograma Atual',command=self.exibirHistograma)
        self.subMenuHistogramas.add_command(label='Histograma Equalizado',command=self.histEqualizado)

        #SAIR
        self.menuBar.add_command(label='Sair',command=self.frame.quit)

        #CANVAS INICIAL
        self.canvas = tk.Canvas(self.frame,height=600,width=800)
        self.canvas.pack()

        #FRAME SETTINGS
        self.frameSettings = tk.Frame(self.frame)
        self.frameSettings.pack(side=tk.BOTTOM,anchor='sw')

        #SPINNER LIMIAR
        self.limiarSelectorFrame = tk.LabelFrame(self.frameSettings,text='Controle de limiar:')
        self.limiarSelectorFrame.pack(side=tk.LEFT,anchor='w')

        self.spinnerLimiar = tk.Spinbox(self.limiarSelectorFrame,from_=0,to=255,increment=5,width=4)
        spinnerValidation = self.spinnerLimiar.register(self.validateSpinner)

        self.spinnerLimiar.pack(side=tk.LEFT)
        self.spinnerLimiar.configure(state=tk.DISABLED,validatecommand=spinnerValidation,validate='focusout')
    ################FILTROS DAS IMAGENS##################

    def filtroGaussiano(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.workImg
        
        if(self.isImgColored()):
            mask = ut.filtroGaussiano(img[:,:,2])
            img[:,:,2] = np.clip(mask,0,255).astype(int)
            ut.gravarArquivo(img,colorSpace='HSV')
        else:
            mask = ut.filtroGaussiano(img)
            img = np.clip(mask,0,255).astype(int)
            ut.gravarArquivo(img)

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())
    
    def filtroMediana(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMediana(img[:,:,2],3,3)
        else:
            img = ut.filtroMediana(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())


    def filtroMinimo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMinimo(img[:,:,2],3,3)
        else:
            img = ut.filtroMinimo(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def filtroMaximo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMaximo(img[:,:,2],3,3)
        else:
            img = ut.filtroMaximo(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def filtroLogaritmico(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroLogaritmico(img[:,:,2])
        else:
            img = ut.filtroLogaritmico(img)

        self.workImg = img.copy().astype('uint8')
        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def filtroPotencia(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return
        
        dialogConfig = {'c':'Digite o valor da Constante','gamma':'Digite o valor de Gamma'}
        dataHolder = {}
        
        d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)

        self.root.wait_window(d.top)

        if(len(dataHolder)>0):
            img = self.workImg.copy()

            if(self.isImgColored()):
                img[:,:,2] = ut.filtroPotencia(img[:,:,2],dataHolder['gamma'],C=dataHolder['c'])
            else:
                img = ut.filtroPotencia(img,dataHolder['gamma'],C=dataHolder['c'])

            self.workImg = img.copy().astype('uint8')
            self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def filtroLaplace(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.workImg.copy()
        
        if(self.isImgColored()):
            mask = ut.filtroLaplaciano(img[:,:,2])
            mask = np.clip(mask,0,255).astype(int)
            img[:,:,2] = np.clip((img[:,:,2]+mask),0,255)
        else:
            mask = ut.filtroLaplaciano(img)
            mask = np.clip(mask,0,255).astype(int)
            img = np.clip((img+mask),0,255)

        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())
    
    def ajustarContraste(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return
        
        dialogConfig = {'gMin':'Valor minimo esperado (g_min)','gMax':'Valor maximo esperado (g_max)'}
        dataHolder = {}

        d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)
        
        self.root.wait_window(d.top)

        if(len(dataHolder) > 0):

            img = self.workImg.copy()

            if(self.isImgColored()):
                img[:,:,2] = ut.ajusteContraste(img[:,:,2],int(dataHolder['gMin']),int(dataHolder['gMax']))
            else:
                img = ut.ajusteContraste(img,int(dataHolder['gMin']),int(dataHolder['gMax']))

            self.workImg = img.copy().astype('uint8')

            self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def equalizar(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.equalizarImagem(img[:,:,2])
        else:
            img = ut.equalizarImagem(img)

        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())
        

    def filtroLimiarS(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return

        self.spinnerLimiar.configure(state=tk.NORMAL)

        self.valorLimiar = int(self.spinnerLimiar.get())

        img = self.originalFile.copy()

        img = ut.limiarizacao_simples(img,self.valorLimiar)
        
        self.workImg = img.copy().astype('uint8')

        self.showImageOnCanvas(filename=self.tempPath)

    def filtroMedia(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMedia(img[:,:,2])
        else:
            img = ut.filtroMedia(img)

        self.workImg = img.copy()
        
        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    def filtroNegativo(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg

        if(self.isImgColored()):
            img = cv.cvtColor(img,cv.COLOR_HSV2RGB)
            imgTuple = self.parallelProcess(3,ut.filtroNegativo,img,MainWindow.PROCESS_MULTI_CHANNEL)
            img = cv.merge(imgTuple)
            img = cv.cvtColor(img,cv.COLOR_RGB2HSV)
        else:
            img = ut.filtroNegativo(img)

        self.workImg = img

        self.showImageOnCanvas(filename=self.tempPath,colored=self.isImgColored())

    ################GERENCIAMENTO##################
    def isImgColored(self):
        return self.hasColor

    def openColoredImage(self):
        self.hasColor = True
        self.abrir()

    def openGrayImage(self):
        self.hasColor = False
        self.abrir()

    def parallelProcess(self,threadCount,method,methodArgs,processType):
        threadPool = ThreadPool(processes=threadCount)
        threads = []

        args = []
        if(processType == MainWindow.PROCESS_MULTI_CHANNEL):
            for i in xrange(threadCount):
                threads.append(threadPool.apply_async(method,args=(methodArgs[:,:,i],)))
        elif(processType == MainWindow.PROCESS_SINGLE_CHANNEL):

            for i in xrange(threadCount):
                threads.append(threadPool.apply_async(method,args=methodArgs))

        threadPool.close()
        threadPool.join()

        returnList = []

        for thread in threads:
            returnList.append(thread.get())

        return tuple(returnList)

    def histEqualizado(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.originalFile.copy()
         
        if(self.isImgColored()):
            hist = ut.equalizarHistogramaImagem(img[:,:,2])
        else:
            hist = ut.equalizarHistogramaImagem(img)


        ut.graficoHistograma(hist)


    def exibirHistograma(self):
        if(self.img is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg
        
        if(self.isImgColored()):
            ut.histogramaImagem(img[:,:,2])
        else:
            ut.histogramaImagem(img)

    def resetComponents(self):
        self.spinnerLimiar.configure(state=tk.DISABLED)

    def validateSpinner(self):
        try:
            self.valorLimiar = int(self.spinnerLimiar.get())
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
            if(self.isImgColored()):
                ut.gravarArquivo(self.workImg,filename,colorSpace='HSV')
            else:
                ut.gravarArquivo(self.workImg,filename)

    def abrir(self):
        self.showImageOnCanvas(filename=None,colored=self.isImgColored())

    def resetImage(self):
        if(self.originalFile is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return 

        self.resetComponents()
        self.workImg = self.originalFile.copy()
        self.showImageOnCanvas(filename=self.originalFile,colored=self.isImgColored())        

    def showImageOnCanvas(self,filename=None,colored=False):
        if(filename is None):   
            imgPath = ut.abrirArquivo()
            try:
                self.workImg = self.prepareImage(imgPath,color=colored)
            except:
                return

        self.canvas.destroy()

        if(colored):
            displayImg = cv.cvtColor(self.workImg,cv.COLOR_HSV2RGB)
        else:
            displayImg = cv.cvtColor(self.workImg,cv.COLOR_GRAY2RGB)


        self.img = ImageTk.PhotoImage(Image.fromarray(displayImg))
        self.canvas = tk.Canvas(self.frame,height=self.img.height(),width=self.img.width())
        self.canvas.pack()
        imgID = self.canvas.create_image(0,0,image=self.img,anchor='nw')
        self.canvas.pack()

    def prepareImage(self,filename,color=False):
        img = cv.imread(filename)

        if(color is False):
            img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        else:
            img = cv.cvtColor(img,cv.COLOR_BGR2HSV)

        self.originalFile = img.copy()
        print('forma workimg: ',img.shape)
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
else:
    print 'Inicie o projeto rodando o main.py'
