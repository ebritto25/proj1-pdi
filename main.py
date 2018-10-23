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
import tkFont
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
        self.workCanvasImg = None
        self.workImg = None
        self.originalCanvasImg = None
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
       
        #MENU SEGMENTAÇÃO
        self.subMenuLimiarizacao = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Segmentação',menu=self.subMenuLimiarizacao)
        self.subMenuLimiarizacao.add_command(label='Limiarização Manual',command = self.filtroLimiarS)
        self.subMenuLimiarizacao.add_command(label='Limiarização Otsu',command = self.filtroLimiarOtsu)
        self.subMenuLimiarizacao.add_command(label='Crescimento de regiões',command = lambda: self.filtroCrescimento(None,ready=False))

        #MENU HISTOGRAMAS
        self.subMenuHistogramas = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Histogramas',menu=self.subMenuHistogramas)
        self.subMenuHistogramas.add_command(label='Histograma Atual',command=self.exibirHistograma)
        self.subMenuHistogramas.add_command(label='Histograma Equalizado',command=self.histEqualizado)

        #MENU DETECÇÃO
        self.subMenuDeteccao = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Detecção de Bordas',menu=self.subMenuDeteccao)
        self.subMenuDeteccao.add_command(label='Detecção de Sobel',command=self.deteccaoSobel)
        self.subMenuDeteccao.add_command(label='Detecção de Canny',command=self.deteccaoCanny)

        #MENU DETECÇÃO
        self.subMenuTransformacao = tk.Menu(self.menuBar)
        self.menuBar.add_cascade(label='Transformações',menu=self.subMenuTransformacao)
        self.subMenuTransformacao.add_command(label='Abertura',command=self.transAbertura)
        self.subMenuTransformacao.add_command(label='Fechamento',command=self.transFechamento)

        #SAIR
        self.menuBar.add_command(label='Sair',command=self.frame.quit)

        #FRAME CANVAS
        self.workCanvas = None
        self.originalCanvas = None
        self.frameCanvas = tk.Frame(self.frame)
        self.frameCanvas.pack(side=tk.TOP)

        #CANVAS INICIAL
        self.workCanvas = tk.Canvas(self.frameCanvas,height=600,width=800)
        self.workCanvas.pack()

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
    ################ FILTROS E TRANSFORMAÇÕES ##################
    ############### FUNÇÕES PROJETO 2 ####################
    def transAbertura(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return

        img = self.workImg.copy()

        img = ut.transAbertura(img)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()

    def transFechamento(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return

        img = self.workImg.copy()

        img = ut.transFechamento(img)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()

    def deteccaoCanny(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return
        
        img = self.workImg.copy()

        dialogConfig = {'inf':'Digite o valor do Threshold Inferior','sup':'Digite o valor do Threshold Superior'}
        dataHolder = {}
        
        d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)

        self.root.wait_window(d.top)

        if(len(dataHolder)>0):
            img = ut.deteccaoDeCanny(self.workImg.copy(),dataHolder['inf'],dataHolder['sup'])

            self.workImg = img.copy().astype('uint8')
                
            self.updateWorkCanvas()
            
    def deteccaoSobel(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return
        
        img = self.workImg.copy()

        self.workImg = ut.deteccaoDeSobel(img).copy()

        self.updateWorkCanvas()

    def filtroCrescimento(self,event,ready = True):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return

        if (ready is False): 
            tkMessageBox.showinfo('Informação','Selecione o pixel semente')

            self.workCanvas.bind('<Button-1>',self.filtroCrescimento)
        else:
            seedCoord = {'x':event.x,'y':event.y}

            dialogConfig = {'thresh':'Digite o valor do Threshold'}
            dataHolder = {}
            
            d = ValueEntryDialog(self.root,dataHolder,**dialogConfig)

            self.root.wait_window(d.top)

            if(len(dataHolder)>0):
                threshold = dataHolder['thresh']
                img = ut.crescimentoRegioes(self.workImg.copy(),threshold,seedCoord)

                self.workImg = img.copy().astype('uint8')
                    
                self.updateWorkCanvas()

            self.workCanvas.unbind('<Button-1>')

    def filtroLimiarOtsu(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        if(self.isImgColored()):
            tkMessageBox.showerror('Erro','Abra a imagem em modo: Tons de Cinza')
            return

        _,img = ut.limiarOtsu(self.workImg)

        self.workImg = img.copy().astype('uint8')
        
        self.updateWorkCanvas()


    ############### FUNÇÕES PROJETO 1 ####################
    def filtroGaussiano(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.workImg.copy()
        
        if(self.isImgColored()):
            mask = ut.filtroGaussiano(img[:,:,2])
            img[:,:,2] = np.clip(mask,0,255).astype(int)
        else:
            mask = ut.filtroGaussiano(img)
            img = np.clip(mask,0,255).astype(int)

        self.workImg = img.copy().astype('uint8')
        self.updateWorkCanvas()
    
    def filtroMediana(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMediana(img[:,:,2],3,3)
        else:
            img = ut.filtroMediana(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()


    def filtroMinimo(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMinimo(img[:,:,2],3,3)
        else:
            img = ut.filtroMinimo(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()

    def filtroMaximo(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMaximo(img[:,:,2],3,3)
        else:
            img = ut.filtroMaximo(img,3,3)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()

    def filtroLogaritmico(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroLogaritmico(img[:,:,2])
        else:
            img = ut.filtroLogaritmico(img)

        self.workImg = img.copy().astype('uint8')
        self.updateWorkCanvas()

    def filtroPotencia(self):
        if(self.workCanvasImg is None):
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
            self.updateWorkCanvas()

    def filtroLaplace(self):
        if(self.workCanvasImg is None):
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

        self.updateWorkCanvas()
    
    def ajustarContraste(self):
        if(self.workCanvasImg is None):
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

            self.updateWorkCanvas()

    def equalizar(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.equalizarImagem(img[:,:,2])
        else:
            img = ut.equalizarImagem(img)

        self.workImg = img.copy().astype('uint8')

        self.updateWorkCanvas()
        

    def filtroLimiarS(self):
        if(self.workCanvasImg is None):
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

        self.updateWorkCanvas()

    def filtroMedia(self):
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return

        img = self.workImg.copy()

        if(self.isImgColored()):
            img[:,:,2] = ut.filtroMedia(img[:,:,2])
        else:
            img = ut.filtroMedia(img)

        self.workImg = img.copy()
        
        self.updateWorkCanvas()

    def filtroNegativo(self):
        if(self.workCanvasImg is None):
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

        self.updateWorkCanvas()

    ################ GERENCIAMENTO ##################
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
        if(self.workCanvasImg is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return


        img = self.originalFile.copy()
         
        if(self.isImgColored()):
            hist = ut.equalizarHistogramaImagem(img[:,:,2])
        else:
            hist = ut.equalizarHistogramaImagem(img)


        ut.graficoHistograma(hist)


    def exibirHistograma(self):
        if(self.workCanvasImg is None):
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
        filename = tkF.asksaveasfilename(filetypes=(('PNG','*.png'),))
        if(filename != ''):
            if(self.isImgColored()):
                ut.gravarArquivo(self.workImg,filename,colorSpace='HSV')
            else:
                ut.gravarArquivo(self.workImg,filename)

    def abrir(self):
        imgPath = ut.abrirArquivo()
        try:
            self.workImg = self.prepareImage(imgPath,color=self.isImgColored())
        except:
            return

        self.updateWorkCanvas()
        self.updateOriginalCanvas()

    def updateOriginalCanvas(self):
        if(self.originalCanvas != None):
            self.originalCanvas.destroy()

        self.originalCanvasImg = self.getPhotoImage(self.originalFile)
        self.originalCanvas = tk.Canvas(self.frameCanvas,height=self.originalCanvasImg.height(),width=self.originalCanvasImg.width())
        self.originalCanvas.pack(side=tk.LEFT,anchor='w')
        _ = self.originalCanvas.create_image(0,0,image=self.originalCanvasImg,anchor='nw')
        self.originalCanvas.pack(anchor='w')
        _ = self.originalCanvas.create_text(70,14,font=tkFont.Font(size=12),text='Imagem Original',fill='red')


    def resetImage(self):
        if(self.originalFile is None):
            tkMessageBox.showerror('Erro','Nenhuma imagem aberta')
            return 

        self.resetComponents()
        self.workImg = self.originalFile.copy()
        self.updateWorkCanvas()        

    def getPhotoImage(self, img):

        if(self.isImgColored()):
            displayImg = cv.cvtColor(img,cv.COLOR_HSV2RGB)
        else:
            displayImg = cv.cvtColor(img,cv.COLOR_GRAY2RGB)

        return ImageTk.PhotoImage(Image.fromarray(displayImg))


    def updateWorkCanvas(self):

        self.workCanvas.destroy()

        self.workCanvasImg = self.getPhotoImage(self.workImg)
        self.workCanvas = tk.Canvas(self.frameCanvas,height=self.workCanvasImg.height(),width=self.workCanvasImg.width())
        self.workCanvas.pack(side=tk.RIGHT,anchor='e')
        imgID = self.workCanvas.create_image(0,0,image=self.workCanvasImg,anchor='nw')
        self.workCanvas.pack(anchor='e')
        _ = self.workCanvas.create_text(80,14,font=tkFont.Font(size=12),text='Imagem Modificada',fill='red')


    def getCanvasCoordinates(self,eventOrigin,coord):
        coord['x'],coord['y'] = eventOrigin.x,eventOrigin.y
        print 'x: %s y: %s' % (eventOrigin.x,eventOrigin.y)

    def prepareImage(self,filename,color=False):
        img = cv.imread(filename)

        if(color is False):
            img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        else:
            img = cv.cvtColor(img,cv.COLOR_BGR2HSV)

        self.originalFile = img.copy()
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
