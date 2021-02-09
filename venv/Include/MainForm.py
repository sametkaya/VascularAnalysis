import sys
import os

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QFile, QFileInfo
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtGui import QPixmap
import Filters
import VAImage
import UiMessage
from Loging import *
from MainForm_Ui import Ui_MainWindow
from VAT_QGraphicsView import VAT_QGraphicsView
from ImageViewerForm import ImageViewerForm
class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):


        super(MainForm, self).__init__()
        self.setupUi(self)



        # # QWidget.__init__(self)
        # QMainWindow.__init__(self)
        # ###load designer
        # designer_file = QFile("MainForm.ui")
        # designer_file.open(QFile.ReadOnly)
        # loader = QUiLoader()
        # self.ui = loader.load(designer_file, self)
        # self.setupUi(self)
        # designer_file.close()
        # ###

        self.processingImageList = {}
        self.rawImageList = {}
        self.initilizeComponent()
        self.logger = Logger(self.lstw_loging)
        self.windows = []
        self.logger.log(LogTypes.success,"Program has been opened successfully")
        self.logger.log(LogTypes.warning, "Form başarıyla yüklendi")
        self.logger.log(LogTypes.error, "Form başarıyla yüklendi ")
        self.logger.log(LogTypes.none, "Form başarıyla yüklendi")
        self.logger.log(LogTypes.running, "Form başarıyla yüklendi")

    def initilizeComponent(self):

        self.grapvw_sceneImage=VAT_QGraphicsView(self.wgt_sceneImage)
        self.grapvw_sceneImage.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.wgt_sceneImage.sizePolicy().hasHeightForWidth())
        self.grapvw_sceneImage.setSizePolicy(sizePolicy)
        self.grapvw_sceneImage.setMinimumSize(QtCore.QSize(650, 400))
        self.grapvw_sceneImage.setInteractive(True)
        self.lyt_sceneImage.addWidget(self.grapvw_sceneImage)
        self.grapvw_sceneImage.toggleDragMode()


        #beginLoad
        self.btn_loadImages.clicked.connect(self.btn_loadImages_clicked)
        self.lstw_imagesRawList.itemActivated.connect(self.lstw_imagesRawList_itemActivated)
        self.sldr_images.valueChanged.connect(self.sldr_images_valueChanged)
        self.btn_viewRawImage.clicked.connect(self.btn_viewRawImage_clicked)
        #endLoad
        #beginChannels

        #endChannels
        #beginFilter
        self.btn_filterRgbToGray.clicked.connect(self.btn_filterRgbToGray_clicked)
        #self.sldr_rawImageOpacity.valueChanged.connect(self.sldr_rawImageOpacity_valueChanged)
        self.sldr_filteredImageOpacity.valueChanged.connect(self.sldr_filteredImageOpacity_valueChanged)
        self.btn_filterHistogramGlobalEqualization.clicked.connect(self.btn_filterHistogramGlobalEqualization_clicked)
        #self.btn_filterTest.clicked.connect(self.btn_filterTest_clicked)
        #self.btn_filterHistogramLocalEqualization.clicked.connect(self.btn_filterHistogramLocalEqualization_clicked)
        self.btn_filterSave.clicked.connect(self.btn_filterSave_clicked)
        self.btn_filterSaveAs.clicked.connect(self.btn_filterSaveAs_clicked)
        self.btn_filterOtsuBinarization.clicked.connect(self.btn_filterOtsuBinarization_clicked)
        self.btn_filterAdaptiveGaussianTreshold.clicked.connect(self.btn_filterAdaptiveGaussianTreshold_clicked)
        #endFilter
        # self.btn_applyFilters.clicked.connect(self.btn_applyFilters_clicked)

        self.sldr_images.setTracking(False)
        self.sldr_filteredImageOpacity.setEnabled(False)
        self.setWindowTitle("Vascular Graph Tool")

    def btn_viewRawImage_clicked(self):
        if bool(self.rawImageList):
            imageViewForm = ImageViewerForm(self.rawImageList,self.processingImageList,self.sldr_images.value(),self)
            imageViewForm.show()
            imageViewForm.sldr_images_valueChanged(self.sldr_images.value())

    #beginImagesEvent
    def btn_loadImages_clicked(self):
        self.logger.log(LogTypes.running, "Images are loading...")
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setWindowTitle("Load Images")
        dlg.setNameFilter("Image Files (*.jpg *.png *.tif *.tiff)")
        imageFiles = []
        self.rawImageList = {}
        self.processingImageList = {}
        if dlg.exec_():
            imageFiles = dlg.selectedFiles()
            self.lstw_imagesRawList.clear()
            self.lstw_imagesProcessingList.clear()
            self.sldr_images.setTracking(False)
            if len(imageFiles) <= 0:
                self.logger.log(LogTypes.error, "Images couldn't be found!")
                return
            #find VAI_ folder to create new folder
            path=QFileInfo(imageFiles[0]).absolutePath()
            folderCount=VAImage.findCountOfDirectoryStartWithName(path,"VAI_ProcessingImages")
            processingFolderPath=VAImage.createFolder(path,"VAI_ProcessingImages_"+str(folderCount))
            for item in imageFiles:
                imageName=QFileInfo(item).fileName()
                self.rawImageList[imageName] = QFileInfo(item).absoluteFilePath()
                self.processingImageList[imageName]=os.path.join(processingFolderPath, imageName)
                VAImage.copyImage(self.rawImageList[imageName],self.processingImageList[imageName])
                self.lstw_imagesRawList.addItem(imageName)
                self.lstw_imagesProcessingList.addItem(imageName)
            if len(imageFiles) != 0:
                imageName = QFileInfo(imageFiles[0]).fileName()
                self.txtle_imagesRawPath.setText(QFileInfo(self.rawImageList[imageName]).absolutePath())
                self.txtle_imagesProcessingPath.setText(QFileInfo(self.processingImageList[imageName]).absolutePath())
                self.lstw_imagesRawList.setCurrentRow(0)
                self.sldr_images.setMaximum(len(imageFiles) - 1)
                self.sldr_images.setTracking(True)
                self.sldr_images_valueChanged(0)
        self.logger.log(LogTypes.success, "Images have been loaded.")

    def lstw_imagesRawList_itemActivated(self, item):
        if bool(self.processingImageList):
            self.sldr_filteredImageOpacity.setEnabled(True)
        else:
            self.sldr_filteredImageOpacity.setEnabled(False)
        painter = QPainter()
        imagePath = self.rawImageList.get(item.text())
        image = QImage(imagePath)
        #painter.setOpacity(float(self.sldr_rawImageOpacity.value()) / 100)
        painter.begin(image)
        if item.text() in self.processingImageList:
            imagePath2= self.processingImageList.get(item.text())
            image2 = QImage(imagePath2)
            painter.setOpacity(float(self.sldr_filteredImageOpacity.value())/100)
            painter.drawImage(0, 0, image2)
        painter.end()
        self.grapvw_sceneImage.setPhoto(image)
        # pix=QPixmap.fromImage(image)
        # pixmapItem = QGraphicsPixmapItem(pix)
        # scene = QtWidgets.QGraphicsScene()
        # scene.addItem(pixmapItem)
        # self.grapvw_sceneImage.setScene(scene)
        self.sldr_images.setValue(self.lstw_imagesRawList.indexFromItem(item).row())
    # endImagesEvent
    def sldr_images_valueChanged(self, value):
        index = QModelIndex(self.lstw_imagesProcessingList.model().index(value, 0))
        item = self.lstw_imagesProcessingList.itemFromIndex(index)
        self.lstw_imagesProcessingList.setCurrentItem(item)
        index = QModelIndex(self.lstw_imagesRawList.model().index(value, 0))
        item = self.lstw_imagesRawList.itemFromIndex(index)
        self.lstw_imagesRawList.setCurrentItem(item)
        self.lstw_imagesRawList_itemActivated(item)


    # def sldr_rawImageOpacity_valueChanged(self, value):
    #     index = QModelIndex(self.lstw_imagesRawList.model().index(self.sldr_images.value(), 0))
    #     item = self.lstw_imagesRawList.itemFromIndex(index)
    #     #self.lstw_imagesRawList.setCurrentItem(item)
    #     self.lstw_imagesRawList_itemActivated(item)

    def sldr_filteredImageOpacity_valueChanged(self, value):
        index = QModelIndex(self.lstw_imagesRawList.model().index(self.sldr_images.value(), 0))
        item = self.lstw_imagesRawList.itemFromIndex(index)
        #self.lstw_imagesRawList.setCurrentItem(item)

        self.lstw_imagesRawList_itemActivated(item)




    def btn_filterTest_clicked(self):
        Filters.filterAll(Filters.speededUpAdaptiveContrastEnhancement, self.processingImageList, 21, 21)
        self.sldr_images_valueChanged(self.sldr_images.value())
    def btn_filterRgbToGray_clicked(self):
        Filters.filterAll(Filters.rgbToGray,self.processingImageList)
        self.sldr_images_valueChanged(self.sldr_images.value())

    def btn_filterHistogramGlobalEqualization_clicked(self):
        if self.rdio_filterHistogramLocal.isChecked():
            Filters.filterAll(Filters.histogramLocalEqualize, self.processingImageList)
            self.sldr_images_valueChanged(self.sldr_images.value())
        else:
            Filters.filterAll(Filters.histogramGlobalEqualize, self.processingImageList)
            self.sldr_images_valueChanged(self.sldr_images.value())

    def btn_filterAdaptiveGaussianTreshold_clicked(self):
        size=self.spbox_filterAdaptiveGaussianTreshold.value()
        Filters.filterAll(Filters.adaptiveGaussianTreshold, self.processingImageList,size)
        self.sldr_images_valueChanged(self.sldr_images.value())

    def btn_filterOtsuBinarization_clicked(self):
        size = self.spbox_filterOtsuBinarization.value()
        Filters.filterAll(Filters.otsuBinarization, self.processingImageList, size)
        self.sldr_images_valueChanged(self.sldr_images.value())

    def btn_filterSave_clicked(self):
        rvalue = UiMessage.openOkCancelWarningMessageBox("Warning: Save Over Raw Images","Filtered images will be saved over raw images")
        if rvalue != QMessageBox.Ok:
            return
        if bool(self.processingImageList):
            fileName = list(self.processingImageList.keys())[0]
            filePath = list(self.processingImageList.values())[0]
            filePath = filePath.replace(fileName, "")
            #self.txtle_imagesRawPath.setText(filePath)
            for fImageName, fImagePath in self.processingImageList.items():
                rImagePath=self.rawImageList.get(fImageName)
                VAImage.copyImage(fImagePath, rImagePath)
            self.processingImageList = {}
            self.sldr_images_valueChanged(self.sldr_images.value())
            VAImage.deleteFolder(filePath)

    def btn_filterSaveAs_clicked(self):
        dlg = QFileDialog()
        dlg.setWindowTitle("Save As")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setOption(QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            folderPath = dlg.getSaveFileUrl()
            #dlg.selectedFiles()[0]
            fileName=dlg.getSaveFileName()
            if bool(self.processingImageList):
                fileName = list(self.processingImageList.keys())[0]
                filePath = list(self.processingImageList.values())[0]
                filePath = filePath.replace(fileName, "")
                self.txtle_imagesRawPath.setText(folderPath)
                for fImageName, fImagePath in self.processingImageList.items():
                    #rImageName = self.rawImageList.get(fImageName)
                    dImagePath = os.path.join(folderPath, fImageName)
                    VAImage.copyImage(fImagePath, dImagePath)
                    self.rawImageList[fImageName]=dImagePath
                self.processingImageList = {}
                self.sldr_images_valueChanged(self.sldr_images.value())
                VAImage.deleteFolder(filePath)


        return