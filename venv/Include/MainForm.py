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

class MainForm(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        designer_file = QFile("MainForm.ui")
        designer_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.ui = loader.load(designer_file, self)
        designer_file.close()

        self.filteredImageList = {}
        self.rawImageList = {}
        self.imageFiles = []
        self.initilizeComponent()

    def initilizeComponent(self):
        self.ui.btn_loadImages.clicked.connect(self.btn_loadImages_clicked)
        self.ui.lstw_imagesList.itemActivated.connect(self.lstw_imagesList_itemActivated)
        self.ui.sldr_images.valueChanged.connect(self.sldr_images_valueChanged)
        self.ui.btn_filterRgbToGray.clicked.connect(self.btn_filterRgbToGray_clicked)
        #self.ui.sldr_rawImageOpacity.valueChanged.connect(self.sldr_rawImageOpacity_valueChanged)
        self.ui.sldr_filteredImageOpacity.valueChanged.connect(self.sldr_filteredImageOpacity_valueChanged)
        self.ui.btn_filterHistogramGlobalEqualization.clicked.connect(self.btn_filterHistogramGlobalEqualization_clicked)
        self.ui.btn_filterTest.clicked.connect(self.btn_filterTest_clicked)
        #self.ui.btn_filterHistogramLocalEqualization.clicked.connect(self.btn_filterHistogramLocalEqualization_clicked)
        self.ui.btn_filterSave.clicked.connect(self.btn_filterSave_clicked)
        self.ui.btn_filterSaveAs.clicked.connect(self.btn_filterSaveAs_clicked)
        self.ui.btn_filterOtsuBinarization.clicked.connect(self.btn_filterOtsuBinarization_clicked)
        self.ui.btn_filterAdaptiveGaussianTreshold.clicked.connect(self.btn_filterAdaptiveGaussianTreshold_clicked)
        # self.ui.btn_applyFilters.clicked.connect(self.btn_applyFilters_clicked)

        self.ui.sldr_images.setTracking(False)
        self.setWindowTitle("Vascular Graph Tool")



    def btn_loadImages_clicked(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setWindowTitle("Load Images")
        dlg.setNameFilter("Image Files (*.jpg *.png *.tif *.tiff)")
        self.imageFiles = []
        self.rawImageList = {}
        self.filteredImageList = {}
        if dlg.exec_():
            self.imageFiles = dlg.selectedFiles()
            self.ui.lstw_imagesList.clear()
            self.ui.sldr_images.setTracking(False)
            for item in self.imageFiles:
                imageName=QFileInfo(item).fileName()
                self.ui.lstw_imagesList.addItem(imageName)
                self.rawImageList[imageName] = QFileInfo(item).absoluteFilePath()
            if len(self.imageFiles) != 0:
                self.ui.txtle_imagesPath.setText(QFileInfo(self.imageFiles[0]).absolutePath())
                self.ui.lstw_imagesList.setCurrentRow(0)
                self.ui.sldr_images.setMaximum(len(self.imageFiles) - 1)
                self.ui.sldr_images.setTracking(True)
                self.sldr_images_valueChanged(0)

    def sldr_images_valueChanged(self, value):
        index = QModelIndex(self.ui.lstw_imagesList.model().index(value, 0))
        item = self.ui.lstw_imagesList.itemFromIndex(index)
        self.ui.lstw_imagesList.setCurrentItem(item)
        self.lstw_imagesList_itemActivated(item)

    # def sldr_rawImageOpacity_valueChanged(self, value):
    #     index = QModelIndex(self.ui.lstw_imagesList.model().index(self.ui.sldr_images.value(), 0))
    #     item = self.ui.lstw_imagesList.itemFromIndex(index)
    #     #self.ui.lstw_imagesList.setCurrentItem(item)
    #     self.lstw_imagesList_itemActivated(item)

    def sldr_filteredImageOpacity_valueChanged(self, value):
        index = QModelIndex(self.ui.lstw_imagesList.model().index(self.ui.sldr_images.value(), 0))
        item = self.ui.lstw_imagesList.itemFromIndex(index)
        #self.ui.lstw_imagesList.setCurrentItem(item)
        self.lstw_imagesList_itemActivated(item)

    def lstw_imagesList_itemActivated(self, item):
        if bool(self.filteredImageList):
            self.ui.sldr_filteredImageOpacity.setEnabled(True)
        else:
            self.ui.sldr_filteredImageOpacity.setEnabled(False)
        painter = QPainter()
        imagePath = self.rawImageList.get(item.text())
        image = QImage(imagePath)
        #painter.setOpacity(float(self.ui.sldr_rawImageOpacity.value()) / 100)
        painter.begin(image)
        if item.text() in self.filteredImageList:
            imagePath2= self.filteredImageList.get(item.text())
            image2 = QImage(imagePath2)
            painter.setOpacity(float(self.ui.sldr_filteredImageOpacity.value())/100)
            painter.drawImage(0, 0, image2)
        painter.end()
        pix=QPixmap.fromImage(image)
        pixmapItem = QGraphicsPixmapItem(pix)
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(pixmapItem)
        self.ui.grapvw_rawImages.setScene(scene)
        self.ui.sldr_images.setValue(self.ui.lstw_imagesList.indexFromItem(item).row())

    def btn_filterTest_clicked(self):
        self.filteredImageList = Filters.filterAll(Filters.speededUpAdaptiveContrastEnhancement, self.rawImageList, 21, 21)
        self.sldr_images_valueChanged(self.ui.sldr_images.value())
    def btn_filterRgbToGray_clicked(self):
        self.filteredImageList = Filters.filterAll(Filters.rgbToGray,self.rawImageList)
        self.sldr_images_valueChanged(self.ui.sldr_images.value())

    def btn_filterHistogramGlobalEqualization_clicked(self):
        if self.ui.rdio_filterHistogramLocal.isChecked():
            self.filteredImageList = Filters.filterAll(Filters.histogramLocalEqualize, self.rawImageList)
            self.sldr_images_valueChanged(self.ui.sldr_images.value())
        else:
            self.filteredImageList = Filters.filterAll(Filters.histogramGlobalEqualize, self.rawImageList)
            self.sldr_images_valueChanged(self.ui.sldr_images.value())

    def btn_filterAdaptiveGaussianTreshold_clicked(self):
        size=self.ui.spbox_filterAdaptiveGaussianTreshold.value()
        self.filteredImageList = Filters.filterAll(Filters.adaptiveGaussianTreshold, self.rawImageList,size)
        self.sldr_images_valueChanged(self.ui.sldr_images.value())

    def btn_filterOtsuBinarization_clicked(self):
        size = self.ui.spbox_filterOtsuBinarization.value()
        self.filteredImageList = Filters.filterAll(Filters.otsuBinarization, self.rawImageList, size)
        self.sldr_images_valueChanged(self.ui.sldr_images.value())

    def btn_filterSave_clicked(self):
        rvalue = UiMessage.openOkCancelWarningMessageBox("Warning: Save Over Raw Images","Filtered images will be saved over raw images")
        if rvalue != QMessageBox.Ok:
            return
        if bool(self.filteredImageList):
            fileName = list(self.filteredImageList.keys())[0]
            filePath = list(self.filteredImageList.values())[0]
            filePath = filePath.replace(fileName, "")
            #self.ui.txtle_imagesPath.setText(filePath)
            for fImageName, fImagePath in self.filteredImageList.items():
                rImagePath=self.rawImageList.get(fImageName)
                VAImage.copyImage(fImagePath, rImagePath)
            self.filteredImageList = {}
            self.sldr_images_valueChanged(self.ui.sldr_images.value())
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
            if bool(self.filteredImageList):
                fileName = list(self.filteredImageList.keys())[0]
                filePath = list(self.filteredImageList.values())[0]
                filePath = filePath.replace(fileName, "")
                self.ui.txtle_imagesPath.setText(folderPath)
                for fImageName, fImagePath in self.filteredImageList.items():
                    #rImageName = self.rawImageList.get(fImageName)
                    dImagePath = os.path.join(folderPath, fImageName)
                    VAImage.copyImage(fImagePath, dImagePath)
                    self.rawImageList[fImageName]=dImagePath
                self.filteredImageList = {}
                self.sldr_images_valueChanged(self.ui.sldr_images.value())
                VAImage.deleteFolder(filePath)


        return