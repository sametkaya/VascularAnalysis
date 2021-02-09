from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from ImageViewer_Ui import Ui_ImageViewer
from VAT_QGraphicsView import VAT_QGraphicsView
class ImageViewerForm(QMainWindow, Ui_ImageViewer):
    def __init__(self, rawImageList,processingImageList,startIndex=0,parent=None):
        super(ImageViewerForm, self).__init__(parent)
        self.setupUi(self)
        self.rawImageList = rawImageList
        self.processingImageList = processingImageList
        self.imageList = list(rawImageList.keys())
        self.initilizeComponent()
        #self.sldr_images_valueChanged(startIndex)


    def initilizeComponent(self):
        self.grapvw_sceneImage = VAT_QGraphicsView(self.wgt_sceneImage)
        self.grapvw_sceneImage.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.wgt_sceneImage.sizePolicy().hasHeightForWidth())
        self.grapvw_sceneImage.setSizePolicy(sizePolicy)
        self.grapvw_sceneImage.setMinimumSize(QSize(650, 400))
        self.grapvw_sceneImage.setInteractive(True)
        self.lyt_sceneImage.addWidget(self.grapvw_sceneImage)
        self.grapvw_sceneImage.toggleDragMode()
        self.spnb_images.valueChanged.connect(self.spnb_images_valueChanged)
        self.sldr_images.valueChanged.connect(self.sldr_images_valueChanged)
        self.spnb_images.setMaximum(len(self.rawImageList)-1)
        self.sldr_images.setMaximum(len(self.rawImageList)-1)

        if bool(self.processingImageList):
            self.sldr_filteredImageOpacity.setEnabled(True)
        else:
            self.sldr_filteredImageOpacity.setEnabled(False)

    def imageChanged(self, key):

        painter = QPainter()
        imagePath = self.rawImageList.get(key)
        image = QImage(imagePath)
        # painter.setOpacity(float(self.sldr_rawImageOpacity.value()) / 100)
        painter.begin(image)
        if key in self.processingImageList:
            imagePath2 = self.processingImageList.get(key)
            image2 = QImage(imagePath2)
            painter.setOpacity(float(self.sldr_filteredImageOpacity.value()) / 100)
            painter.drawImage(0, 0, image2)
        painter.end()
        self.grapvw_sceneImage.setPhoto(image)

    def sldr_images_valueChanged(self, value):
        key= self.imageList[value]
        self.imageChanged(key)
        self.spnb_images.setValue(value)

    def spnb_images_valueChanged(self, value):
        key= self.imageList[value]
        self.imageChanged(key)
        self.sldr_images.setValue(value)

