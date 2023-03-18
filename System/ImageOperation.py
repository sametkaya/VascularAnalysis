import os

from PySide6.QtCore import QFileInfo
from PySide6.QtWidgets import QFileDialog, QWidget

from Datas.Data import Data
from Models.BatImage import BatImage
from System.FileSystem import FileSystem


class ImageOperation(object):

    @staticmethod
    def LoadImages(parentWidget:QWidget):

        dlg = QFileDialog(parentWidget)
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setWindowTitle("Load Images")
        dlg.setNameFilter("Image Files (*.jpg *.png *.tif *.tiff)")
        imageFiles = []

        if dlg.exec_():
            imageFiles = dlg.selectedFiles()
            if len(imageFiles) <= 0:
                raise Exception("No image found")
                return
            #find VAI_ folder to create new folder
            path = QFileInfo(imageFiles[0]).absolutePath()
            #folderCount = VAImage.findCountOfDirectoryStartWithName(path, "VAI_ProcessingImages")
            #processingFolderPath = VAImage.createFolder(path, "VAI_ProcessingImages_" + str(folderCount))
            for item in imageFiles:
                imageName = QFileInfo(item).fileName()
                imagePath=os.path.join(FileSystem.projectFolderImagesPath, imageName)
                nimg= BatImage(name=imageName, path=imagePath)

                Data.AppendImage(nimg)
                #Data.processingBatImageList[imageName] = nimg
                #VAImage.copyImage(self.rawImageList[imageName], self.processingImageList[imageName])
                #self.lstw_imagesRawList.addItem(imageName)
                #self.lstw_imagesProcessingList.addItem(imageName)
            # if len(imageFiles) != 0:
            #     imageName = QFileInfo(imageFiles[0]).fileName()
            #     self.txtle_imagesRawPath.setText(QFileInfo(self.rawImageList[imageName]).absolutePath())
            #     self.txtle_imagesProcessingPath.setText(QFileInfo(self.processingImageList[imageName]).absolutePath())
            #     self.lstw_imagesRawList.setCurrentRow(0)
            #     self.sldr_images.setMaximum(len(imageFiles) - 1)
            #     self.sldr_images.setTracking(True)
            #     self.sldr_images_valueChanged(0)


        return Data.batImageList
