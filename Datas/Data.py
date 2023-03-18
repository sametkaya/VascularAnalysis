class Data(object):
    batImageList = []

    def RemoveBatImage(index):
        if (-1 < index) and (index < len(Data.batImageList)):
            Data.batImageList.pop(index)

    def AppendImage(image):
        Data.batImageList.append(image)

    def GetBatImage(index):
        if (-1 < index) and (index < len(Data.batImageList)):
            return Data.batImageList[index]
        return None

    def SetRawImageNamestToListWidget(qlistWidget):
        qlistWidget.clear()
        for bimage in Data.batImageList:
            qlistWidget.addItem(bimage.rawImageName)
        #qlistWidget.setCurrentRow(0)

    def SetProcessingImageNamestToListWidget(qlistWidget):
        qlistWidget.clear()
        for bimage in Data.batImageList:
            qlistWidget.addItem(bimage.processingImageName)
        #qlistWidget.setCurrentRow(0)