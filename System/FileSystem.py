import json
import os

from PySide6 import QtWidgets
from PySide6.QtCore import QDir, QFileInfo
from PySide6.QtWidgets import QFileDialog, QWidget

from Datas.Data import Data


class FileSystem(object):
    projectParentFolderPath = None
    projectFolderGeneralName = "BAT_Project"
    projectConfigFileName = "BAT_Project_Config.json"
    projectFolderPath = None

    projectConfigFilePath = None
    projectFolderImagesPath = None

    @staticmethod
    def CreateNewProject(parentWidget: QWidget):

        dlg = QFileDialog(parentWidget)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        home = os.getcwd()
        dlg.setDirectory(home)
        dlg.setWindowTitle("Create Project")
        # dlg.set(QFileDialog.ShowDirsOnly)

        if dlg.exec_():

            path = dlg.selectedFiles()[0]
            FileSystem.projectParentFolderPath = QDir.toNativeSeparators(path)
            path = QDir.toNativeSeparators(os.path.join(FileSystem.projectParentFolderPath, FileSystem.projectFolderGeneralName))
            # print(path)
            num = 0
            extend = ""
            while os.path.exists(path + extend):
                extend = "_" + str(num)
                num = num + 1
            path = path + extend
            FileSystem.projectFolderPath = path
            # print(path)
            os.mkdir(path)
            FileSystem.projectFolderImagesPath = QDir.toNativeSeparators(os.path.join(path, "Images"))
            os.mkdir(FileSystem.projectFolderImagesPath)
            FileSystem.projectConfigFilePath = QDir.toNativeSeparators(
                os.path.join(path, FileSystem.projectConfigFileName))
            open(FileSystem.projectConfigFilePath, "w")

    @staticmethod
    def SaveProject():
        jsonDict={}
        jsonDict[""]
        batImageDict = [bi.toDict() for bi in Data.batImageList]
        data = {'Bat Images': batImageDict}
        json_string = json.dumps(data)

        return

    @staticmethod
    def OpenProject(parentWidget: QWidget):
        dlg = QFileDialog(parentWidget)
        dlg.setFileMode(QFileDialog.ExistingFile)
        home = os.getcwd()
        dlg.setDirectory(home)
        dlg.setWindowTitle("Open Project")
        dlg.setNameFilter("BAT Project File (BAT_Project_Config.json)")

        if dlg.exec_():
            batConfigFile = dlg.selectedFiles()[0]
            FileSystem.projectParentFolderPath = QDir.toNativeSeparators(QFileInfo(batConfigFile).dir().path())
            FileSystem.projectFolderImagesPath = QDir.toNativeSeparators(os.path.join(FileSystem.projectParentFolderPath, "Images"))

        return
