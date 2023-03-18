import os

from PySide6 import QtCore
from PySide6.QtCore import QPropertyAnimation, QFileInfo
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog


from Datas.Data import Data
from System.FileSystem import FileSystem
from System.ImageOperation import ImageOperation


class MainWindow_Controller():
    def __init__(self):

        return
    # region top

    def pbtn_exit_clicked(self):
        self.close()

    def pbtn_minimise_clicked(self):
        self.showMinimized()

    def pbtn_expand_clicked(self):
        if self.isScreenModeNormal:
            self.setFullScreen()
        else:
            self.setNormalScreen()


    # endregion

    # region middle

    def hsldr_frm_image_process_image_number_valueChanged(self):
        index= self.grapvw_sceneImage.ui.spnbx_frm_image_process_image_number.value()
        self.ui.lstw_left_menuContent_images_processingImages.setCurrentRow(index)

    #def spnbx_frm_image_process_image_number_valueChanged(self):

    # endregion

    #### region buttons

    ### left content
    #  menu buttons controls
    def pbtn_left_menu_openClose_clicked(self):
        self.frm_left_content_left_menuTools_OpenClose_switchAction()

    def pbtn_left_menu_tool_clicked(self):
        self.frm_left_content_right_menuContent_switchAction(self.sender())
        self.wgtstck_middle_content_switchAction(self.sender())
    #
    # left menu content images controls


    def pbtn_right_menu_tool_images_loadImages_clicked(self):
        #(rawImages,processingImages) = LoadImages()
        batImageList = ImageOperation.LoadImages(self)
        self.grapvw_sceneImage.cntlr.SetBatImageList()
        self.grapvw_sceneImage.cntlr.SetImage(0)

        Data.SetRawImageNamestToListWidget(self.ui.lstw_left_menuContent_images_rawImages)
        #rawImageName = self.ui.lstw_left_menuContent_images_rawImages.currentItem().text()
        #self.ui.ledt_left_menuContent_images_rawImages.setText(str(rawImageName))

        Data.SetProcessingImageNamestToListWidget(self.ui.lstw_left_menuContent_images_processingImages)
        self.ui.lstw_left_menuContent_images_processingImages.setCurrentRow(0)
        #processingImageName = self.ui.lstw_left_menuContent_images_processingImages.currentItem().text()
        #self.ui.ledt_left_menuContent_images_processingImage.setText(str(processingImageName))
        return

    def lstw_left_menuContent_images_rawImages_selectionChanged(self, selected, deselected):
        index= self.ui.lstw_left_menuContent_images_rawImages.currentIndex().row()
        self.ui.lstw_left_menuContent_images_processingImages.setCurrentRow(index)

        rawImageName = self.ui.lstw_left_menuContent_images_rawImages.currentItem().text()
        self.ui.ledt_left_menuContent_images_rawImages.setText(str(rawImageName))
        processingImageName = self.ui.lstw_left_menuContent_images_processingImages.currentItem().text()
        self.ui.ledt_left_menuContent_images_processingImage.setText(str(processingImageName))

        self.grapvw_sceneImage.cntlr.SetImage(index)
        return

    def lstw_left_menuContent_images_processingImages_selectionChanged(self, selected, deselected):
        index = self.ui.lstw_left_menuContent_images_processingImages.currentIndex().row()
        self.ui.lstw_left_menuContent_images_rawImages.setCurrentRow(index)

        rawImageName = self.ui.lstw_left_menuContent_images_rawImages.currentItem().text()
        self.ui.ledt_left_menuContent_images_rawImages.setText(str(rawImageName))
        processingImageName = self.ui.lstw_left_menuContent_images_processingImages.currentItem().text()
        self.ui.ledt_left_menuContent_images_processingImage.setText(str(processingImageName))

        self.grapvw_sceneImage.cntlr.SetImage(index)
        return



    #

    # left content project menu
    def pbtn_right_menu_tool_project_openProject_clicked(self):
        FileSystem.OpenProject(self)

    def pbtn_right_menu_tool_project_createProject_clicked(self):
        FileSystem.CreateNewProject(self)

    def pbtn_right_menu_tool_project_saveProject_clicked(self):
        FileSystem.SaveProject()
        return

    #
    ### left content end
    #  buttons controls
    def pbtn_right_content_openClose_clicked(self):
        self.frm_right_content_openClose_switchAction()


    #### endregion



        #self.wdgt_leftMenuSubContainer.setMaximumWidth(newWidth)


        # self.animation = QPropertyAnimation(self.wgt_leftToolsContainer,b"minimumWidth")
        # self.animation.setDuration(250)
        # self.animation.setStartValue(width)
        # self.animation.setEndValue(newWidth)
        # self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        # self.animation.start()

        return


