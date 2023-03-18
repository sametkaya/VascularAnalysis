from PySide6 import QtWidgets
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QWidget, QStackedLayout

from Datas.Data import Data
from Models.BatImage import BatImage


class ImageScreenWidget_Controller():
    def __init__(self, imageScreenWidget_Form):
        self.form = imageScreenWidget_Form
        return

    def hsldr_frm_image_process_image_number_valueChanged(self):
        self.form.ui.spnbx_frm_image_process_image_number.setValue(self.form.ui.hsldr_frm_image_process_image_number.value())
        #self.SetImage(self.form.ui.hsldr_frm_image_process_image_number.value())
        return

    def vsldr_frm_image_process_opacity_valueChanged(self):
        self.SetImage(self.form.ui.hsldr_frm_image_process_image_number.value())
        return

    def spnbx_frm_image_process_image_number_valueChanged(self):
        self.form.ui.hsldr_frm_image_process_image_number.setValue(self.form.ui.spnbx_frm_image_process_image_number.value())
        #self.SetImage(self.form.ui.spnbx_frm_image_process_image_number.value())
        return

    def SetBatImageList(self):
        self.form.ui.hsldr_frm_image_process_image_number.setEnabled(True)
        self.form.ui.hsldr_frm_image_process_image_number.setValue(0)
        self.form.ui.hsldr_frm_image_process_image_number.setMinimum(0)
        self.form.ui.hsldr_frm_image_process_image_number.setMaximum(len(Data.batImageList)-1)

        self.form.ui.spnbx_frm_image_process_image_number.setEnabled(True)
        self.form.ui.spnbx_frm_image_process_image_number.setValue(0)
        self.form.ui.spnbx_frm_image_process_image_number.setMaximum(len(Data.batImageList)-1)
        self.form.ui.spnbx_frm_image_process_image_number.setMinimum(0)

        self.form.ui.vsldr_frm_image_process_opacity.setEnabled(True)
        self.form.ui.vsldr_frm_image_process_opacity.setValue(100)


    def SetDisable(self):
        self.form.ui.hsldr_frm_image_process_image_number.setEnabled(False)
        self.form.ui.hsldr_frm_image_process_image_number.setValue(0)
        self.form.ui.hsldr_frm_image_process_image_number.setMaximum(0)
        self.form.ui.hsldr_frm_image_process_image_number.setMinimum(0)

        self.form.ui.spnbx_frm_image_process_image_number.setEnabled(False)
        self.form.ui.spnbx_frm_image_process_image_number.setValue(0)

        self.form.ui.vsldr_frm_image_process_opacity.setEnabled(False)
        self.form.ui.vsldr_frm_image_process_opacity.setValue(100)



    def SetImage(self, index):

        batImage= Data.GetBatImage(index)
        if (batImage==None):
            return

        painter = QPainter()
        image = batImage.rawImage.copy()
        # painter.setOpacity(float(self.sldr_rawImageOpacity.value()) / 100)
        painter.begin(image)
        num= self.form.ui.vsldr_frm_image_process_opacity.value()
        painter.setOpacity(float(self.form.ui.vsldr_frm_image_process_opacity.value()) / 100)
        painter.drawImage(0, 0, batImage.processingImage)
        painter.end()
        self.form.ui.grpvw_image_process_scane.setImageBackground(image)
        return

