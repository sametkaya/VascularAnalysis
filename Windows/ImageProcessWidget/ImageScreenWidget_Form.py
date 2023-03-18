from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget

from Windows.CustomTools.Bat_QGraphicsView import Bat_QGraphicsView
from Windows.ImageProcessWidget.ImageScreenWidget_Controller import ImageScreenWidget_Controller
from Windows.ImageProcessWidget.ImageScreenWidget_View import Ui_ImageScreenWidget


class ImageScreenWidget_Form(QWidget):

    def __init__(self):
        super(ImageScreenWidget_Form, self).__init__()
        self.ui = Ui_ImageScreenWidget()
        self.ui.setupUi(self)
        self.cntlr = ImageScreenWidget_Controller(self)
        self.initilizeComponent()


    def initilizeComponent(self):
        self.ui.grpvw_image_process_scane= Bat_QGraphicsView(self.ui.frm_image_process_scane)
        self.ui.vlyt_frm_image_process_scane.addWidget(self.ui.grpvw_image_process_scane)

        self.ui.hsldr_frm_image_process_image_number.valueChanged.connect(self.cntlr.hsldr_frm_image_process_image_number_valueChanged)
        self.ui.hsldr_frm_image_process_image_number.setEnabled(False)
        self.ui.hsldr_frm_image_process_image_number.setValue(0)
        self.ui.hsldr_frm_image_process_image_number.setMaximum(0)
        self.ui.hsldr_frm_image_process_image_number.setMinimum(0)

        self.ui.vsldr_frm_image_process_opacity.valueChanged.connect(self.cntlr.vsldr_frm_image_process_opacity_valueChanged)
        self.ui.vsldr_frm_image_process_opacity.setEnabled(False)
        self.ui.hsldr_frm_image_process_image_number.setValue(100)

        self.ui.spnbx_frm_image_process_image_number.valueChanged.connect(self.cntlr.spnbx_frm_image_process_image_number_valueChanged)
        self.ui.spnbx_frm_image_process_image_number.setEnabled(False)
        self.ui.spnbx_frm_image_process_image_number.setValue(0)







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MyImageWindow_Form = ImageScreenWidget_Form()
    MyImageWindow_Form.show()
    sys.exit(app.exec())

