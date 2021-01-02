from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import QMessageBox
def openOkCancelWarningMessageBox(title,text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    retval = msg.exec_()
    return retval