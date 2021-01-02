# This Python file uses the following encoding: utf-8
import sys
import os

import PySide2
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from MainForm import MainForm

if __name__ == "__main__":
    app = QApplication(sys.argv)
    wdgt_mainForm1 = MainForm()
    wdgt_mainForm1.show()

    sys.exit(app.exec_())
