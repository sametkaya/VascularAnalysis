# This Python file uses the following encoding: utf-8
from PySide6 import QtWidgets

from Windows.MainWindow.MainWindow_Form import MainWindow_Form

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MyMainWindow = MainWindow_Form()
    MyMainWindow.show()
    sys.exit(app.exec())