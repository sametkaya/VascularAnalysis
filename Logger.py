from enum import Enum
from PySide6.QtWidgets import *
from PySide6.QtGui import *

class LogTypes(Enum):
    none = 0
    success = 1
    error = 2
    warning = 3
    running = 4

class Logger():
    def __init__(self, listView):
        self.listView = listView

    def log(self, logType, message):
        if logType == LogTypes.none:
            item = QListWidgetItem(">> done: "+message)
            item.setForeground(QColor("green"))
            self.listView.addItem(item)
        elif logType == LogTypes.success:
            item = QListWidgetItem(">> success: "+message)
            item.setForeground(QColor("cyan"))
            self.listView.addItem(item)
        elif logType == LogTypes.error:
            item = QListWidgetItem(">> error: "+message)
            item.setForeground(QColor("red"))
            self.listView.addItem(item)
        elif logType == LogTypes.warning:
            item = QListWidgetItem(">> warning: "+message)
            item.setForeground(QColor("yellow"))
            self.listView.addItem(item)
        elif logType == LogTypes.running:
            item = QListWidgetItem(">> running: "+message)
            item.setForeground(QColor("magenta"))
            self.listView.addItem(item)

    def clearLogs(self):
        self.listView.clear()




