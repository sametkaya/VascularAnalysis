from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
class Bat_QGraphicsView(QtWidgets.QGraphicsView):
    photoClicked = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent):
        super(Bat_QGraphicsView, self).__init__(parent)
        self.zoom = 0
        self.isEmpty = True
        self.scane = QtWidgets.QGraphicsScene(self)
        self.photo = QtWidgets.QGraphicsPixmapItem()
        self.scane.addItem(self.photo)
        #self.scane.setForegroundBrush(QColor(255, 255, 255, 127));
        #self.size(parent.size())
        self.setScene(self.scane)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        #self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self.isEmpty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self.photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self.zoom = 0

    def setImageBackground(self, qimage=None):

        pixmap = QPixmap.fromImage(qimage)
        self.zoom = 0
        if pixmap and not pixmap.isNull():
            self.isEmpty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.photo.setPixmap(pixmap)
        else:
            self.isEmpty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self.zoom += 1
            else:
                factor = 0.8
                self.zoom -= 1
            if self.zoom > 0:
                self.scale(factor, factor)
            elif self.zoom == 0:
                self.fitInView()
            else:
                self.zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self.photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self.photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(Bat_QGraphicsView, self).mousePressEvent(event)