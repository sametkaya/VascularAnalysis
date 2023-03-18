import json
from PySide6.QtGui import QImage


class BatImage:
    def __init__(self, name=None, path=None):
        self.rawImagePath = path
        self.rawImageName = name
        self.rawImage = QImage(self.rawImagePath)
        self.processingImagePath = path
        self.processingImageName = name
        self.processingImage = QImage(self.processingImagePath)
        self.veins = {}

    def toJSON(self):
        return json.dumps(self, self.toDict(),
                          sort_keys=True, indent=4)

    def toDict(self):
        return {
            'rawImagePath': self.rawImagePath,
            'rawImageName': self.rawImageName,
            'processingImagePath': self.processingImagePath,
            'processingImageName': self.processingImageName
            #'veins': self.veins.to_dict(),
        }
