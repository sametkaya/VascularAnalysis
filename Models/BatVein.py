import json

from Models.BatPoint import BatPoint


class BatVein:
    def __init__(self, start=BatPoint(), end=BatPoint()):
        self.start = start
        self.end = end
        self.points = []
        self.lenght = -1
        self.segments = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)