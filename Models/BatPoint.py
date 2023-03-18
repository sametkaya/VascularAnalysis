import json


class BatPoint:
    def __init__(self, z=0, x=0, y=0, radius=-1):
        self.coordinate = [z, x, y]
        self.radius = radius

    def __init__(self, coordinate=[0, 0, 0], radius=-1):
        self.coordinate = coordinate
        self.radius = radius

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)