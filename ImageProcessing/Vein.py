
class MPoint:
    def __init__(self, x=-1, y=-1, radius=0):
        self.center = [x, y]
        self.radius = 0

class Vein:
    def __init__(self):
        self.len = 0
        self.startMPoint = [-1, -1]
        self.endMPoint = [-1, -1]
        self.mPoints = []
        self.pointCount = 0


