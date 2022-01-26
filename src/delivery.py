from src.coordinate import Coordinate


class Delivery:
    def __init__(self, raw_delivery: dict, index: int):
        self.id = index + 1
        self.point = Coordinate(str(raw_delivery["point"]["lat"]), str(raw_delivery["point"]["lng"]))
        self.size = raw_delivery["size"]

    def __str__(self):
        return str(self.id)
