import uuid

class Field:
    id = None
    feature = None
    render_coords = None
    render_info = None
    covered_area = None

    planted = False
    plants = None

    plant_date = 0
    harvested = False
    harvest_date = 0
    crop_height = 0.0
    crop_height_at_harvest = 0.0

    def __init__(self, **kwargs):
        self.id = uuid.uuid4()
        self.plants = []
        for k, v in kwargs.items():
            assert k in self.__class__.__dict__, "Not a field property."
            self.__dict__[k] = v
        