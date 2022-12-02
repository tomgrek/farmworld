import uuid

class Crop: # should be subclassed
    name = "corn"
    height = -1
    start_date = None

    def __init__(self, **kwargs):
        self.id = uuid.uuid4()
        for k, v in kwargs.items():
            assert k in self.__class__.__dict__, "Not a crop property."
            self.__dict__[k] = v
    
    def sow(self, start_date):
        self.height = 0
        self.start_date = start_date
    
    def is_planted(self):
        return self.height >= 0
    
    def grow(self):
        self.height += 1
    
    def reap(self, harvest_date):
        if self.start_date is None:
            return 0
        if harvest_date - self.start_date < 10:
            return 0
        yield_ = self.height
        yield_ *= (harvest_date - self.start_date) / 10
        self.height = -1
        return yield_

class Field:
    id = None
    feature = None
    render_coords = None
    render_info = None
    covered_area = None

    plants = None # for rendering only
    crop = None

    def __init__(self, **kwargs):
        self.id = uuid.uuid4()
        self.idx = kwargs.get("idx", -1)
        self.plants = []
        for k, v in kwargs.items():
            assert k in self.__class__.__dict__, "Not a field property."
            self.__dict__[k] = v
    
    def plant(self, planting_date, crop=None):
        self.crop = crop or Crop()
        self.crop.sow(planting_date)
    
    def harvest(self, harvest_date):
        yield_ = self.crop.reap(harvest_date)
        self.crop = None
        return yield_
    
    def grow(self):
        if self.is_planted:
            self.crop.grow()
    
    @property
    def is_planted(self):
        if self.crop is not None:
            return self.crop.is_planted
        return False
    
    @property
    def crop_height(self):
        if self.crop is None or not self.crop.is_planted:
            return -1
        return self.crop.height
    
    def reset(self):
        self.crop = None
        self.plants = []