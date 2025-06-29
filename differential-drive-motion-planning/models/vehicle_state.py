class VehicleState:
    def __init__(self, x=0, y=0, yaw=0, v=0, w=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.w = w
        
    def to_array(self):
        return np.array([self.x, self.y, self.yaw, self.v, self.w])
    
    @classmethod
    def from_array(cls, arr):
        return cls(*arr)