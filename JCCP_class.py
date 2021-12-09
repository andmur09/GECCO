class timePoint(object):
    def __init__(self, name, description, controllable = None):
        self.id = name
        self.description = description
        self.controllable = controllable
        # self.id = next(timePoint.newName)
        # self.name = "t({})".format(str(self.id))
        
    def setControllable(self, logic):
    ## Used to set attribute for timepoint. If logic == True, timepoint is controllable, if logic == False, timepoint is uncontrollable
        self.controllable = logic
    
    def isControllable(self):
        if self.controllable == True:
            return True
        elif self.controllable == False:
            return False
        else:
            raise AttributeError("Controllable attribute not set for time-point")
    
    def getName(self):
        return self.name
    
    def __str__(self):
        return "Time-point {}".format(self.id)
    
    def copy(self):
        return timePoint(self.description[:])