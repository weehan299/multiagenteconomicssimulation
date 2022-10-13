    
def validate_action_space_num(self,attribute,value):
    if value <= 0:
        raise ValueError("number of elements in action space should be positive")

def validate_total_periods(self,attribute,value):
    if value <= 0:
        raise ValueError("total number of periods in total period should be positive")

def validate_xi(self,attribute,value):
    if value < 0:
        raise ValueError("xi should be positive")

def validate_gamma(self, attribute, value):
    if value > 1 or value < 0:
        raise ValueError("gamma shld be between 0 and 1")

def validate_alpha(self, attribute, value):
    if value > 1 or value < 0:
        raise ValueError("alpha shld be between 0 and 1")

def validate_beta(self, attribute, value):
    if value < 0:
        raise ValueError("beta shld be positive")