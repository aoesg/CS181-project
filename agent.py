import numpy as np

import wheat_field
import random

class Agent():
    def __init__(self):
        self.reward_info = None

    def learning(self):
        pass

    def to_contiune(self, field):
        pass

    def get_the_wheat(self, field):
        self.reward_info = None

        field.go_another_field()

        while not field.is_finished():
            if self.to_contiune(field) == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.learning()
        return self.reward_info

class Agent_37(Agent):
    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            return True

        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True

class Agent_37_t3(Agent):
    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            return True
        t3_h = field.wheat_record.copy()
        t3_h.sort(reverse = True)
        if field.height_of_this_wheat() in t3_h[0:3]:
            return False
        else:
            return True

class Agent_sqrt_n(Agent):
    def to_contiune(self, field):
        if field.k < np.sqrt(field.N):
            return True
        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True

class Agent_sqrt_n_t3(Agent):
    def to_contiune(self, field):
        if field.k < np.sqrt(field.N):
            return True
        t3_h = field.wheat_record.copy()
        t3_h.sort(reverse=True)
        if field.height_of_this_wheat() in t3_h[0:3]:
            return False
        else:
            return True

"""
model_Field.N
model_Field.k
model_Field.wheat_record
model_Field.compute_explore_rate()
model_Field.height_of_this_wheat()
"""



