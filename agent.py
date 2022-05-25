import wheat_field
import random

"""
model_Field.go_next_wheat()
model_Field.is_finished()
model_Field.compute_explore_rate()
model_Field.finish_and_check()
model_Field.go_another_field()
model_Field.height_of_this_wheat()

model_Field.N
model_Field.k
model_Field.wheat_record

can be used by an agent.
"""
class Agent():
    def __init__(self):
        self.reward_info = None

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
        return self.reward_info

class Agent_37(Agent):
    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            return True

        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True


