import wheat_field
import random
import numpy as np  # new import
from scipy.stats import norm        # new import

class Agent():
    def __init__(self):
        self.reward_info = None

    def learning(self):
        pass

    def to_contiune(self, field):
        pass

    def get_the_wheat(self, field):
        self.__init__()
        field.go_another_field()

        field.go_next_wheat() # Ensure at least one wheat now
        while not field.is_finished():
            if self.to_contiune(field) == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.learning()
        return self.reward_info

class Agent_random(Agent):
    def to_contiune(self, field):
        if len(field.wheat_record) == 0:
            return True
        random_number = random.uniform(0, 1)
        if random_number <= 0.5:
            return False
        else:
            return True

class Agent_37(Agent):
    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            return True

        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True

class Agent_37_t3(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.t3_rank = [float('-inf'), float('-inf'), float('-inf')]

    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            if field.height_of_this_wheat() > self.t3_rank[0]:
                self.t3_rank[0] = field.height_of_this_wheat()
                self.t3_rank.sort()
            return True

        if field.height_of_this_wheat() > self.t3_rank[0]:
            return False
        else:
            return True


class Agent_sqrt_n(Agent):
    def to_contiune(self, field):
        if field.k < field.N**0.5:
            return True
        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True

class Agent_prob_decision(Agent):
    def to_contiune(self, field):
        if len(field.wheat_record) == 0:
            return True
        wheat_list = np.array(field.wheat_record)
        # use MLE to perdict normal distribution parameters with before and current samples
        mu = np.mean(wheat_list)
        sigma = np.var(wheat_list)
        not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        if not_larger_prob >= 0.5:
            return False
        else:
            return True

class Agent_prob(Agent):
    def to_contiune(self, field):   # to contiune 写错
        if len(field.wheat_record) == 0:
            return True
        wheat_list = np.array(field.wheat_record)
        # use MLE to perdict normal distribution parameters with before and current samples
        mu = np.mean(wheat_list)
        sigma = np.var(wheat_list)
        not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        random_number = random.uniform(0, 1)
        if random_number <= not_larger_prob:
            return False
        else:
            return True

class Agent_prob_decision_37(Agent):
    def to_contiune(self, field):
        if field.compute_explore_rate() < 0.37:
            return True
        if len(field.wheat_record) == 0:
            return True
        wheat_list = np.array(field.wheat_record)
        # use MLE to perdict normal distribution parameters with before and current samples
        mu = np.mean(wheat_list)
        sigma = np.var(wheat_list)
        not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        if not_larger_prob >= 0.5:
            return False
        else:
            return True

class Agent_prob_decision_former(Agent):
    def to_contiune(self, field):
        if len(field.wheat_record) == 0:
            return True
        wheat_list = np.array(field.wheat_record)
        # use MLE to perdict normal distribution parameters with before and current samples
        mu = np.mean(wheat_list)
        sigma = np.var(wheat_list)
        temp_prob = norm.cdf(wheat_list[-1], mu, np.sqrt(sigma))
        if temp_prob >= 0.9:
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


