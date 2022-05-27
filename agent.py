import wheat_field
import random
import numpy as np
from scipy.stats import norm
import math


class Agent():
    def __init__(self):
        self.reward_info = None

    def learning(self):
        pass

    def to_continue(self, field):
        pass

    def get_the_wheat(self, field):
        self.__init__()
        field.go_another_field()

        field.go_next_wheat() # Ensure at least one wheat now
        while not field.is_finished():
            if self.to_continue(field) == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.learning()
        return self.reward_info

class Agent_random(Agent):
    def to_continue(self, field):
        if len(field.wheat_record) == 0:
            return True
        random_number = random.uniform(0, 1)
        if random_number <= 0.5:
            return False
        else:
            return True

class Agent_37(Agent):
    def to_continue(self, field):
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

    def to_continue(self, field):
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
    def to_continue(self, field):
        if field.k < field.N**0.5:
            return True
        if field.height_of_this_wheat() == max(field.wheat_record):
            return False
        else:
            return True

class Agent_normal_model(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.mu = None
        self.sigma = None # std

    def to_continue(self, field):
        pass

    def compute_noLarger_from_now(self, field):
        wheat_arr = np.array(field.wheat_record)
        self.mu = np.mean(wheat_arr)
        self.sigma = np.std(wheat_arr)
        if self.sigma < 0.001:
            return -1
        # not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        not_larger_prob = norm.logcdf(field.height_of_this_wheat(), self.mu, self.sigma)
        not_larger_prob *= field.N - field.k
        return np.exp(not_larger_prob)

class Agent_prob_decision(Agent_normal_model):
    def to_continue(self, field):
        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True

        if noLarger_prob > 0.5:
            return False
        else:
            return True

class Agent_prob_decision_10(Agent_normal_model):
    def to_continue(self, field):
        if field.k < 10:
            return True

        noLarger_prob = self.compute_noLarger_from_now(field)

        if noLarger_prob > 0.5:
            return False
        else:
            return True

class Agent_prob_decision_d9(Agent_normal_model):
    def to_continue(self, field):
        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True

        if noLarger_prob > 0.9:
            return False
        else:
            return True

class Agent_prob_rand(Agent_normal_model):
    def to_continue(self, field):
        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True

        if random.uniform(0,1) < noLarger_prob:
            return False
        else:
            return True

class Agent_prob_rand_10(Agent_normal_model):
    def to_continue(self, field):
        if field.k < 10:
            return True

        noLarger_prob = self.compute_noLarger_from_now(field)

        if random.uniform(0,1) < noLarger_prob:
            return False
        else:
            return True

# Can only explore Normal_Field_Leak !!
class Agent_prob_decision_leak(Agent_normal_model):
    def to_continue(self, field):
        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True

        if noLarger_prob > 0.5:
            return False
        else:
            return True

    def compute_noLarger_from_now(self, field):
        self.mu = field.mean_leak()
        self.sigma = field.std_leak()

        # not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        not_larger_prob = norm.logcdf(field.height_of_this_wheat(), self.mu, self.sigma)
        not_larger_prob *= field.N - field.k
        return np.exp(not_larger_prob)


"""
model_Field.N
model_Field.k
model_Field.wheat_record
model_Field.compute_explore_rate()
model_Field.height_of_this_wheat()
"""


