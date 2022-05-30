import wheat_field
import random
import numpy as np
from scipy.stats import norm
from torch.distributions import Normal, kl_divergence


class Agent():
    def __init__(self):
        self.reward_info = None

    def to_continue(self, field):
        pass

    def reset_before_getWheat(self):
        self.__init__()

    def pre_train(self, field=None):
        pass

    def get_the_wheat(self, field):
        self.reset_before_getWheat()
        field.go_another_field()

        field.go_next_wheat() # Ensure at least one wheat now
        while not field.is_finished():
            if self.to_continue(field) == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
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

class Agent_prob_decision_5(Agent_normal_model):
    def to_continue(self, field):
        if field.k < 5:
            return True

        noLarger_prob = self.compute_noLarger_from_now(field)

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

class Agent_prob_decision_20(Agent_normal_model):
    def to_continue(self, field):
        if field.k < 20:
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

class Agent_threshold_learning(Agent_normal_model):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.reward = None
        self.w_size = 20
        self.weights = np.zeros((self.w_size,))
        self.lr_alpha = 0.1
        self.a_reduce = 0.0001
        self.lr_beta = 0.1
        self.b_reduce = 0.0001

    def normal_KL_with_now(self,mu1,sigma1):
        # print(self.mu, self.sigma,mu1, sigma1)
        return kl_divergence(Normal(self.mu, self.sigma), Normal(mu1, sigma1))

    def train_once(self, field):
        KL_3_20 = [0, 0]
        self.reset_before_getWheat()
        field.go_another_field()
        field.go_next_wheat()  # Ensure at least one wheat now

        while not field.is_finished():
            mu1 = self.mu
            sigma1 = self.sigma
            if field.k == 3:
                mu1 = np.mean(field.wheat_record[:-1])
                sigma1 = np.std(field.wheat_record[:-1])
            decide = self.to_continue(field)
            if field.k >= 3 and field.k <= 20:
                KL_3_20.append(self.normal_KL_with_now(mu1, sigma1))
            if decide == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.reward = self.reward_info[0]
        for j in range(min(20, field.k)):
            if field.wheat_record[j] > field.wheat_record[-1]:
                self.weights[j] = max(0, self.weights[j] - self.lr_alpha * KL_3_20[j])
                # self.weights[j] -= self.lr_alpha * KL_3_20[j]
        if field.k < self.w_size and self.reward < 3:
            self.weights[field.k - 1] += self.lr_beta * KL_3_20[field.k - 1]

        self.lr_alpha -= self.a_reduce
        self.lr_beta -= self.b_reduce

    def train(self, field):
        epoches = 1000
        for i in range(epoches):
            self.train_once(field)

    def reset_before_getWheat(self):
        Agent_normal_model().__init__()

    def to_continue(self, field):
        if field.k <= 2:
            return True

        mu1 = self.mu
        sigma1 = self.sigma
        if field.k == 3:
            mu1 = np.mean(field.wheat_record[:-1])
            sigma1 = np.std(field.wheat_record[:-1])
        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True

        if field.k <= self.w_size:
            noLarger_prob -= self.weights[field.k-1] * self.normal_KL_with_now(mu1, sigma1)
        if noLarger_prob > 0.5:
            return False
        else:
            return True

class Agent_fast_learning(Agent_threshold_learning):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.reward = None
        self.w_size = 20
        self.weights = np.zeros((self.w_size,))
        self.lr_alpha = 0
        self.a_discount = 0.9
        self.lr_beta = 0.1
        self.b_discount = 1

        self.separate_value = 3

    def train_once(self, field):
        KL_3_size = [0, 0]
        self.reset_before_getWheat()
        field.go_another_field()
        field.go_next_wheat()  # Ensure at least one wheat now

        while field.k <= self.w_size and field.is_finished() == False:
            mu1 = self.mu
            sigma1 = self.sigma
            if field.k == 3:
                mu1 = np.mean(field.wheat_record[:-1])
                sigma1 = np.std(field.wheat_record[:-1])
            decide = self.to_continue(field)
            if field.k >= 3 and field.k <= self.w_size:
                KL_3_size.append(self.normal_KL_with_now(mu1, sigma1))
            if decide == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.reward = self.reward_info[0]
        for j in range(min(self.w_size, field.k)):
            if field.wheat_record[j] > self.separate_value:
                self.weights[j] = max(0, self.weights[j] - self.lr_alpha * KL_3_size[j])
                # self.weights[j] -= self.lr_alpha * KL_3_20[j]
        if field.k <= self.w_size and self.reward < self.separate_value:
            self.weights[field.k - 1] += self.lr_beta * KL_3_size[field.k - 1]

        self.lr_alpha *= self.a_discount
        self.lr_beta *= self.b_discount

    def train(self, field):
        epoches = 1000
        for i in range(epoches):
            self.train_once(field)

"""
model_Field.N
model_Field.k
model_Field.wheat_record
model_Field.compute_explore_rate()
model_Field.height_of_this_wheat()
"""
