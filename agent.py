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

class Agent_upper_bound(Agent):
    def to_continue(self, field):
        return True

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
        choosen = max(field.wheat_record)
        normazlized_res = field.reward_normalize(choosen)
        self.reward_info = field.check_info(normazlized_res)
        return self.reward_info

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

class Agent_threshold_learning(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.threshold = -1
        self.lr = 5e-2

    def train(self, field):
        self.lr *= 1e2/field.N
        self.threshold = int(1e-1*field.N)
        upper_bound = int(20e-2*field.N)
        lower_bound = int(5e-2*field.N)
        epoches = 10000
        for _ in range(epoches):
            self.reset_before_getWheat()
            field.go_another_field()
            field.go_next_wheat()  # Ensure at least one wheat now

            while not field.is_finished():
                field.go_next_wheat()
            reward = self.simulate_threshold_choose(field,self.threshold)
            temp1 = self.threshold
            temp2 = self.threshold          
            for j in range(lower_bound, upper_bound):
                if self.simulate_threshold_choose(field, j) > reward:
                    temp2 += -self.lr*temp1+self.lr*j
            self.threshold = temp2

    def simulate_threshold_choose(self, field, threshold):
        threshold = int(threshold)
        max_wheat = max(field.wheat_record[:threshold])
        for i in range(threshold,field.N):
            if field.wheat_record[i] > max_wheat:
                return field.wheat_record[i]
        return -1

    def reset_before_getWheat(self):
        Agent_normal_model.__init__(self)
        self.threshold = int(self.threshold)

    def to_continue(self, field):
        if field.k <= self.threshold:
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

class Agent_prob_decision_3(Agent_normal_model):
    def to_continue(self, field):
        if field.k < 3:
            return True

        noLarger_prob = self.compute_noLarger_from_now(field)

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

class Agent_prob_record(Agent_normal_model):
    def __init__(self):
        Agent_normal_model.__init__(self)

        self.mu_record = []
        self.sigma_record = []

    def to_continue(self, field):
        wheat_arr = np.array(field.wheat_record)
        self.mu = np.mean(wheat_arr)
        self.sigma = np.std(wheat_arr)

        self.mu_record.append(self.mu)
        self.sigma_record.append(self.sigma)

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

class Agent_prob_gain_learning(Agent_normal_model):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.reward = None
        self.w_size = 10
        self.weights = np.zeros((self.w_size,))
        for i in range(3):
            self.weights[i] = 1.0
        self.lr_alpha = 1e-4
        self.lr_beta = 1e-3

    def train(self, field):
        epoches = 5000
        for _ in range(epoches):
            self.reset_before_getWheat()
            field.go_another_field()
            field.go_next_wheat()  # Ensure at least one wheat now
            
            while not field.is_finished():
                decide = self.to_continue(field)
                if  decide == True:
                    field.go_next_wheat()
                else:
                    break
            self.reward_info = field.finish_and_check()
            self.reward = self.reward_info[0]
            for j in range(min(self.w_size, field.k)):
                if field.wheat_record[j] > field.wheat_record[-1]:
                    self.weights[j] -= self.lr_alpha
            if field.k < self.w_size and self.reward < 3:
                self.weights[field.k-1] += self.lr_beta

    def reset_before_getWheat(self):
        Agent_normal_model.__init__(self)

    def to_continue(self, field):
        if field.k < 3:
            return True

        noLarger_prob = self.compute_noLarger_from_now(field)
        if noLarger_prob == -1:
            return True
        if field.k <= self.w_size:
            noLarger_prob -= self.weights[field.k-1]
        if noLarger_prob > 0.5:
            return False
        else:
            return True

class Agent_prob_KL_gain_learning(Agent_normal_model):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.reward = None
        self.w_size = 10
        self.weights = np.zeros((self.w_size,))
        self.lr_alpha = 0.1
        self.a_reduce = 1e-4
        self.lr_beta = 0.1
        self.b_reduce = 1e-4

    def normal_KL_with_now(self,mu1,sigma1):
        return kl_divergence(Normal(self.mu, self.sigma), Normal(mu1, sigma1))

    def train_once(self, field):
        KL_3_size = [0, 0]
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
            if field.k >= 3 and field.k <= self.w_size:
                KL_3_size.append(self.normal_KL_with_now(mu1, sigma1))
            if decide == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.reward = self.reward_info[0]
        for j in range(min(self.w_size, field.k)):
            if field.wheat_record[j] > field.wheat_record[-1]:
                self.weights[j] = max(0, self.weights[j] - self.lr_alpha*KL_3_size[j])
        if field.k <= self.w_size and self.reward < 2:
            self.weights[field.k-1] += self.lr_beta*KL_3_size[field.k-1];

        self.lr_alpha -= self.a_reduce
        self.lr_beta -= self.b_reduce

    def train(self, field):
        epoches = 5000
        for _ in range(epoches):
            self.train_once(field)

    def reset_before_getWheat(self):
        Agent_normal_model.__init__(self)

    def to_continue(self, field):
        if field.k < 3:
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
            noLarger_prob -= self.weights[field.k-1]*self.normal_KL_with_now(mu1, sigma1)
        if noLarger_prob > 0.5:
            return False
        else:
            return True

class Agent_prob_KL_gain_fast_learning(Agent_prob_KL_gain_learning):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.reward = None
        self.w_size = 10
        self.weights = np.zeros((self.w_size,))
        self.lr_alpha = 0.0001
        self.a_discount = 0.9
        self.lr_beta = 0.005
        self.b_discount = 1

        self.separate_value = 1.9

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
            if (field.wheat_record[j]-self.mu)/self.sigma > self.separate_value:
                self.weights[j] = max(0, self.weights[j] - self.lr_alpha*KL_3_size[j]);
        if field.k <= self.w_size and self.reward < self.separate_value:
            self.weights[field.k-1] += self.lr_beta*KL_3_size[field.k-1]

        self.lr_alpha *= self.a_discount
        self.lr_beta *= self.b_discount

    def train(self, field):
        epoches = 5000
        for _ in range(epoches):
            self.train_once(field)

class Agent_prob_KL_gain_fast_learning_i2(Agent_prob_KL_gain_learning):
    def __init__(self):
        Agent_normal_model.__init__(self)
        # normalized reward
        self.mu_record = []
        self.sigma_record = []

        self.reward = None
        self.w_size = 10
        self.weights = np.zeros((self.w_size,))
        self.lr_alpha = 0.0001
        self.a_discount = 0.9
        self.lr_beta = 0.005
        self.b_discount = 1

        self.separate_value = 2.2

    def train_once(self, field):
        self.reset_before_getWheat()
        field.go_another_field()

        field.go_next_wheat()  # Ensure at least one wheat now
        """
        wheat_arr = np.array(field.wheat_record)
        self.mu = np.mean(wheat_arr)
        self.sigma = np.std(wheat_arr)
        if self.sigma < 0.001:
            return -1
        # not_larger_prob = (norm.cdf(wheat_list[-1], mu, np.sqrt(sigma)))**(field.N - len(wheat_list))
        not_larger_prob = norm.logcdf(field.height_of_this_wheat(), self.mu, self.sigma)
        not_larger_prob *= field.N - field.k
        return np.exp(not_larger_prob)
        """
        KL_4_size = [0,0,0]
        while field.k <= self.w_size and field.is_finished() == False:
            wheat_arr = np.array(field.wheat_record)
            self.mu = np.mean(wheat_arr)
            self.sigma = np.std(wheat_arr)

            self.mu_record.append(self.mu)
            self.sigma_record.append(self.sigma)

            decide = self.to_continue(field)

            if field.k >= 4 and field.k <= self.w_size:
                KL_4_size.append(self.normal_KL_with_now(self.mu_record[-2], self.sigma_record[-2]))
            if decide == True:
                field.go_next_wheat()
            else:
                break
        self.reward_info = field.finish_and_check()
        self.reward = self.reward_info[0]
        for j in range(min(self.w_size, field.k)):
            if (field.wheat_record[j]-self.mu)/self.sigma > self.separate_value:
                self.weights[j] = max(0, self.weights[j] - self.lr_alpha*KL_4_size[j]);
        if field.k <= self.w_size and self.reward < self.separate_value:
            self.weights[field.k-1] += self.lr_beta*KL_4_size[field.k-1]

        self.lr_alpha *= self.a_discount
        self.lr_beta *= self.b_discount

    def train(self, field):
        epoches = 5000
        for _ in range(epoches):
            self.train_once(field)

"""
model_Field.N
model_Field.k
model_Field.wheat_record
model_Field.compute_explore_rate()
model_Field.height_of_this_wheat()
"""
