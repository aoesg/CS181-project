import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Field():
    def __init__(self, N = 1000):
        self.N = N
        self.k = 0
        self.wheat_record = []
        self.__finished = False

    def gen_wheat(self):
        pass

    def reward_normalize(self, reward):
        pass

    def check_info(self, normazlized_R):
        pass

    def go_another_field(self):
        self.k = 0
        self.wheat_record = []
        self.__finished = False

    def debug_hist(self):
        pass

    def go_next_wheat(self):
        if self.__finished == True:
            print("You have made your choice!")
            return None
        elif self.k == self.N:
            self.__finished = True
            print("You reach the end of field.")
            return None
        else:
            new_wheat = self.gen_wheat()
            self.wheat_record.append(new_wheat)
            self.k += 1
            return new_wheat

    def is_finished(self):
        return self.__finished

    def compute_explore_rate(self):
        return self.k / self.N

    def height_of_this_wheat(self):
        if self.k == 0:
            print("You haven't ever entered the field!")
            return
        return self.wheat_record[-1]

    def finish_and_check(self):
        if self.k == 0:
            print("You haven't ever entered the field!")
            return None

        self.__finished = True

        choosen = self.wheat_record[self.k - 1]
        normazlized_res = self.reward_normalize(choosen)
        return self.check_info(normazlized_res)

class Normal_Field(Field):
    def __init__(self, N = 1000):
        Field.__init__(self, N)

        self.__mean = random.uniform(0, 500)
        self.__std = random.uniform(0.1, 4)

    def gen_wheat(self):
        return np.random.normal(self.__mean, self.__std)

    def reward_normalize(self, reward):
        return (reward - self.__mean) / self.__std

    def check_info(self, normazlized_R):
        return (normazlized_R, self.__mean, self.__std)

    def debug_hist(self):
        plt.hist(self.reward_normalize(np.array(self.wheat_record)), bins=10, rwidth=0.8, density=True)
        plt.title('Normalized'.format(round(self.__mean,2), round(self.__std,2)))
        plt.show()

class Beta_Field(Field):
    def __init__(self, N=1000):
        Field.__init__(self, N)

        self.__scale = random.uniform(10, 500)
        self.__scale = 1
        self.__alpha = random.uniform(1, 100)
        self.__beta = random.uniform(1, 100)

    def gen_wheat(self):
        return stats.beta.rvs(self.__alpha, self.__beta) * self.__scale

    def reward_normalize(self, reward):
        return reward

    def check_info(self, normazlized_R):
        return normazlized_R

    def debug_hist(self):
        plt.hist(self.reward_normalize(np.array(self.wheat_record)), bins=10, rwidth=0.8, density=True)
        plt.title('Beta({0},{1}), scale={2}'.format(round(self.__alpha, 2), round(self.__beta, 2),
                                                 round(self.__scale, 2)))
        plt.show()