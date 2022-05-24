import random
import numpy as np
import matplotlib.pyplot as plt

class field():
    def __init__(self, N = 1000):
        self.N = N
        self.k = 0
        self.wheat_record = []

        self.__mean = random.uniform(0, 500)
        self.__std = random.uniform(0.1, 4)
        self.__finished = False

    def __gen_wheat(self):
        return np.random.normal(self.__mean, self.__std)

    def go_next_wheat(self):
        if self.__finished == True:
            print("You have made your choice!")
            return None
        elif self.k == self.N:
            self.__finished = True
            print("You reach the end of field.")
            return None
        else:
            new_wheat = self.__gen_wheat()
            self.wheat_record.append(new_wheat)
            self.k += 1
            return new_wheat

    def is_finished(self):
        return self.__finished

    def __reward_normalize(self, reward):
        return (reward - self.__mean) / self.__std

    def __check_info(self, normazlized_R):
        return (normazlized_R, self.__mean, self.__std)

    def reward_check(self):
        if self.k == 0:
            print("You haven't ever entered the field!")
            return None

        self.__finished = True

        choosen = self.wheat_record[self.k - 1]
        normazlized_res = self.__reward_normalize(choosen)
        return self.__check_info(normazlized_res)

    def debug_normalize(self):
        plt.hist(self.__reward_normalize(np.array(self.wheat_record)), bins=10, rwidth=0.8, density=True)
        plt.title('normalized_distribution')
        plt.show()



