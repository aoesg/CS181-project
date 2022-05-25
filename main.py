import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np

IM = 13

if __name__ == '__main__':
    if IM == 0:
        fn1 = wheat_field.Normal_Field()
        while fn1.go_next_wheat():
            pass
        fn1.debug_hist()
        print(fn1.finish_and_check())
    if IM == 1:
        fb1 = wheat_field.Beta_Field()
        while fb1.go_next_wheat():
            pass
        fb1.debug_hist()
        print(fb1.finish_and_check())

    if IM == 11:
        nf1 = wheat_field.Normal_Field()
        ag1 = agent.Agent_37()
        print(ag1.get_the_wheat(nf1))
    if IM == 12:
        bf1 = wheat_field.Beta_Field()
        ag1 = agent.Agent_37()
        print(ag1.get_the_wheat(bf1))
        bf1.debug_hist()
    if IM == 13:
        num_game = 100
        nf1 = wheat_field.Normal_Field()
        ag1 = agent.Agent_37()
        ag2 = agent.Agent_37_t3()

        res_normalized_height_1 = []
        for i in range(num_game):
            res_normalized_height_1.append(ag1.get_the_wheat(nf1)[0])
        res_normalized_height_2 = []
        for i in range(num_game):
            res_normalized_height_2.append(ag2.get_the_wheat(nf1)[0])
        print("ag1,avg = ", sum(res_normalized_height_1) / num_game)
        print("ag2,avg = ", sum(res_normalized_height_2) / num_game)

