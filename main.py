import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np

IM = 10

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

    nf1 = wheat_field.Normal_Field()
    ag1 = agent.Agent_37()

    print(ag1.get_the_wheat(nf1))

