import wheat_field
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    fn1 = wheat_field.normal_field()
    while fn1.go_next_wheat():
        pass
    fn1.debug_hist()
    print(fn1.finish_and_check())

    # fb1 = wheat_field.beta_field()
    # while fb1.go_next_wheat():
    #     pass
    # fb1.debug_hist()
    # print(fb1.finish_and_check())


