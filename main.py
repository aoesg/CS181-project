import wheat_field
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # f1 = wheat_field.normal_field()
    # while f1.go_next_wheat():
    #     pass
    # f1.debug_hist()

    fb1 = wheat_field.beta_field()
    while fb1.go_next_wheat():
        pass
    fb1.debug_hist()


