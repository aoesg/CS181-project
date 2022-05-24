import wheat_field
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    f1 = wheat_field.field()

    while f1.go_next_wheat():
        pass

    f1.debug_normalize()


