import wheat_field
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    f1 = wheat_field.field()

    while f1.go_next_wheat():
        pass

    nor_list = []
    plt.hist(f1.reward_normalize(np.array(f1.wheat_record)), bins=10, rwidth=0.8, density=True)
    plt.title('distribution')
    plt.show()


