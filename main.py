import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np
import agent_test
import time
from scipy import stats
from torch.distributions import Normal, kl_divergence

if __name__ =='__main__':
    nf1 =wheat_field.Normal_Field()
    # agent_test.debug_field(nf1)

    ag_random = agent.Agent_random()
    ag_upper_bound = agent.Agent_upper_bound()
    ag_37 = agent.Agent_37()
    ag_37_t3 = agent.Agent_37_t3()
    ag_sqrt_n = agent.Agent_sqrt_n()

    ag_threshold_learning = agent.Agent_threshold_learning()

    ag_prob_decision = agent.Agent_prob_decision()
    ag_prob_decision_3 = agent.Agent_prob_decision_3()
    ag_prob_decision_5 = agent.Agent_prob_decision_5()
    ag_prob_decision_10 = agent.Agent_prob_decision_10()
    ag_prob_decision_20 = agent.Agent_prob_decision_20()
    ag_prob_decision_d9 = agent.Agent_prob_decision_d9()

    ag_prob_rand = agent.Agent_prob_rand()
    ag_prob_rand_10 = agent.Agent_prob_rand_10()

    ag_prob_record = agent.Agent_prob_record()

    ag_prob_decision_leak = agent.Agent_prob_decision_leak()

    ag_prob_gain_learning = agent.Agent_prob_gain_learning()
    ag_prob_KL_gain_learning = agent.Agent_prob_KL_gain_learning()
    ag_prob_KL_gain_fast_learning = agent.Agent_prob_KL_gain_fast_learning()

    num_game = 100

    # print("ag_random, avg =", agent_test.normalFieldTest_avg(num_game,ag_random))
    # print("ag_upper_bound, avg =", agent_test.normalFieldTest_avg(num_game, ag_upper_bound))
    # print("ag_37, avg =", agent_test.normalFieldTest_avg(num_game,ag_37))
    # print("ag_37_t3, avg =", agent_test.normalFieldTest_avg(num_game, ag_37_t3))
    # print("ag_sqrt_n, avg =", agent_test.normalFieldTest_avg(num_game, ag_sqrt_n))

    # agent_test.train_in_normalFieldTest(ag_threshold_learning)
    # print("ag_threshold_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_threshold_learning))
    # print(ag_threshold_learning.threshold)

    # print("ag_prob_decision, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision))
    # print("ag_prob_decision_3, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_3))
    # print("ag_prob_decision_5, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_5))
    # print("ag_prob_decision_10, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_10))
    # print("ag_prob_decision_20, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_20))
    # print("ag_prob_decision_d9, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_d9))

    # print("ag_prob_rand, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_rand))
    # print("ag_prob_rand_10, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_rand_10))

    # print("ag_prob_decision_leak, avg =", agent_test.normalField_leak_test(num_game, ag_prob_decision_leak))

    # agent_test.train_in_normalFieldTest(ag_prob_gain_learning)
    # print("ag_prob_gain_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_gain_learning))
    # print(ag_prob_gain_learning.weights)

    # agent_test.train_in_normalFieldTest(ag_prob_KL_gain_learning)
    # print("ag_prob_KL_gain_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_KL_gain_learning))
    # print(ag_prob_KL_gain_learning.weights)

    # agent_test.train_in_normalFieldTest(ag_prob_KL_gain_fast_learning)
    # print("ag_prob_KL_gain_fast_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_KL_gain_fast_learning))
    # print(ag_prob_KL_gain_fast_learning.weights)

    # print("Normal_Field, True mu, sigma =", agent_test.normalFieldTest_record(ag_prob_record))
    # print(ag_prob_record.mu_record)
    # print(ag_prob_record.sigma_record)
    # print(len(ag_prob_record.mu_record))
    # print(len(ag_prob_record.sigma_record))

    # draw
    x = range(1, 201, 1)
    y_mu, y_sigma = agent_test.normalFieldTest_record(ag_prob_record)
    y_mu_list = []
    y_sigma_list = []
    for i in range(200):
        y_mu_list.append(y_mu)
        y_sigma_list.append(y_sigma)
    y_mu_record = ag_prob_record.mu_record
    y_sigma_record = ag_prob_record.sigma_record
    print(len(y_mu_record))
    print(len(y_sigma_record))
    # mu graph
    plt.plot(x, y_mu_list, label="true mu")
    plt.plot(x, y_mu_record, color='red', linestyle='--', label="estimated mu")
    plt.xlabel("number of wheats")
    plt.ylabel("mu")
    plt.legend()
    plt.show()
    # sigma graph
    plt.plot(x, y_sigma_list, label="true sigma")
    plt.plot(x, y_sigma_record, color='red', linestyle='--', label="estimated sigma")
    plt.xlabel("number of wheats")
    plt.ylabel("sigam")
    plt.legend()
    plt.show()
    # kl graph
    kl_list = []
    x = range(1, 200)
    for i in range(1, 200):
        # print(y_mu, y_sigma, y_mu_record[i], y_sigma_record[i])
        kl_list.append(kl_divergence(Normal(y_mu, y_sigma), Normal(y_mu_record[i], y_sigma_record[i])))
    plt.plot(x, kl_list)
    plt.xlabel("number of wheats")
    plt.ylabel("kl")
    plt.show()