import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np
import agent_test
import time
from scipy import stats

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
    ag_prob_decision_5 = agent.Agent_prob_decision_10()
    ag_prob_decision_10 = agent.Agent_prob_decision_10()
    ag_prob_decision_20 = agent.Agent_prob_decision_20()
    ag_prob_decision_d9 = agent.Agent_prob_decision_d9()

    ag_prob_rand = agent.Agent_prob_rand()
    ag_prob_rand_10 = agent.Agent_prob_rand_10()

    ag_prob_decision_leak = agent.Agent_prob_decision_leak()

    ag_prob_gain_learning = agent.Agent_prob_gain_learning()
    ag_prob_KL_gain_learning = agent.Agent_prob_KL_gain_learning()
    ag_prob_KL_gain_fast_learning = agent.Agent_prob_KL_gain_fast_learning()

    num_game = 8000

    # print("ag_random, avg =", agent_test.normalFieldTest_avg(num_game,ag_random))
    # print("ag_upper_bound, avg =", agent_test.normalFieldTest_avg(num_game, ag_upper_bound))
    # print("ag_37, avg =", agent_test.normalFieldTest_avg(num_game,ag_37))
    # print("ag_37_t3, avg =", agent_test.normalFieldTest_avg(num_game, ag_37_t3))
    # print("ag_sqrt_n, avg =", agent_test.normalFieldTest_avg(num_game, ag_sqrt_n))

    # agent_test.train_in_normalFieldTest(ag_threshold_learning)
    # print("ag_threshold_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_threshold_learning))
    # print(ag_threshold_learning.threshold)

    # print("ag_prob_decision, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision))
    print("ag_prob_decision_5, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_5))
    # print("ag_prob_decision_10, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_10))
    # print("ag_prob_decision_20, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_20))
    # print("ag_prob_decision_d9, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_d9))

    # print("ag_prob_rand, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_rand))
    # print("ag_prob_rand_10, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_rand_10))

    # print("ag_prob_decision_leak, avg =", agent_test.normalField_leak_test(num_game, ag_prob_decision_leak))

    # agent_test.train_in_normalFieldTest(ag_prob_gain_learning)
    # print("ag_prob_gain_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_gain_learning))
    # print(ag_prob_gain_learning.weights)
    agent_test.train_in_normalFieldTest(ag_prob_KL_gain_learning)
    print("ag_prob_KL_gain_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_KL_gain_learning))
    print(ag_prob_KL_gain_learning.weights)
    agent_test.train_in_normalFieldTest(ag_prob_KL_gain_fast_learning)
    print("ag_prob_KL_gain_fast_learning, avg =", agent_test.normalFieldTest_avg(num_game, ag_prob_KL_gain_fast_learning))
    print(ag_prob_KL_gain_fast_learning.weights)