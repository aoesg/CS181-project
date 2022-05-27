import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np
import agent_test
import time

if __name__ == '__main__':
    nf1 = wheat_field.Normal_Field()
    # agent_test.debug_field(nf1)

    ag_random = agent.Agent_random()
    ag_37 = agent.Agent_37()
    ag_37_t3 = agent.Agent_37_t3()
    ag_sqrt_n = agent.Agent_sqrt_n()
    ag_prob_decision = agent.Agent_prob_decision()
    ag_prob = agent.Agent_prob()
    ag_prob_decision_37 = agent.Agent_prob_decision_37()
    ag_prob_decision_former = agent.Agent_prob_decision_former()

    num_game = 100

    # print("ag_random,avg=", agent_test.normalFieldTest_avg(num_game,ag_random))
    # print("ag_37,avg=", agent_test.normalFieldTest_avg(num_game,ag_37))
    # print("ag_37_t3,avg=", agent_test.normalFieldTest_avg(num_game, ag_37_t3))
    # print("ag_sqrt_n,avg=", agent_test.normalFieldTest_avg(num_game, ag_sqrt_n))
    print("ag_prob_decision,avg=", agent_test.normalFieldTest_avg(num_game, ag_prob_decision))
    # print("ag_prob,avg=", agent_test.normalFieldTest_avg(num_game, ag_prob))
    # print("ag_prob_decision_37,avg=", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_37))
    # print("ag_prob_decision_former,avg=", agent_test.normalFieldTest_avg(num_game, ag_prob_decision_former))