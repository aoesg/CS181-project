import wheat_field
import agent
import matplotlib.pyplot as plt
import numpy as np
import agent_test

if __name__ == '__main__':
    nf1 = wheat_field.Normal_Field()
    # agent_test.debug_field(nf1)

    ag_37 = agent.Agent_37()
    ag_37_t3 = agent.Agent_37_t3()
    ag_sqrt_n = agent.Agent_sqrt_n()
    ag_sqrt_n_t2 = agent.Agent_sqrt_n_t2()

    print("ag_37,avg=", agent_test.normalFieldTest_avg(100,ag_37))
    print("ag_37_t3,avg=", agent_test.normalFieldTest_avg(100, ag_37_t3))
    print("ag_sqrt_n,avg=", agent_test.normalFieldTest_avg(100, ag_sqrt_n))
    print("ag_sqrt_n_t2,avg=", agent_test.normalFieldTest_avg(100, ag_sqrt_n_t2))




