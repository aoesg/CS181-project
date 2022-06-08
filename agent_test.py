import wheat_field
import agent
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence

def debug_field(field):
    while field.go_next_wheat():
        pass
    field.debug_hist()
    print(field.finish_and_check())

def normalFieldTest_one(agent):
    normalField = wheat_field.Normal_Field()
    return agent.get_the_wheat(normalField)

def normalFieldTest_avg(num_game, agent):
    normalField = wheat_field.Normal_Field()
    # agent.pre_train(normalField)
    res_normalized_height = []
    for i in range(num_game):
        res_normalized_height.append(agent.get_the_wheat(normalField)[0])
    return sum(res_normalized_height) / num_game

def normalFieldTest_avg_var(num_game, agent):
    normalField = wheat_field.Normal_Field()
    # agent.pre_train(normalField)
    res_normalized_height = []
    for i in range(num_game):
        res_normalized_height.append(agent.get_the_wheat(normalField)[0])
    res_normalized_height_arr = np.array(res_normalized_height)
    return res_normalized_height_arr.mean(), res_normalized_height_arr.var()

def normalField_leak_test(num_game, agent_leak):
    normalField_leak = wheat_field.Normal_Field_Leak()
    res_normalized_height = []
    for i in range(num_game):
        res_normalized_height.append(agent_leak.get_the_wheat(normalField_leak)[0])
    return sum(res_normalized_height) / num_game

def train_in_normalFieldTest(agent):
    normalField = wheat_field.Normal_Field()
    agent.train(normalField)

def normalFieldTest_record(agent, N):
    normalField = wheat_field.Normal_Field(N)
    info = agent.get_the_wheat(normalField)
    return info[1], info[2]

def mu_sigma_KL_graph():
    # draw, 麦田长度 = 200 才可以跑
    ag_prob_record = agent.Agent_prob_record()
    x = range(1, 201, 1)
    y_mu, y_sigma = normalFieldTest_record(ag_prob_record, 200)
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
    plt.savefig("mu")
    plt.show()
    # sigma graph
    plt.plot(x, y_sigma_list, label="true sigma")
    plt.plot(x, y_sigma_record, color='red', linestyle='--', label="estimated sigma")
    plt.xlabel("number of wheats")
    plt.ylabel("sigam")
    plt.legend()
    plt.savefig("sigma")
    plt.show()

    # kl graph
    kl_list = []
    x = range(4, 198)
    for i in range(4, 198):
        # print(y_mu, y_sigma, y_mu_record[i], y_sigma_record[i])
        # kl_list.append(kl_divergence(Normal(y_mu, y_sigma), Normal(y_mu_record[i], y_sigma_record[i])))
        kl_list.append(kl_divergence(Normal(y_mu_record[i + 1], y_sigma_record[i + 1]),
                                     Normal(y_mu_record[i], y_sigma_record[i])))
    plt.plot(x, kl_list)
    plt.xlabel("number of wheats")
    plt.ylabel("KL i~i+1")
    plt.savefig("KL")
    plt.show()