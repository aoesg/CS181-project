import wheat_field
import agent

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

def normalField_leak_test(num_game, agent_leak):
    normalField_leak = wheat_field.Normal_Field_Leak()
    res_normalized_height = []
    for i in range(num_game):
        res_normalized_height.append(agent_leak.get_the_wheat(normalField_leak)[0])
    return sum(res_normalized_height) / num_game

def train_in_normalFieldTest(agent):
    normalField = wheat_field.Normal_Field()
    agent.train(normalField)