import wheat_field

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
    res_normalized_height = []
    for i in range(num_game):
        res_normalized_height.append(agent.get_the_wheat(normalField)[0])
    return sum(res_normalized_height) / num_game