import numpy as np

import farmworld

env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4)
env.reset()

field, action = env.deconstruct_action(1)
assert field == env.fields[0]
assert action == 1

field, action = env.deconstruct_action(4)
assert field.idx == 1
assert action == 1

field, action = env.deconstruct_action(6)
assert field == env.fields[2]
assert action == 0

field, action = env.deconstruct_action(11)
assert field == env.fields[3]
assert action == 2