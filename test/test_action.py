import numpy as np

import farmworld

env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4)
env.reset()

# action | field | trueaction
# 0 | none | None
# 1 | 0 | 0
# 2 | 0 | 1
# 3 | 1 | 0
# 4 | 1 | 1
# 5 | 2 | 0
# 6 | 2 | 1
# 7 | 3 | 0
# 8 | 3 | 1

field, action = env.deconstruct_action(0)
assert field is None
assert action is None

field, action = env.deconstruct_action(1)
assert field == env.fields[0]
assert action == 0

field, action = env.deconstruct_action(3)
assert field.idx == 1
assert action == 0

field, action = env.deconstruct_action(4)
assert field.idx == 1
assert action == 1

field, action = env.deconstruct_action(6)
assert field == env.fields[2]
assert action == 1

field, action = env.deconstruct_action(8)
assert field == env.fields[3]
assert action == 1