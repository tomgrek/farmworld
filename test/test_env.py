import numpy as np

import farmworld

env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4)

env.reset()
obs, reward, done, info = env.step([1, 1])
assert obs["crop_height"][0] == -1
assert obs["crop_height"][1] == 0

obs, reward, done, info = env.step([1, 1])
assert done

env.reset()

obs, reward, done, info = env.step([3, 1])
assert obs["crop_height"][1] == -1
assert obs["crop_height"][3] == 0

obs, reward, done, info = env.step([3, 1])
assert done

env.reset()

obs, reward, done, info = env.step([0, 1])
assert obs["crop_height"][2] == -1
assert obs["crop_height"][0] == 0

for _ in range(99):
    assert not done
    assert env.total_rewards == 0
    obs, reward, done, info = env.step([0, 0])
assert done
assert obs["crop_height"][0] == 99
assert obs["crop_height"][1] == -1
assert env.day_of_year == 100
assert env.inaction_penalty == 99
assert reward < 0.5 # actually correct but leaving it in as I reward shape

obs = env.reset()
obs, reward, done, info = env.step([0, 1])
assert reward == 0
assert not done
obs, reward, done, info = env.step([0, 0])
assert reward == 0
assert not done
obs, reward, done, info = env.step([0, 0])
assert reward == 0
assert not done
obs, reward, done, info = env.step([0, 2])
assert reward == 0
assert not done
obs, reward, done, info = env.step([0, 1])
assert reward == 0
assert not done
assert env.total_rewards == 3
assert env.day_of_year == 5

obs = env.reset()
obs, reward, done, info = env.step([0, 1])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 2])
obs, reward, done, info = env.step([0, 2])
assert done
assert env.total_rewards == 3
assert env.day_of_year == 5
assert reward < 1

obs = env.reset()
obs, reward, done, info = env.step([0, 1])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 2])
for _ in range(96):
    assert not done
    assert env.total_rewards == 3
    obs, reward, done, info = env.step([0, 0])
assert done
assert env.total_rewards == 3
assert env.day_of_year == 100
assert 1 < reward < 2

obs = env.reset()
obs, reward, done, info = env.step([0, 1])
obs, reward, done, info = env.step([1, 1])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 2])
assert env.total_rewards == 4
obs, reward, done, info = env.step([1, 2])
assert env.total_rewards == 8
for _ in range(94):
    assert not done
    assert env.total_rewards == 8
    obs, reward, done, info = env.step([0, 0])
assert done
assert env.total_rewards == 8
assert env.day_of_year == 100
assert 3 < reward < 5

obs = env.reset()
obs, reward, done, info = env.step([2, 1])
obs, reward, done, info = env.step([3, 1])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([2, 2])
assert env.total_rewards == 4
obs, reward, done, info = env.step([3, 2])
assert env.total_rewards == 8
for _ in range(94):
    assert not done
    assert env.total_rewards == 8
    obs, reward, done, info = env.step([0, 0])
assert done
assert env.total_rewards == 8
assert env.day_of_year == 100
assert 3 < reward < 5

obs = env.reset()
obs, reward, done, info = env.step([0, 1])
obs, reward, done, info = env.step([1, 1])
obs, reward, done, info = env.step([2, 1])
obs, reward, done, info = env.step([3, 1])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([0, 0])
obs, reward, done, info = env.step([1, 2])
obs, reward, done, info = env.step([2, 2])
obs, reward, done, info = env.step([3, 2])
assert env.total_rewards == 15
for _ in range(91):
    assert not done
    assert env.total_rewards == 15
    obs, reward, done, info = env.step([0, 0])
assert done
assert env.total_rewards == 15
assert env.day_of_year == 100
print(reward)
assert 7 < reward < 8