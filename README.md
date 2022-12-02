# FarmWorld

A reinforcement learning library for agriculture.

# HOWTO

```python
pip install farmworld
```

# Install from source

```
make venv
make install
```

# Build/Publish

Put a new release on Github

```shell
poetry build
poetry publish
```

# Test

```python
PYTHONPATH=. python test/test_env.py
```

# Current Status

DQN basically solves it after 100k steps.

# TODO

* normalize the observations to -1/1. this seems to be having an effect, so continue on to normalize
the crop heights and yield, and the reward. create a generic fn to do maxscaling.
* have only 1 zero action
* then, complicate the problem!

# suspect the tests will fail now
# make env realistic -- add different plants, no "reward shaping". is the theoretical max accurate -
# does planting a crop and letting it grow 4 days score better than plant/harvest plant/harvest
# fix planting density
# add different plants which have different maturities, weather needs etc. 
# plus weather forecast, soil quality(split into attributes)