# FarmWorld

A reinforcement learning library for agriculture.

# HOWTO

```python
pip install farmworld
```

# Install from source

```shell
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

```shell
make tests
```

# Current Status

DQN basically solves it after 100k steps.

* Normalized the easy way using vecnormalize.
* Added a zeroth action and trimmed the action space a bit

# TODO

* complicate the problem! multiple crops, and they need to start dieing off at some point

# make env realistic -- add different plants
# fix planting density
# add different plants which have different maturities, weather needs etc. 
# plus weather forecast, soil quality(split into attributes)
