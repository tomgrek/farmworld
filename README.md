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

```
poetry build
poetry publish
```

# TODO

# move plants from just being some random attribute of the field, to a more deliberate
# planting strategy, ie an action that can be taken. at least have field.plants set only when the agent
# has done the PLANT action.
# then, need to add different plants which have different maturities, weather needs etc. 
# plus weather forecast, soil quality(split into attributes)