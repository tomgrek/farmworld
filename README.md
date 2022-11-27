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

# TODO

# move plants from just being some random attribute of the field, to a more deliberate
# planting strategy, ie an action that can be taken. at least have field.plants set only when the agent
# has done the PLANT action. (mostly done, consider density etc)
# episode should be DONE when all harvests have taken place (at least in current setup)
# then, need to add different plants which have different maturities, weather needs etc. 
# plus weather forecast, soil quality(split into attributes)