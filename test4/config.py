"""Central configuration loader for test4.

Values in config.json are the canonical configuration. Keeping this small Python
module means model.py can continue importing attributes such as cfg.vocab_size.
"""
import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _file:
    _values = json.load(_file)
globals().update(_values)


def as_dict():
    return {key: value for key, value in globals().items() if not key.startswith("_") and not callable(value)}
