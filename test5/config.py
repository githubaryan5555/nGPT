"""Central configuration loader for test4."""

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as file:
    _values = json.load(file)

globals().update(_values)


def as_dict():
    """Return only values loaded from config.json.

    This deliberately avoids scanning module globals because globals also contain
    imported modules such as json and os, which cannot be pickled by torch.save().
    """
    return dict(_values)
