from collections import defaultdict
from typing import DefaultDict, Optional, Dict
import json


def read_file(filepath: str) -> str:

    """
    Reads a file given a filepath
    """

    with open(filepath, 'r') as f:
        data = f.read()
    return data


def read_as_dict(filepath: str) -> Dict:

    """
    Given the path to a file loads it as a dictionary
    """

    # loads the data as a dictionary
    with open(filepath, 'r') as f:
        if ".json" in filepath:
            data = json.load(f)
        else:
            try:
                data = json.loads(f.read())
            except:
                raise Exception("Did not receive a dictionary as input")

    return data


def read_as_defaultdict(filepath: str, default_val: Optional[...] = None) -> DefaultDict:

    """
    Given a path to a json file or a file with a string dictionary reads the dict as a default_dict with a given
    default value.
    """

    data = read_as_dict(filepath)

    # loops over the keys in the data and adds them to the defaultdict
    new_dict = defaultdict(lambda : default_val)
    for key in data:
        new_dict[key] = data[key]

    return new_dict
