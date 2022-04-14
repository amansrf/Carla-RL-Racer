"""
    NOTE: The Purpose of this file is to have general purpose utility functions like reading and writing to json files.
    Please DO NOT add functions here that are required for torch models. such functions must go in their own files like
    PPO_util.py or util.py
"""

# Imports for standard python libs
import numpy as np

# imports for reading and writing json config files
import json

# Load spawn parameters from the ppo_configuration file
from configurations.ppo_configuration import spawn_params


def json_read_write(file, load_var=None, mode='r'):
    """

    Args:
        file: address of json file to be loaded
        load_var: variable to be written to, or read from
        mode: 'r' to read from json, 'w' to write to json

    Returns:
        load_var: variable with data that has been read in mode 'r'
                  original variable in case of 'w'

    """
    if mode == 'r':
        with open(file, mode) as json_file:
            load_var = json.load(json_file)  # Reading the file
            print(f"{file} json config read successful")
            json_file.close()
            return load_var
    elif mode == 'w':
        assert load_var is not None, "load_var was None"
        with open(file, mode) as json_file:
            json.dump(load_var, json_file)  # Writing to the file
            print(f"{file} json config write successful")
            json_file.close()
            return load_var
    else:
        assert mode == 'w' or 'r', f"unsupported mode type: {mode}"
        return None


def next_spawn_point(curr_spawn_pt_id):

    # Type check and in case of error assign safe value
    if not isinstance(curr_spawn_pt_id, int):
        curr_spawn_pt_id = spawn_params["init_spawn_pt"]
        print("[WARNING] e2e_Model_roar_env.py: curr_spawn_pt_id was not an int, setting to init_spawn_pt for safety")

    # Load Spawn Params
    num_spawn_pts = spawn_params["num_spawn_pts"]
    init_spawn_pt = spawn_params["init_spawn_pt"]
    dynamic = spawn_params["dynamic_spawn"]
    mode = spawn_params["dynamic_type"]
    spawn_list = spawn_params["custom_list"]
    spawn_pt_iterator = spawn_params["spawn_pt_iterator"]

    # Check if dynamic
    if not dynamic:
        return curr_spawn_pt_id
    else:
        if mode == "uniform random":
            # return from uniform random distribution
            return np.random.randint(low=init_spawn_pt, high=num_spawn_pts)
        elif mode == "linear forward":
            # After reset spawn point increments by one. Loops back to init after num_spawn_pts reached
            if curr_spawn_pt_id < num_spawn_pts - 1:
                return curr_spawn_pt_id + 1
            else:
                return init_spawn_pt
        elif mode == "linear backward":
            if curr_spawn_pt_id > init_spawn_pt:
                return curr_spawn_pt_id - 1
            else:
                return num_spawn_pts - 1
        elif mode == "custom spawn pts":
            assert spawn_list[0] == init_spawn_pt,\
                "[ASSERTION ERROR] utility.py: spawn list first point is not equal to init_spawn_pt"
            next_spawn = spawn_pt_iterator
            spawn_pt_iterator += 1
            spawn_pt_iterator = spawn_pt_iterator % len(spawn_list)
            spawn_params["spawn_pt_iterator"] += 1
            spawn_params["spawn_pt_iterator"] = spawn_params["spawn_pt_iterator"] % len(spawn_list)
            return next_spawn
        else:
            # TODO; Implement other mode functions in elif blocks. For now return curr_spawn_pt_id
            return curr_spawn_pt_id


