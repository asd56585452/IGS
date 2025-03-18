import os
import re

import torch
from packaging import version

from igs.utils.typing import *


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load

def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, '_fields')

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, *args, **kwargs):
        # print(type(vars), is_namedtuple(vars))
        if isinstance(vars, list):
            return [wrapper(x, *args, **kwargs) for x in vars]
        elif isinstance(vars, tuple) and not is_namedtuple(vars):
            return tuple([wrapper(x, *args, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in vars.items()}
        elif is_namedtuple(vars):

            namedtuple_type = type(vars)
            # 递归处理 namedtuple 的每个字段
            processed_fields = [wrapper(x, *args, **kwargs) for x in vars]
            # 使用实际类型创建新的 namedtuple 实例
            return namedtuple_type(*processed_fields)
            # return vars.to("cuda")
        # elif isinstance(vars, GaussianModel):
        #     return GaussianModel(wrapper(x, *args, **kwargs) for x in vars)
        else:
            return func(vars, *args, **kwargs)

    return wrapper

@make_recursive_func
def todevice(vars, device="cuda"):
    # print("todevice",type(vars))
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, bool):
        return vars
    elif isinstance(vars, float):
        return vars
    elif isinstance(vars, int):
        return vars
    elif vars == None:
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
