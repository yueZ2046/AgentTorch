"""
AgentTorch 通用工具函数
=======================
这里是整个框架最常用的底层工具，几乎所有 substep 文件都会用到。

最重要的两个函数：
  get_var(state, "environment/vitality")
      → 按路径读 state 中的值，等价于 state["environment"]["vitality"]
  set_by_path(state, ["environment", "vitality"], new_tensor)
      → 按路径写 state 中的值

路径约定：用 "/" 分隔嵌套层级，与 YAML config 中的变量路径格式一致。
"""

import re
from functools import reduce
import operator
import torch
from torch import nn
import copy
from omegaconf import OmegaConf
import pandas as pd


def get_by_path(root, items):
    """按路径列表递归读取嵌套 dict/ModuleDict 中的值。

    参数：
        root : 根对象（通常是 state 字典）
        items: 路径列表，如 ["environment", "vitality"]

    注意：如果遇到 nn.Module（非 ModuleDict），会调用其 forward()。
    """
    property_obj = reduce(operator.getitem, items, root)
    if isinstance(property_obj, nn.ModuleDict):
        return property_obj
    elif isinstance(property_obj, nn.Module):
        return property_obj()
    else:
        return property_obj


def get_var(state, var):
    """substep 代码中读取 state 变量的标准方式。

    用法：
        features = get_var(state, self.input_variables["block_features"])
        # 等价于 state["environment"]["block_features"]
        # 其中 input_variables["block_features"] = "environment/block_features"
    """
    return get_by_path(state, re.split("/", var))


def set_by_path(root, items, value):
    """按路径列表递归写入嵌套 dict 中的值。

    由 Controller.progress() 调用，将 Transition 的返回值写回 state。
    注意：对 nn.ModuleDict 写入会断开梯度（框架已知限制）。
    """
    val_obj = get_by_path(root, items[:-1])

    if isinstance(val_obj, nn.ModuleDict):
        # 已知问题：通过 ModuleDict 写入会破坏梯度图
        print("set_by_path on nn.ModuleDict breaks gradient currently!")
        val_obj[items[-1]].param.data.copy_(value)
        val_obj[items[-1]].param.requires_grad = value.requires_grad
    else:
        val_obj[items[-1]] = value
        return root


def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]


def copy_module(dict_to_copy):
    r"""
    Creates a new dictionary with a copy of each PyTorch tensor in the input dictionary.
    Handles nested dictionaries of PyTorch tensors of variable depth.
    """
    copied_dict = {}
    for key, value in dict_to_copy.items():
        if torch.is_tensor(value):
            copied_dict[key] = torch.clone(value)
        elif isinstance(value, dict):
            copied_dict[key] = copy_module(value)
        elif not torch.is_tensor(value):
            copied_dict[key] = copy.deepcopy(value)
        else:
            raise TypeError("Type error.. ", type(value))

    return copied_dict


def to_cpu(dict_to_copy):
    r"""
    Creates a new dictionary with a copy of each PyTorch tensor in the input dictionary.
    Handles nested dictionaries of PyTorch tensors of variable depth.
    """
    copied_dict = {}
    for key, value in dict_to_copy.items():
        # value = dict_to_copy[key]
        if torch.is_tensor(value):
            copied_dict[key] = torch.clone(value).cpu()
        elif isinstance(value, dict):
            copied_dict[key] = to_cpu(value)
        elif not torch.is_tensor(value):
            copied_dict[key] = value
        else:
            raise TypeError("Type error.. ", type(value))

    del dict_to_copy

    return copied_dict


def process_shape(config, s):
    if type(s) == str:
        return get_by_path(config, re.split("/", s))
    else:
        return s


def register_resolver(name, resolver):
    OmegaConf.register_new_resolver(name, resolver)


def read_config(config_file, register_resolvers=True):
    if register_resolvers:
        resolvers = [
            ("sum", lambda x, y: x + y),
            ("multiply", lambda x, y: x * y),
            ("divide", lambda x, y: x // y),
        ]

        for name, func in resolvers:
            try:
                OmegaConf.register_new_resolver(name, func)
            except AssertionError as e:
                if "is already registered" in str(e):
                    continue
                else:
                    raise e

    if config_file[-5:] != ".yaml":
        raise ValueError("Config file type should be yaml")

    try:
        config = OmegaConf.load(config_file)
        config = OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(
            f"Could not load config file. Please check path and file type. Error message is {str(e)}"
        )

    return config


def read_from_file(shape, params):
    file_path = params["file_path"]

    if file_path[-3:] == "csv":
        data = pd.read_csv(file_path)

    data_values = data.values
    assert data_values.shape == tuple(shape)

    data_tensor = torch.from_numpy(data_values)

    return data_tensor


def memory_checkpoint(name):
    print("Checkpoint: ", name)
    checkpoint_allocated = torch.cuda.memory_allocated()
    checkpoint_reserved = torch.cuda.memory_reserved()

    print("Allocated: ", checkpoint_allocated, " Reserved: ", checkpoint_reserved)

    return checkpoint_allocated, checkpoint_reserved
