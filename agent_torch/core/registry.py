"""
AgentTorch 注册表模块
====================
Registry 是框架的"插件系统"。

问题：YAML config 里写的是字符串名称（如 "MovePolicy"），
      框架怎么知道该实例化哪个 Python 类？
答案：通过 Registry。用户在代码里把类注册到 Registry，
      Initializer 在构建子步骤时从 Registry 里按名字查找并实例化。

五类注册槽：
  - transition   ：SubstepTransition 的子类（状态转移）
  - observation  ：SubstepObservation 的子类（观察）
  - policy       ：SubstepAction 的子类（策略/行动）
  - initialization：状态初始化函数（如从 CSV 读数据）
  - network      ：图/网络构建函数（如生成 k-NN 图）

用法示例：
    registry = Registry()
    registry.register(MovePolicy, "MovePolicy", "policy")

    # 或使用装饰器写法（等价）
    @Registry.register_helper("MovePolicy", "policy")
    class MovePolicy(SubstepAction): ...
"""

import pandas as pd
import torch
import torch.nn as nn
import json


class Registry(nn.Module):
    # 类变量：所有 Registry 实例共享同一份注册表（全局单例语义）
    helpers = {
        "transition": {},    # 状态转移类
        "observation": {},   # 观察类
        "policy": {},        # 策略/行动类
        "initialization": {},# 状态初始化函数
        "network": {},       # 网络/图构建函数
    }

    def __init__(self):
        super().__init__()
        # 为了代码可读性，给每类注册槽起别名
        self.initialization_helpers = self.helpers["initialization"]
        self.observation_helpers = self.helpers["observation"]
        self.policy_helpers = self.helpers["policy"]
        self.transition_helpers = self.helpers["transition"]
        self.network_helpers = self.helpers["network"]

    def register(self, obj_source, name, key):
        """将类/函数 obj_source 以 name 为键注册到 key 类别的槽中。"""
        self.helpers[key][name] = obj_source

    def view(self):
        """打印当前注册表的全部内容（调试用）。"""
        return json.dumps(self.helpers, indent=2)

    def forward(self):
        print("Invoke registry.register(class_obj, key)")

    @classmethod
    def register_helper(cls, name, key):
        """装饰器写法：@Registry.register_helper("MyClass", "policy")"""
        def decorator(fn):
            cls.helpers[key][name] = fn
            return fn
        return decorator

    # register_substep 是 register_helper 的别名，语义相同
    register_substep = register_helper
