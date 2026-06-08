# backend.py — LLM 后端抽象层
#
# 职责：定义统一的 LLM 调用接口，屏蔽不同 LLM 框架（DSPy、直接调用等）的实现细节。
#
# 设计原则：
#   - LLMBackend 是抽象基类（ABC），所有后端必须实现 prompt() 方法
#   - prompt() 接受 prompt_list，返回等长的 output_list，每项为 {"text": str}
#   - 子类通过继承扩展；测试时使用 MockLLM（见 mock_llm.py），无需真实 API 调用
#
# 支持的后端：
#   MockLLM  — 随机返回数值，用于单元测试（mock_llm.py）
#   DspyLLM  — 基于 DSPy 的链式思维（CoT）推理，当前默认使用 OpenAI 模型
#
# 扩展方法：继承 LLMBackend，实现 prompt()，并在 Archetype 初始化时注入即可

import os
import sys
import io
import concurrent.futures
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """所有 LLM 后端的抽象基类。

    子类必须实现 prompt() 方法，其余方法提供默认的 NotImplementedError 实现。
    """

    def __init__(self):
        pass

    def initialize_llm(self):
        """初始化 LLM 客户端（如加载模型权重、配置 API Key 等）。

        子类按需覆写；若不需要显式初始化步骤可不实现。
        """
        raise NotImplementedError

    @abstractmethod
    def prompt(self, prompt_list):
        """向 LLM 发送一批 prompt，返回等长的输出列表。

        Args:
            prompt_list: prompt 列表，每项可以是：
                - str: 纯文本 prompt
                - dict: {"agent_query": str, "chat_history": list}
                  （携带对话历史，用于多轮交互场景）

        Returns:
            输出列表，长度与 prompt_list 相同。
            每项为 dict，至少包含 {"text": str}，text 应可转换为 float。
        """
        pass

    def inspect_history(self, last_k, file_dir):
        """查看最近 k 次 LLM 调用历史，并可选写入文件（用于调试）。

        子类如有对话历史追踪能力，可覆写此方法。
        """
        raise NotImplementedError


class DspyLLM(LLMBackend):
    """基于 DSPy 的 LLM 后端，使用链式思维（Chain-of-Thought）进行结构化推理。

    DSPy 将提示工程封装为可组合的模块（Signature + Module），
    此类通过 ChainOfThought 模块调用 OpenAI 模型，并以线程池并发处理 prompt。

    Args:
        openai_api_key: OpenAI API Key
        qa:             DSPy Signature 类，定义输入/输出字段（如 question -> answer）
        cot:            DSPy Module 类（通常是 dspy.ChainOfThought）
        model:          模型名称，默认 "gpt-4o-mini"
    """

    def __init__(self, openai_api_key, qa, cot, model="gpt-4o-mini"):
        super().__init__()
        self.qa = qa
        self.cot = cot
        self.backend = "dspy"
        self.openai_api_key = openai_api_key
        self.model = model

    def initialize_llm(self):
        """配置 DSPy 全局 LM，并实例化 CoT predictor。

        必须在首次调用 prompt() 前手动调用一次（或由 Archetype 自动触发）。
        """
        import dspy
        self.llm = dspy.OpenAI(
            model=self.model, api_key=self.openai_api_key, temperature=0.0
        )
        # 全局绑定 LM，后续所有 dspy 模块调用均使用此设置
        dspy.settings.configure(lm=self.llm)
        # 将 Signature(qa) 包装成 CoT predictor
        self.predictor = self.cot(self.qa)
        return self.predictor

    def prompt(self, prompt_list):
        """并发调用 DSPy CoT predictor，返回每条 prompt 的文本答案。"""
        agent_outputs = self.call_dspy_agent(prompt_list)
        return agent_outputs

    def call_dspy_agent(self, prompt_inputs):
        """使用 ThreadPoolExecutor 并发处理 prompt 列表。

        并发度由 Python 默认线程数决定（通常为 CPU 核数 × 5）。
        LLM 调用是 I/O 密集型，线程级并发足以提升吞吐量。
        """
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(
                    executor.map(self.dspy_query_and_get_answer, prompt_inputs)
                )
        except Exception as e:
            print(e)
        return agent_outputs

    def dspy_query_and_get_answer(self, prompt_input):
        """处理单条 prompt，返回标准化的 {"text": answer} 格式。

        兼容两种输入格式：
          - str：直接作为 query，history 为空
          - dict：提取 "agent_query" 和 "chat_history" 字段
        """
        if type(prompt_input) is str:
            agent_output = self.query_agent(prompt_input, [])
        else:
            agent_output = self.query_agent(
                prompt_input["agent_query"], prompt_input["chat_history"]
            )
        return {"text": agent_output}

    def query_agent(self, query, history):
        """调用 DSPy CoT predictor，返回 answer 字段的字符串值。"""
        pred = self.predictor(question=query, history=history)
        return pred.answer

    def inspect_history(self, last_k, file_dir):
        """将最近 last_k 次 DSPy 调用历史捕获并写入 inspect_history.md。

        实现原理：临时劫持 stdout，调用 dspy LM 的 inspect_history() 方法，
        再将输出写入文件并恢复 stdout。
        """
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        self.llm.inspect_history(last_k)
        printed_data = buffer.getvalue()
        if file_dir is not None:
            save_path = os.path.join(file_dir, "inspect_history.md")
            with open(save_path, "w") as f:
                f.write(printed_data)
        sys.stdout = original_stdout
