# -*- coding: utf-8 -*-
"""多条接收臂的批量前向与 CUDA Graph 回放。

只改变执行方式，不改变输入、信息边界或接收状态规则。批量化把同帧各臂的
一次前向合并为 batch=N；CUDA Graph 在固定形状和固定标量输入下复用启动序列。
任何图化失败都应显式报错，而不是静默退回不同语义的路径。
"""

from __future__ import annotations

from dataclasses import fields, replace

import torch

from agent.cir_estimator import CIRCondition


def _join(values):
    """沿 batch 维合并；一维张量没有 batch 维，需用 stack。"""

    if len(values) == 1:
        return values[0].clone()
    if values[0].ndim == 1:
        return torch.stack(values, dim=0)
    return torch.cat(values, dim=0)


def _merge_condition(conditions):
    return replace(conditions[0], **{
        field.name: _join([getattr(item, field.name) for item in conditions])
        for field in fields(CIRCondition)
        if torch.is_tensor(getattr(conditions[0], field.name))
    })


def merge_calls(calls):
    """把多条臂的 ``(args, kwargs)`` 合并为一次 batch 前向。"""

    if not calls:
        raise ValueError('批量前向至少需要一条臂。')
    args_tuple, kwargs_dict = calls[0]
    for args, kwargs in calls[1:]:
        if len(args) != len(args_tuple) or set(kwargs) != set(kwargs_dict):
            raise ValueError('待合并前向的输入结构不一致。')
    merged_args = []
    for index, value in enumerate(args_tuple):
        values = [call[0][index] for call in calls]
        if torch.is_tensor(value):
            merged_args.append(_join(values))
        elif isinstance(value, CIRCondition):
            merged_args.append(_merge_condition(values))
        else:
            if any(item != value for item in values):
                raise ValueError('非张量位置参数在臂间不一致，无法合并。')
            merged_args.append(value)
    merged_kwargs = {}
    for key, value in kwargs_dict.items():
        values = [call[1][key] for call in calls]
        if torch.is_tensor(value):
            merged_kwargs[key] = _join(values)
        else:
            if any(item != value for item in values):
                raise ValueError(f'非张量关键字参数 {key} 在臂间不一致，无法图化。')
            merged_kwargs[key] = value
    return tuple(merged_args), merged_kwargs


class BatchedForward:
    """一次前向服务同帧多条臂；返回 batch logits。"""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, calls):
        args, kwargs = merge_calls(calls)
        self.calls += 1
        logits, _ = self.model(*args, **kwargs)
        return logits


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, CIRCondition):
        return replace(value, **{
            field.name: getattr(value, field.name).detach().clone()
            for field in fields(CIRCondition)
            if torch.is_tensor(getattr(value, field.name))
        })
    return value


def _copy_value(destination, source):
    if torch.is_tensor(source):
        if (destination.shape != source.shape or destination.dtype != source.dtype
                or destination.device != source.device):
            raise ValueError('CUDA Graph 静态输入形状/类型/设备不匹配。')
        destination.copy_(source)
        return
    if isinstance(source, CIRCondition):
        for field in fields(CIRCondition):
            item = getattr(source, field.name)
            if torch.is_tensor(item):
                _copy_value(getattr(destination, field.name), item)
        return
    if destination != source:
        raise ValueError('CUDA Graph 不支持变化的非张量输入。')


class GraphBatchedForward:
    """固定形状下的批量前向 CUDA Graph；首帧捕获，后续复制输入并回放。"""

    def __init__(self, model, warmup=3):
        self.model = model
        self.warmup = int(warmup)
        self.calls = 0
        self.graph = None
        self.static_args = None
        self.static_kwargs = None
        self.output = None
        self.stream = None

    def _capture(self, args, kwargs):
        device = args[0].device
        if device.type != 'cuda':
            raise ValueError('CUDA Graph 需要 CUDA 输入。')
        self.static_args = tuple(_clone_value(value) for value in args)
        self.static_kwargs = {key: _clone_value(value) for key, value in kwargs.items()}
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            for _ in range(self.warmup):
                self.model(*self.static_args, **self.static_kwargs)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            output, _ = self.model(*self.static_args, **self.static_kwargs)
        self.graph = graph
        self.output = output

    def __call__(self, calls):
        args, kwargs = merge_calls(calls)
        if self.graph is None:
            self._capture(args, kwargs)
        else:
            for target, source in zip(self.static_args, args):
                _copy_value(target, source)
            for key, source in kwargs.items():
                _copy_value(self.static_kwargs[key], source)
        self.calls += 1
        self.graph.replay()
        return self.output
