# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

import threading

from .parallel_state import get_tp_group


class CommunicationLock:
    def __init__(self):
        self._count = 0
        self._flag = False
        self._cond = threading.Condition()

    def _check_condition_normal(self):
        return self._count % 3 != 0

    def _check_condition_cache(self):
        return self._count % 3 == 0

    def acquire_for_normal(self):
        with self._cond:
            while self._flag and not self._check_condition_normal():
                self._cond.wait()

    def acquire_for_cache(self):
        with self._cond:
            while self._flag and not self._check_condition_cache():
                self._cond.wait()

    def release(self):
        with self._cond:
            if self._flag:
                self._count += 1
                self._cond.notify_all()

    def set_flag(self):
        with self._cond:
            self._count = 0
            self._flag = True
            self._cond.notify_all()

    def reset(self):
        self._count = 0
        self._flag = False
        self._cond.notify_all()

    @property
    def count(self):
        return self._count

communication_counter = None

def get_communication_counter():
    global communication_counter
    communication_counter = CommunicationLock()
    return communication_counter

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    global communication_counter
    if communication_counter is not None:
        communication_counter.acquire_for_normal()

    output = get_tp_group().all_reduce(input_)

    if communication_counter is not None:
        communication_counter.release()
    return output


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1, cache_all_gather=False
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    global communication_counter
    if communication_counter is not None:
        if cache_all_gather:
            communication_counter.acquire_for_cache()
        else:
            communication_counter.acquire_for_normal()

    output = get_tp_group().all_gather(input_, dim)

    if communication_counter is not None:
        communication_counter.release()
    return output


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    global communication_counter
    if communication_counter is not None:
        communication_counter.acquire_for_normal()

    output = get_tp_group().gather(input_, dst, dim)

    if communication_counter is not None:
        communication_counter.release()

    return output


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
