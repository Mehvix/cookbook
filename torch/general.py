"""
Source: https://github.com/ezhang7423/eztorchtils/blob/main/eztorchtils/utils/arrays.py
        https://github.com/ezhang7423/eztorchtils/blob/main/eztorchtils/utils/__init__.py
"""

import pdb
import random
import numbers
from typing import Any, Union

import numpy as np

import torch

DTYPE = torch.float
DEVICE = "cuda:0"


def seed_everything(seed):
    """
    Set the seed for all the possible random number generators
    for global packages.
    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def abspath():
    # https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    # return os.path.dirname(inspect.stack()[1][1]) # type: ignore
    # return os.path.dirname(getsourcefile(lambda:0)) # type: ignore
    return os.path.dirname(getsourcefile(inspect_.stack()[1][0]))  # type: ignore


# -----------------------------------------------------------------------------#
# ------------------------------ numpy <--> torch -----------------------------#
# -----------------------------------------------------------------------------#

def to_tensor(v) -> torch.Tensor:
    """Return a torch.Tensor version of the input.

    # Parameters

    v : Input values that can be coerced into being a tensor.

    # Returns

    A tensor version of the input.
    """
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(
            v, dtype=torch.int64 if isinstance(v, numbers.Integral) else torch.float
        )

def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")
        pdb.set_trace()
    # return [x.to(device) for x in xs]


def batchify(batch):
    """
    convert a single dataset item to a batch suitable for passing to a model by
            1) converting np arrays to torch tensors and
            2) and ensuring that everything has a batch dimension
    """
    fn = lambda x: to_torch(x[None])

    batched_vals = []
    for field in batch._fields:
        val = getattr(batch, field)
        val = apply_dict(fn, val) if type(val) is dict else fn(val)
        batched_vals.append(val)
    return type(batch)(*batched_vals)


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x -= x.min()
    x /= x.max()
    return x


def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255).astype(np.uint8)


def set_device(device):
    DEVICE = device
    if "cuda" in device:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#
# https://github.com/allenai/allenact/blob/f00445e4ae8724ccc001b3300af5c56a7f882614/allenact/utils/tensor_utils.py#L1
def to_device_recursively(
    input: Any, device: Union[str, torch.device, int], inplace: bool = True
) -> Any:
    """Recursively places tensors on the appropriate device."""
    if input is None:
        return input
    elif isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, tuple):
        return tuple(
            to_device_recursively(input=subinput, device=device, inplace=inplace)
            for subinput in input
        )
    elif isinstance(input, list):
        if inplace:
            for i in range(len(input)):
                input[i] = to_device_recursively(
                    input=input[i], device=device, inplace=inplace
                )
            return input
        else:
            return [
                to_device_recursively(input=subpart, device=device, inplace=inplace)
                for subpart in input
            ]
    elif isinstance(input, dict):
        if inplace:
            for key in input:
                input[key] = to_device_recursively(
                    input=input[key], device=device, inplace=inplace
                )
            return input
        else:
            return {
                k: to_device_recursively(input=input[k], device=device, inplace=inplace)
                for k in input
            }
    elif isinstance(input, set):
        if inplace:
            for element in list(input):
                input.remove(element)
                input.add(
                    to_device_recursively(element, device=device, inplace=inplace)
                )
        else:
            return {
                to_device_recursively(k, device=device, inplace=inplace) for k in input
            }
    elif isinstance(input, np.ndarray) or np.isscalar(input) or isinstance(input, str):
        return input
    elif hasattr(input, "to"):
        # noinspection PyCallingNonCallable
        return input.to(device=device, inplace=inplace)
    else:
        raise NotImplementedError(
            f"Sorry, value of type {type(input)} is not supported."
        )


def detach_recursively(input: Any, inplace=True):
    """Recursively detaches tensors in some data structure from their
    computation graph."""
    if input is None:
        return input
    elif isinstance(input, torch.Tensor):
        return input.detach()
    elif isinstance(input, tuple):
        return tuple(
            detach_recursively(input=subinput, inplace=inplace) for subinput in input
        )
    elif isinstance(input, list):
        if inplace:
            for i in range(len(input)):
                input[i] = detach_recursively(input[i], inplace=inplace)
            return input
        else:
            return [
                detach_recursively(input=subinput, inplace=inplace)
                for subinput in input
            ]
    elif isinstance(input, dict):
        if inplace:
            for key in input:
                input[key] = detach_recursively(input[key], inplace=inplace)
            return input
        else:
            return {k: detach_recursively(input[k], inplace=inplace) for k in input}
    elif isinstance(input, set):
        if inplace:
            for element in list(input):
                input.remove(element)
                input.add(detach_recursively(element, inplace=inplace))
        else:
            return set(detach_recursively(k, inplace=inplace) for k in input)
    elif isinstance(input, np.ndarray) or np.isscalar(input) or isinstance(input, str):
        return input
    elif hasattr(input, "detach_recursively"):
        # noinspection PyCallingNonCallable
        return input.detach_recursively(inplace=inplace)
    else:
        raise NotImplementedError(
            "Sorry, hidden state of type {} is not supported.".format(type(input))
        )


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])  # type: ignore
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        (
            f"... and {len(counts)-topk} others accounting for"
            f" {_to_str(remaining_parameters)} parameters"
        ),
    )
    return n_parameters
