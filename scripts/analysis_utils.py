from collections import OrderedDict

from .common import (
    MODEL_CKPTS,
    TDANN_CKPTS,
    UNOPTIMIZED_CKPTS,
    SWAPOPT_CKPTS,
    ONELAYER_CKPTS,
    MODEL_C,
    DEFAULT_C,
)
from .get_localizers import localizers


CKPT_GROUPS = OrderedDict(
    [
        ("TopoTransform", MODEL_CKPTS),
        ("TDANN", TDANN_CKPTS),
        ("UNOPTIMIZED", UNOPTIMIZED_CKPTS),
        ("SWAPOPT", SWAPOPT_CKPTS),
        ("ONELAYER", ONELAYER_CKPTS),
    ]
)

DEFAULT_METHOD_ORDER = ("TopoTransform", "TDANN", "SWAPOPT", "UNOPTIMIZED")
FULL_METHOD_ORDER = ("TopoTransform", "TDANN", "SWAPOPT", "UNOPTIMIZED", "ONELAYER")

METHOD_LABELS = {
    "TopoTransform": "TopoTransform",
    "TDANN": "TDANN",
    "UNOPTIMIZED": "VJEPA",
    "SWAPOPT": "SwapOpt",
    "ONELAYER": "OneLayer",
}

METHOD_COLORS = {
    "TopoTransform": MODEL_C,
    "TDANN": DEFAULT_C,
    "UNOPTIMIZED": DEFAULT_C,
    "SWAPOPT": DEFAULT_C,
    "ONELAYER": DEFAULT_C,
}


def resolve_group_names(group_names=None, default=DEFAULT_METHOD_ORDER):
    if group_names is None:
        group_names = default
    if isinstance(group_names, str):
        group_names = (group_names,)
    return [name for name in group_names if name in CKPT_GROUPS]


def get_ckpt_groups(group_names=None, default=DEFAULT_METHOD_ORDER):
    group_names = resolve_group_names(group_names, default=default)
    return {name: CKPT_GROUPS[name] for name in group_names}


def collect_by_ckpt(ckpt_names, fn, *args, verbose=False, prefix="Processing checkpoint: ", **kwargs):
    results = []
    for ckpt_name in ckpt_names:
        if verbose:
            print(f"{prefix}{ckpt_name}")
        results.append(fn(ckpt_name, *args, **kwargs))
    return results


def collect_group_results(group_names, fn, first_kwargs=None, rest_kwargs=None):
    results = {}
    for i, group_name in enumerate(group_names):
        kwargs = first_kwargs if i == 0 else rest_kwargs
        results[group_name] = fn(CKPT_GROUPS[group_name], **(kwargs or {}))
    return results


def collect_localizer_tvals(
    ckpt_names,
    dataset="robert",
    ret_merged=True,
    verbose=True,
    on_result=None,
    **localizer_kwargs,
):
    all_t_vals = []
    for ckpt_name in ckpt_names:
        if verbose:
            print(f"Processing checkpoint: {ckpt_name}")
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(
            ckpt_name, ret_merged=ret_merged, **localizer_kwargs
        )
        if on_result is not None:
            on_result(ckpt_name, t_vals_dicts, p_vals_dicts, layer_positions)
        all_t_vals.append(t_vals_dicts[dataset])
    return all_t_vals
