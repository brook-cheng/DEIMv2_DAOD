"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import random
import numpy as np
import atexit

import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler

# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader


def setup_distributed(
    print_rank: int = 0,
    print_method: str = "builtin",
    seed: int = None,
):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv("RANK", -1))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        torch.distributed.init_process_group(init_method="env://")
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        print(f"Rank {rank}, Local Rank {LOCAL_RANK}, World Size {WORLD_SIZE}")
        torch.cuda.set_device(LOCAL_RANK)
        torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print("Initialized distributed mode...")

    except Exception as e:
        if RANK == -1:
            # Plain single-process run — no torchrun env; non-distributed is
            # the intended mode, stay silent.
            enabled_dist = False
            print("Not init distributed mode.")
        else:
            # torchrun started us but init failed: degrading silently would
            # fork N competing single-GPU processes on cuda:0 (observed on
            # the training server). Surface everything and abort this rank.
            import traceback

            print(
                f"[dist] FATAL: torchrun context (RANK={RANK}, LOCAL_RANK={LOCAL_RANK}, "
                f"WORLD_SIZE={WORLD_SIZE}) but init_process_group failed — "
                f"{type(e).__name__}: {e}"
            )
            print(
                f"[dist] env: MASTER_ADDR={os.getenv('MASTER_ADDR')} "
                f"MASTER_PORT={os.getenv('MASTER_PORT')} "
                f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}"
            )
            traceback.print_exc()
            raise

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method="builtin"):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    if method == "builtin":
        builtin_print = __builtin__.print

    elif method == "rich":
        import rich

        builtin_print = rich.print

    else:
        raise AttributeError("")

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment"""
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_on_master_atomic(state, path):
    """Atomic checkpoint write on the main process: tmp file + os.replace.

    Power loss mid-``torch.save`` must never corrupt the recovery point:
    the payload goes to ``<path>.tmp`` first and ``os.replace`` (atomic on
    POSIX) swaps it in, so readers always see either the old or the new
    complete file — never a truncated one.
    """
    if not is_main_process():
        return
    tmp = f"{path}.tmp"
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def warp_model(
    model: torch.nn.Module,
    sync_bn: bool = False,
    dist_mode: str = "ddp",
    find_unused_parameters: bool = False,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    **kwargs,
):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == "dp":
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == "ddp":
            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            raise AttributeError("")

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model


def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(
            loader.dataset,
            loader.batch_size,
            sampler=sampler,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            num_workers=loader.num_workers,
        )
    return loader


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        # Rank-local loss dicts can hold DIFFERENT key sets: e.g. a rank whose
        # batch has zero GT boxes skips denoising, so its criterion omits
        # every ``*_dn_*`` loss key. Reducing rank-local keys directly makes
        # ranks enter all_reduce with different numel and desync the
        # collective stream (NCCL hang). Build ONE rank-consistent schema:
        # the union of all ranks' keys in deterministic sorted order; a rank
        # missing a key contributes a zero tensor of matching dtype/device.
        local_keys = sorted(data.keys())
        keys_on_ranks = [None] * world_size
        torch.distributed.all_gather_object(keys_on_ranks, local_keys)
        union_keys = sorted({k for keys in keys_on_ranks for k in keys})

        reference = next(iter(data.values()), None)
        if reference is None:
            dtype, device = torch.float32, torch.device("cpu")
        else:
            dtype, device = reference.dtype, reference.device

        values = torch.stack(
            [
                data[k] if k in data else torch.zeros((), dtype=dtype, device=device)
                for k in union_keys
            ],
            dim=0,
        )
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(union_keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import torch
    import warnings

    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok


def is_compile(model):
    import torch._dynamo

    return type(model) in (torch._dynamo.OptimizedModule,)


def de_complie(model):
    return model._orig_mod if is_compile(model) else model
