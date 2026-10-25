# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import parse
from safetensors.torch import load_file
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


def _base_model():
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(12, 8, bias=False), nn.Linear(8, 6, bias=False))


def _save_fsdp2_worker(rank, world_size, root, device_type):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=Path(root, "store").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        model = get_peft_model(_base_model(), LoraConfig(r=2, target_modules=["0", "1"], init_lora_weights=False))
        model.add_adapter("selected", LoraConfig(r=4, target_modules=["0", "1"], init_lora_weights=False))
        model.set_adapter("selected")
        model.to(device)
        reference = copy.deepcopy(model).eval()
        expected = {
            key: value.clone() for key, value in get_peft_model_state_dict(model, adapter_name="selected").items()
        }
        mesh_shape = (world_size,) if world_size == 2 else (2, 2)
        mesh_names = ("shard",) if world_size == 2 else ("replicate", "shard")
        mesh = init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_names)
        for layer in model.base_model.model:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        assert not getattr(model, "_tp_size", None)

        actual = get_peft_model_state_dict(model, adapter_name="selected")
        assert actual.keys() == expected.keys()
        for key, tensor in actual.items():
            assert not isinstance(tensor, DTensor), f"Adapter entry {key} was not gathered"
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)

        for safe_serialization in (True, False):
            destination = Path(root, "safe" if safe_serialization else "bin")
            model.save_pretrained(
                destination,
                selected_adapters=["selected"],
                is_main_process=rank == 0,
                safe_serialization=safe_serialization,
            )
            dist.barrier()
            adapter_path = destination / "selected"
            saved = (
                load_file(adapter_path / "adapter_model.safetensors")
                if safe_serialization
                else torch.load(adapter_path / "adapter_model.bin", weights_only=True, map_location="cpu")
            )
            assert saved.keys() == expected.keys()
            for key in expected:
                assert not isinstance(saved[key], DTensor)
                torch.testing.assert_close(saved[key], expected[key].cpu(), rtol=0, atol=0)
            reloaded = PeftModel.from_pretrained(_base_model(), adapter_path).to(device).eval()
            inputs = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 12) / 24
            with torch.no_grad():
                torch.testing.assert_close(reloaded(inputs), reference(inputs), rtol=0, atol=0)
            # The full plain adapter must also load back into distributed parameters.
            set_peft_model_state_dict(model, saved, adapter_name="selected")
            restored = get_peft_model_state_dict(model, adapter_name="selected")
            for key in expected:
                torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(parse(torch.__version__) < parse("2.6"), reason="Requires the public FSDP2 API")
@pytest.mark.skipif(not dist.is_available(), reason="Requires torch.distributed")
@pytest.mark.parametrize("world_size", [2, 4])
def test_fsdp2_save_adapter_without_tp_metadata(world_size, tmp_path):
    """FSDP2/HSDP saves full selected adapters in both formats, without TP attributes (#3802)."""
    if not dist.is_gloo_available():
        pytest.skip("Requires Gloo")
    mp.spawn(_save_fsdp2_worker, args=(world_size, str(tmp_path), "cpu"), nprocs=world_size, join=True)


@pytest.mark.parametrize("safe_serialization", [True, False])
def test_plain_adapter_save_without_process_group(safe_serialization, tmp_path):
    model = get_peft_model(_base_model(), LoraConfig(r=4, target_modules=["0", "1"], init_lora_weights=False))
    expected = {key: value.clone() for key, value in get_peft_model_state_dict(model).items()}
    model.save_pretrained(tmp_path, safe_serialization=safe_serialization)
    actual = (
        load_file(tmp_path / "adapter_model.safetensors")
        if safe_serialization
        else torch.load(tmp_path / "adapter_model.bin", weights_only=True)
    )
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
