# NPU update_weights OOM Root Cause Around 667f3b85

Last updated: 06/02/2026.

## Summary

In an Ascend/NPU run with:

- `rollout.checkpoint_engine.backend=naive`
- actor parameter offload enabled
- rollout `free_cache_engine` / vLLM sleep mode enabled
- no LoRA

the first rollout weight update can OOM after commit `667f3b85`.

The root cause is not that `667f3b85` adds another explicit copy in the weight-transfer code. The commit changes the NPU allocator behavior:

1. `set_expandable_segments(False)` becomes effective on NPU.
2. The previous TrainingWorker-side default `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is removed.

As a result, the existing naive `update_weights` peak-memory window now runs with NPU expandable segments disabled. That window already temporarily overlaps rollout weights, actor weights, full-weight export tensors, and transfer buckets. With expandable segments disabled, the NPU allocator is more likely to fail due to fragmentation or lack of a sufficiently large contiguous block, even when the steady-state memory usage would fit.

## Important Commits

Known observations from bisect:

| Commit | Status | Notes |
| --- | --- | --- |
| `92e2840c` | good | No OOM. |
| `a9b024d4` | good | Large-weight chunking is not the root cause for this case. |
| `667f3b85` | bad | Introduces the OOM. |
| `60546ef2` | good parent | Parent of `667f3b85`. |
| `2eb9cb86` | bad | Later bad commit; not the root cause because `667f3b85` already reproduces. |

Related historical commits:

- `70744059`: added NPU expandable segment support by setting `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` in `TrainingWorker.__init__`.
- `d9939add`: earlier introduced `set_expandable_segments(False)` / `True` calls around weight update. Before `667f3b85`, these calls only affected CUDA and were a no-op on NPU.

## What 667f3b85 Changed

### Before 667f3b85

`verl.utils.device.set_expandable_segments()` only affected CUDA:

```python
if is_cuda_available:
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

So in the NPU naive `update_weights` path, this call was effectively a no-op:

```python
set_expandable_segments(False)
...
set_expandable_segments(True)
```

Also, `TrainingWorker.__init__` set:

```python
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
```

### After 667f3b85

`set_expandable_segments()` also calls the NPU allocator API:

```python
elif is_npu_available:
    torch.npu.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

At the same time, the explicit `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` assignment in `TrainingWorker.__init__` was removed.

For the `ActorRolloutRefWorker` naive update path, the most important part is that this existing line now really disables expandable segments on NPU:

```python
set_expandable_segments(False)
```

## Naive update_weights Memory Timeline

In `verl/workers/engine_workers.py`, the naive path is:

```python
effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
    await self.checkpoint_engine.send_weights(per_tensor_param)
    return

set_expandable_segments(False)

if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["weights"])

per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon,
    base_sync_done=True,
)

await self.rollout.update_weights(
    per_tensor_param,
    peft_config=peft_config,
    base_sync_done=True,
    global_steps=global_steps,
)

if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)

aggressive_empty_cache(force_sync=True)

if self.config.rollout.free_cache_engine:
    await self.rollout.resume(tags=["kv_cache"])

set_expandable_segments(True)
```

With no LoRA, `peft_config` is `None`, so the LoRA base-sync branch is not entered. The OOM is therefore not caused by LoRA double-sync.

The peak window is still large:

1. `set_expandable_segments(False)` disables expandable segments on NPU after `667f3b85`.
2. `rollout.resume(tags=["weights"])` wakes vLLM rollout weights back onto NPU.
3. `actor.engine.get_per_tensor_param(...)` exports actor weights.
4. Because actor param offload is enabled, the actor engine may have to move parameters from CPU back to NPU before exporting.
5. `rollout.update_weights(...)` starts bucketed IPC/shared-memory transfer and vLLM `load_weights`.
6. Only after rollout weight update finishes does actor offload back to CPU.

So, during the critical interval, memory can include:

- vLLM rollout weights after wake-up
- actor training weights loaded back from CPU
- FSDP/Megatron exported tensors or full tensors
- transfer bucket buffers
- receiver-side tensors and vLLM `load_weights` temporary allocations

## Actor Export Adds Temporary Peak

### FSDP path

`FSDPEngine.get_per_tensor_param()` does:

```python
load_fsdp_model_to_gpu(self.module)
params = self.module.state_dict()
...
per_tensor_param = (
    (
        name,
        param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
        if isinstance(param, DTensor)
        else param,
    )
    for name, param in params.items()
)
```

With offload enabled, this means the actor model can be brought back to NPU for export. For FSDP2/DTensor, `.full_tensor()` and dtype conversion can create large temporary tensors.

### Megatron path

`MegatronEngine.get_per_tensor_param()` does:

```python
load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
per_tensor_param = self.bridge.export_hf_weights(self.module)
```

So Megatron also explicitly reloads model weights to device before exporting them.

## Transfer Bucket Adds Another Large Allocation

The vLLM rollout naive weight transfer uses `BucketedWeightSender`.

Default bucket size:

```python
update_weights_bucket_megabytes: int = 2048
```

The sender allocates a bucket buffer and copies weights into it:

```python
buffer = torch.empty(
    self.bucket_size,
    dtype=torch.uint8,
    device=f"{get_device_name()}:{get_device_id()}",
)
...
self.buffer[offset : offset + weight.nbytes].copy_(...)
```

The receiver reconstructs tensors from the bucket, then calls the vLLM weight loader:

```python
tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
weights.append((name, tensor))
on_bucket_received(weights)
```

If shared memory fallback is used, the receiver may additionally do:

```python
tensor = tensor.to(self.device)
```

This bucket allocation is a large dynamic allocation in the same window where rollout and actor weights already overlap.

## Why This Looks Like Fragmentation OOM

This failure is best understood as allocator-fragmentation / large-block allocation pressure, not necessarily true steady-state capacity exhaustion.

Before `667f3b85`, the NPU path either had `expandable_segments:True` from environment setup or did not apply runtime `set_expandable_segments(False)`. The allocator could tolerate the large dynamic allocations in the update window.

After `667f3b85`, `set_expandable_segments(False)` becomes effective on NPU before the highest-memory part of update weights. The same memory pattern now needs larger contiguous reusable blocks from a less flexible allocator state.

The first update after the first training step is especially sensitive because:

- actor parameters may have just been auto-offloaded to CPU after training context exit
- rollout weights are asleep and then woken before actor export
- vLLM memory pool and prefix/KV state have recently been initialized
- large transfer buckets are allocated for the first weight sync

## vLLM-Ascend Memory Pool Compatibility

There is a real compatibility constraint from vLLM-Ascend.

`vllm_ascend.device_allocator.camem.CaMemAllocator` asserts:

```python
conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
assert "expandable_segments:True" not in conf, (
    "Expandable segments are not compatible with memory pool. "
    "Please track https://github.com/pytorch/pytorch/issues/147851 "
    "for the latest updates."
)
```

That explains why `667f3b85` likely removed the global NPU env setting. vLLM-Ascend sleep mode uses `CaMemAllocator` for memory pool based sleep/wake:

```python
allocator = CaMemAllocator.get_instance()
allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
...
allocator.wake_up(tags=tags)
```

Therefore, simply restoring `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` globally can break vLLM-Ascend sleep-mode workers. The fix needs to avoid polluting vLLM server processes that use `CaMemAllocator`.

## Root Cause

The immediate trigger is:

```python
set_expandable_segments(False)
```

becoming effective on NPU inside the naive update path.

The broader root cause is an unsafe allocator-policy change for a high-peak update window:

- The naive path resumes rollout weights first.
- Actor offload then reloads actor weights to NPU for export.
- Weight export and transfer allocate large temporary tensors and bucket buffers.
- `667f3b85` disables NPU expandable segments exactly before this window.
- The old global env workaround was removed because it conflicts with vLLM-Ascend CaMem, but no scoped replacement was added for the trainer/actor update path.

## Bug or Implementation Choice?

This is partly an implementation choice and partly a regression.

The reasonable implementation choice is removing the global NPU environment variable:

```python
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

That global setting can be inherited by vLLM-Ascend server processes and conflicts with `CaMemAllocator`, which explicitly rejects `expandable_segments:True` when using the memory pool. Restoring this global environment variable is therefore not recommended.

The regression is making the existing `set_expandable_segments(False)` call effective on NPU without re-evaluating the naive update path. That call was originally introduced for CUDA-oriented behavior. After `667f3b85`, it disables NPU expandable segments exactly in the highest-peak window of naive weight update:

- rollout weights are resumed first
- actor weights are loaded back from offload for export
- full tensors or converted tensors may be materialized
- transfer buckets are allocated
- vLLM load-weights temporary allocations may happen

For NPU + naive backend + actor offload, this should be treated as a bug/regression rather than a desirable default behavior.

## Recommended Fix Direction

### Preferred

Do not restore the global `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` workaround.

Instead, keep vLLM-Ascend server processes free from that global env setting, and do not disable expandable segments on NPU in the naive update path.

Keep the CUDA behavior if needed, but gate the disable/enable pair by device:

```python
from verl.utils.device import is_npu_available

disable_expandable_segments = not is_npu_available

if disable_expandable_segments:
    set_expandable_segments(False)

try:
    # resume rollout weights
    # get_per_tensor_param
    # rollout.update_weights
    pass
finally:
    if disable_expandable_segments:
        set_expandable_segments(True)
```

For NPU, keep expandable segments enabled in the trainer/actor worker when running the naive update path, but do not set a global environment variable that vLLM server processes inherit before `CaMemAllocator` initialization.

The important behavior is:

- CUDA keeps the historical temporary disable/enable behavior.
- NPU skips the temporary disable during naive update weights.
- The restore path is protected by `try/finally`, so an OOM or intermediate exception does not leave the allocator in the wrong state.

### More explicit config option

Add a config switch such as:

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      disable_expandable_segments_during_update: auto
```

Recommended `auto` semantics:

- CUDA: `true`
- NPU: `false`

This preserves the historical CUDA behavior while avoiding the NPU offload regression.

### Temporary mitigations

These reduce peak pressure but do not address the allocator-policy root cause:

1. Reduce bucket size:

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      update_weights_bucket_megabytes: 512
```

or even `256`.

2. Reduce rollout memory before weight update if possible.
3. Avoid extra full-tensor export pressure by tuning FSDP/Megatron export path, if available.
4. Ensure actor offload happens promptly after weight update, but this does not help the peak before `rollout.update_weights` completes.

## How to Verify

Useful log points:

- `Before resume weights`
- `After resume weights`
- `Before load_fsdp_model_to_gpu`
- `After load_fsdp_model_to_gpu`
- `Before offload_fsdp_model_to_cpu`
- `After offload_fsdp_model_to_cpu`
- `After update_weights`

Expected confirmation:

1. Parent `60546ef2` does not OOM.
2. `667f3b85` OOMs with NPU `set_expandable_segments(False)` active.
3. Patching NPU path to skip `set_expandable_segments(False)` removes the OOM.
4. Lowering `update_weights_bucket_megabytes` reduces or delays the OOM.

## Files to Inspect

- `verl/utils/device.py`
  - `set_expandable_segments`
- `verl/workers/engine_workers.py`
  - naive `update_weights`
- `verl/workers/engine/fsdp/transformer_impl.py`
  - `FSDPEngine.get_per_tensor_param`
- `verl/workers/engine/megatron/transformer_impl.py`
  - `MegatronEngine.get_per_tensor_param`
- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
  - sender/receiver bucket allocation
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
  - rollout `update_weights`
- `vllm-ascend/vllm_ascend/device_allocator/camem.py`
  - `CaMemAllocator` compatibility assertion
- `vllm-ascend/vllm_ascend/worker/worker.py`
  - vLLM-Ascend sleep/wake implementation
