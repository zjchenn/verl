# 667f3b85 引入的 NPU update_weights OOM 根因分析

Last updated: 06/02/2026.

## 结论

在昇腾 NPU 场景下，如果同时满足：

- `rollout.checkpoint_engine.backend=naive`
- actor 开启参数 offload
- rollout 开启 `free_cache_engine` / vLLM sleep mode
- 不使用 LoRA

那么在 commit `667f3b85` 之后，第一轮 rollout 权重更新可能出现 OOM。

根因不是 `667f3b85` 在权重传输代码里显式增加了一份权重拷贝，而是它改变了 NPU allocator 的行为：

1. `set_expandable_segments(False)` 开始在 NPU 上真实生效。
2. 原来 `TrainingWorker` 初始化阶段默认设置的 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 被移除。

结果是：原本 naive `update_weights` 里已经很高的显存峰值窗口，现在会在 NPU expandable segments 被关闭的状态下运行。这个窗口会临时叠加 rollout weights、actor weights、导出的 full tensor、传输 bucket 以及 vLLM `load_weights` 的临时开销。关闭 expandable segments 后，NPU allocator 更容易因为碎片或无法找到足够大的连续块而 OOM，即使稳态显存理论上可以容纳。

## 关键 Commit

本次二分验证结果：

| Commit | 结果 | 说明 |
| --- | --- | --- |
| `92e2840c` | good | 不 OOM。 |
| `a9b024d4` | good | large weight chunking 不是本场景根因。 |
| `667f3b85` | bad | 引入 OOM。 |
| `60546ef2` | good parent | `667f3b85` 的上一个 commit。 |
| `2eb9cb86` | bad | 后续 bad commit；不是根因，因为 `667f3b85` 已可复现。 |

相关历史 commit：

- `70744059`：通过在 `TrainingWorker.__init__` 中设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，给 NPU 打开 expandable segments。
- `d9939add`：较早引入了 update weights 前后的 `set_expandable_segments(False)` / `set_expandable_segments(True)`。在 `667f3b85` 之前，这两个调用只影响 CUDA，对 NPU 是 no-op。

## 667f3b85 改了什么

### 667f3b85 之前

`verl.utils.device.set_expandable_segments()` 只处理 CUDA：

```python
if is_cuda_available:
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

所以在 NPU naive `update_weights` 路径里，下面这组调用实际上不会改变 NPU allocator：

```python
set_expandable_segments(False)
...
set_expandable_segments(True)
```

同时，`TrainingWorker.__init__` 会设置：

```python
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
```

### 667f3b85 之后

`set_expandable_segments()` 也会调用 NPU allocator API：

```python
elif is_npu_available:
    torch.npu.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

同时，`TrainingWorker.__init__` 中显式设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 的逻辑被删除。

对 `ActorRolloutRefWorker` 的 naive update 路径来说，最关键的变化是这行代码现在会真的关闭 NPU expandable segments：

```python
set_expandable_segments(False)
```

## naive update_weights 的显存时序

`verl/workers/engine_workers.py` 中 naive 路径大致如下：

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

在不使用 LoRA 的情况下，`peft_config` 为 `None`，不会进入 LoRA base-sync 分支。因此这个 OOM 不是 LoRA 双同步导致的。

但这个窗口本身已经很重：

1. `set_expandable_segments(False)` 在 `667f3b85` 后关闭 NPU expandable segments。
2. `rollout.resume(tags=["weights"])` 把 vLLM rollout 权重唤回 NPU。
3. `actor.engine.get_per_tensor_param(...)` 开始导出 actor 权重。
4. 因为 actor 参数 offload 已开启，actor engine 可能需要先把参数从 CPU 拉回 NPU。
5. `rollout.update_weights(...)` 启动 bucketed IPC / shared-memory 传输和 vLLM `load_weights`。
6. 只有 rollout 权重更新完成之后，actor 参数才会 offload 回 CPU。

因此在关键区间内，显存里可能同时存在：

- wake up 之后的 vLLM rollout weights
- 从 CPU/offload 状态拉回来的 actor training weights
- FSDP/Megatron 导出的 tensor 或 full tensor
- 权重传输 bucket buffer
- receiver 侧 tensor 以及 vLLM `load_weights` 的临时分配

## Actor 权重导出会放大峰值

### FSDP 路径

`FSDPEngine.get_per_tensor_param()` 会执行：

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

开启 offload 时，这意味着 actor 模型可能会为了导出权重被重新加载到 NPU。对于 FSDP2 / DTensor，`.full_tensor()` 和 dtype 转换还可能产生较大的临时 tensor。

### Megatron 路径

`MegatronEngine.get_per_tensor_param()` 会执行：

```python
load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
per_tensor_param = self.bridge.export_hf_weights(self.module)
```

所以 Megatron 路径也会在导出前显式把模型权重加载回 device。

## 权重传输 Bucket 也会增加大块分配

vLLM rollout 的 naive 权重传输使用 `BucketedWeightSender`。

默认 bucket 大小是：

```python
update_weights_bucket_megabytes: int = 2048
```

sender 会分配一个 bucket buffer，并把权重拷进去：

```python
buffer = torch.empty(
    self.bucket_size,
    dtype=torch.uint8,
    device=f"{get_device_name()}:{get_device_id()}",
)
...
self.buffer[offset : offset + weight.nbytes].copy_(...)
```

receiver 会从 bucket 重建 tensor，然后调用 vLLM 权重加载逻辑：

```python
tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
weights.append((name, tensor))
on_bucket_received(weights)
```

如果走 shared memory fallback，receiver 侧还可能额外执行：

```python
tensor = tensor.to(self.device)
```

这个 bucket 分配是一个大块动态分配，正好发生在 rollout weights 和 actor weights 已经叠加的窗口里。

## 为什么更像碎片型 OOM

这个问题更应该理解为 allocator fragmentation / large-block allocation pressure，而不一定是稳态显存容量真的不够。

在 `667f3b85` 之前，NPU 路径要么通过环境变量开启了 `expandable_segments:True`，要么运行时的 `set_expandable_segments(False)` 对 NPU 没有作用。因此 allocator 能容忍 update weights 期间的大块动态分配。

在 `667f3b85` 之后，`set_expandable_segments(False)` 会在 update weights 的最高峰值阶段之前真实关闭 NPU expandable segments。同样的内存模式现在需要 allocator 在更不灵活的状态下找到足够大的连续可用块。

第一轮训练后的第一次 update weights 尤其敏感，因为：

- actor 参数可能刚在训练 context 退出时被自动 offload 到 CPU
- rollout weights 处于 sleep 状态，随后会在 actor 导出前被唤醒
- vLLM memory pool、prefix cache / KV 状态刚完成初始化
- 第一次权重同步会分配较大的传输 bucket

## vLLM-Ascend Memory Pool 兼容性背景

vLLM-Ascend 这里有一个真实的兼容性约束。

`vllm_ascend.device_allocator.camem.CaMemAllocator` 中有断言：

```python
conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
assert "expandable_segments:True" not in conf, (
    "Expandable segments are not compatible with memory pool. "
    "Please track https://github.com/pytorch/pytorch/issues/147851 "
    "for the latest updates."
)
```

这解释了为什么 `667f3b85` 可能要移除全局 NPU env 设置。vLLM-Ascend sleep mode 使用 `CaMemAllocator` 做 memory pool 形式的 sleep/wake：

```python
allocator = CaMemAllocator.get_instance()
allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
...
allocator.wake_up(tags=tags)
```

因此，简单恢复全局 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 可能会破坏 vLLM-Ascend sleep-mode worker。修复时需要避免污染使用 `CaMemAllocator` 的 vLLM server 进程。

## 根因

直接触发点是：

```python
set_expandable_segments(False)
```

开始在 NPU naive update 路径中真实生效。

更完整的根因是：对高峰值 update weights 窗口做了不安全的 allocator 策略变更。

- naive 路径先恢复 rollout weights。
- actor offload 随后把 actor weights 重新加载到 NPU 以便导出。
- 权重导出和传输会产生大 tensor 与 bucket buffer。
- `667f3b85` 正好在这个窗口之前关闭 NPU expandable segments。
- 旧的全局 env workaround 因为和 vLLM-Ascend CaMem 冲突被删除，但 trainer/actor update 路径没有补一个 scoped replacement。

## 这是 Bug 还是实现选择？

这是一个“合理实现选择 + 回归 bug”的组合。

合理的实现选择是移除全局 NPU 环境变量：

```python
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

这个全局设置可能被 vLLM-Ascend server 进程继承，并且和 `CaMemAllocator` 冲突。`CaMemAllocator` 在使用 memory pool 时会显式拒绝 `expandable_segments:True`。因此不建议恢复这个全局环境变量。

真正的问题是：让已有的 `set_expandable_segments(False)` 在 NPU 上生效，却没有重新评估 naive update path。这个调用最早是 CUDA 语义下的历史行为。`667f3b85` 之后，它会在 naive weight update 峰值最高的窗口关闭 NPU expandable segments：

- rollout weights 已经恢复
- actor weights 从 offload 状态加载回来用于导出
- full tensor 或转换后的 tensor 可能被物化
- transfer bucket 被分配
- vLLM load-weights 可能产生临时分配

对于 NPU + naive backend + actor offload，这应该视为 bug / regression，而不是期望的默认行为。

## 建议修复方向

### 首选方案

不要恢复全局 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` workaround。

更合理的做法是：不要让 vLLM-Ascend server 进程继承这个全局 env，同时在 NPU naive update path 里不要关闭 expandable segments。

CUDA 如果仍然需要历史行为，可以保留；但应该按 device gate 住 disable/enable 逻辑：

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

关键语义：

- CUDA 保留历史上的临时 disable/enable 行为。
- NPU 在 naive update weights 期间跳过临时 disable。
- 使用 `try/finally` 确保中间 OOM 或异常时不会把 allocator 留在错误状态。

### 更显式的配置项

可以增加一个配置开关：

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      disable_expandable_segments_during_update: auto
```

推荐 `auto` 语义：

- CUDA: `true`
- NPU: `false`

这样可以保留 CUDA 历史行为，同时规避 NPU offload 场景的回归。

### 临时规避

以下方法只能降低峰值压力，不能解决 allocator 策略问题本身：

1. 降低 bucket 大小：

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      update_weights_bucket_megabytes: 512
```

必要时可以进一步降到 `256`。

2. 尽量在 weight update 前释放 rollout 侧可释放显存。
3. 如果有可用开关，优化 FSDP/Megatron 权重导出路径，避免额外 full-tensor 峰值。
4. 确保 update weights 后 actor 参数及时 offload，但这无法降低 `rollout.update_weights` 完成前的最高峰值。

## 验证方法

建议关注这些日志点：

- `Before resume weights`
- `After resume weights`
- `Before load_fsdp_model_to_gpu`
- `After load_fsdp_model_to_gpu`
- `Before offload_fsdp_model_to_cpu`
- `After offload_fsdp_model_to_cpu`
- `After update_weights`

预期验证结果：

1. parent `60546ef2` 不 OOM。
2. `667f3b85` 在 NPU `set_expandable_segments(False)` 生效时 OOM。
3. patch NPU 路径跳过 `set_expandable_segments(False)` 后，OOM 消失。
4. 降低 `update_weights_bucket_megabytes` 会缓解或延后 OOM。

## 需要重点检查的文件

- `verl/utils/device.py`
  - `set_expandable_segments`
- `verl/workers/engine_workers.py`
  - naive `update_weights`
- `verl/workers/engine/fsdp/transformer_impl.py`
  - `FSDPEngine.get_per_tensor_param`
- `verl/workers/engine/megatron/transformer_impl.py`
  - `MegatronEngine.get_per_tensor_param`
- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
  - sender / receiver bucket allocation
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
  - rollout `update_weights`
- `vllm-ascend/vllm_ascend/device_allocator/camem.py`
  - `CaMemAllocator` 兼容性断言
- `vllm-ascend/vllm_ascend/worker/worker.py`
  - vLLM-Ascend sleep/wake 实现
