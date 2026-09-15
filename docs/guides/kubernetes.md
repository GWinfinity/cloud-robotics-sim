# Kubernetes 部署指南

本文介绍如何在 Kubernetes 集群上部署 genesis-cloud-sim 仿真 worker，并使用
[KEDA](https://keda.sh/) 基于 Redis 任务队列长度实现自动扩缩容（空闲时缩到 0，
有任务时按队列长度扩容）。

## 架构概览

```
submit task (scripts/k8s_submit_task.py)
        │  LPUSH
        ▼
┌──────────────┐      BRPOP        ┌──────────────────┐
│ Redis 队列    │ ────────────────▶ │ sim-worker-cpu   │  KEDA ScaledObject
│ sim-tasks-*  │                    │ sim-worker-gpu   │  (listLength 触发)
└──────────────┘ ◀──────────────── └──────────────────┘
        │  RPUSH (sim-results)
        ▼
   结果 JSON
```

- CPU 和 GPU worker 使用**独立的队列**（`sim-tasks-cpu` / `sim-tasks-gpu`），
  避免同一条队列同时触发两类 worker 造成重复消费。
- 每条队列对应一个 KEDA `ScaledObject`，队列中每积压 1 个任务扩容 1 个副本。

## 前置条件

- Kubernetes 集群（≥ 1.25）
- [KEDA](https://keda.sh/docs/latest/deploy/) ≥ 2.9（redis scaler 默认启用）：

  ```bash
  kubectl apply -f https://github.com/kedacore/keda/releases/latest/download/keda.yaml
  ```

- GPU worker 还需要 [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)
  以及带 GPU 的节点（或 [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)）

## 构建并推送镜像

复用仓库根目录的 `Dockerfile`：

```bash
# CPU 镜像
docker build -t <registry>/genesis-cloud-sim:cpu .
docker push <registry>/genesis-cloud-sim:cpu

# GPU 镜像（CUDA）
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --build-arg TORCH_BACKEND=cuda \
  -t <registry>/genesis-cloud-sim:gpu .
docker push <registry>/genesis-cloud-sim:gpu
```

注意镜像内置了 `redis` 依赖（`pip install -e ".[dev]"` 安装的是 `all` extra，
已包含 `k8s`）。然后修改 `deploy/kubernetes/sim-worker-{cpu,gpu}.yaml` 中的
`image` 字段为你的 registry 地址。

## 部署

```bash
kubectl apply -k deploy/kubernetes/
# 或逐个文件：kubectl apply -f deploy/kubernetes/ --recursive
```

验证组件就绪：

```bash
kubectl -n genesis-sim get deploy,svc
kubectl -n genesis-sim get scaledobject
# 空队列时 worker 副本数应为 0：
kubectl -n genesis-sim get pods
```

## 提交仿真任务

在本地（需要 `pip install cloud-robotics-sim[k8s]`）或集群内任一 Pod 中执行：

```bash
# CPU 队列
python scripts/k8s_submit_task.py \
  --redis-url localhost:6379 \
  --queue sim-tasks-cpu \
  --type patent --param run=US821393 --param steps=500

# GPU 队列
python scripts/k8s_submit_task.py \
  --redis-url localhost:6379 \
  --queue sim-tasks-gpu \
  --type patent --param run=US821393 --param device=cuda
```

观察扩容过程：

```bash
kubectl -n genesis-sim get pods -w        # worker pod 出现
kubectl -n genesis-sim logs -l app=sim-worker -f
```

## 查看结果

结果以 JSON 追加到 Redis 的 `sim-results` 列表：

```bash
kubectl -n genesis-sim exec deploy/redis -- redis-cli LRANGE sim-results 0 -1
```

也可以在集群内运行一个一次性 worker 处理任务并直接看日志：

```bash
kubectl -n genesis-sim run worker-once --rm -i --restart=Never \
  --image=<registry>/genesis-cloud-sim:cpu -- \
  python -m cloud_robotics_sim worker --queue sim-tasks-cpu
```

## 任务格式

```json
{
  "task_id": "uuid（可选，自动生成）",
  "type": "patent",
  "params": {"run": "US821393", "steps": 500, "device": "cuda", "seed": 0}
}
```

当前内置的任务类型：

| type | 说明 | params |
|------|------|--------|
| `patent` | 运行经典专利仿真（无头模式） | `run`（必填，专利号）、`steps`、`dt`、`substeps`、`resolution`、`device`、`seed`、`record_path`、`follow` |

新增任务类型：在 `src/cloud_robotics_sim/runtime/queue_worker.py` 的
`_TASK_HANDLERS` 中注册处理函数即可。

## 生产化注意事项

- **Redis**：manifest 中的 Redis 是无持久化的单实例，仅用于演示。生产环境请换成
  托管 Redis / Redis Sentinel，并通过 `deploy/kubernetes/secret.example.yaml`
  的方式以 Secret 注入带凭据的 `REDIS_URL`。
- **GPU 调度**：按集群实际标签调整 `sim-worker-gpu.yaml` 的 `nodeSelector`
  （GPU Operator 常用 `nvidia.com/gpu.present: "true"`）；如需隔离 GPU 节点池，
  为 worker 添加对应的 `tolerations`。
- **资源配额**：CPU worker 默认 requests 1 CPU / 2Gi，GPU worker 独占 1 张 GPU；
  请配合 Namespace `ResourceQuota` / `LimitRange` 防止队列任务过多时打满集群。
- **镜像拉取**：生产集群中请将 `imagePullPolicy` 改为 `Always` 并使用私有
  registry 的 `imagePullSecret`。
