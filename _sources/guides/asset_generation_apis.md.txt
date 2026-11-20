# 3D 资产生成 API 使用指南

`cloud_robotics_sim.asset_gen` 以统一接口调用四家商业文/图生 3D 服务，把生成的
GLB 资产下载到本地 staging 目录（含 provenance JSON），可直接接入对象库入库管线
（`scripts/expand_object_library.py`）。

## 支持的服务商

| Provider | 服务商 | 凭据环境变量 | 备注 |
|---|---|---|---|
| `tripo` | Tripo AI (VAST) | `TRIPO_API_KEY` | V2 OpenAPI；V2 将于 2026-11-01 下线，届时用 `TRIPO_BASE_URL` 切 V3 |
| `meshy` | Meshy | `MESHY_API_KEY` | text-to-3d v2 / image-to-3d v1，PBR 纹理 |
| `hunyuan3d` | 腾讯混元 3D | `TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY` | 腾讯云 `ai3d` 服务（API 版本 2025-05-13），模块内实现 TC3-HMAC-SHA256 签名 |
| `rodin` | Rodin (Hyper3D) | `RODIN_API_KEY` | v2 API，multipart 提交，几何质量最高 |

可选环境变量：`HY3D_REGION`（腾讯，默认 `ap-guangzhou`）、`TRIPO_BASE_URL`。

安装了 `python-dotenv` 时会自动加载项目根目录的 `.env` 文件。

## CLI

```bash
# 查看服务商配置状态
python -m cloud_robotics_sim.asset_gen providers

# 文生 3D
python -m cloud_robotics_sim.asset_gen text "一把中式木椅" --provider tripo

# 图生 3D（无纹理 / 四边形拓扑）
python -m cloud_robotics_sim.asset_gen image chair.png --provider meshy \
    --no-texture --topology quad

# 腾讯混元极速版
python -m cloud_robotics_sim.asset_gen text "一个陶瓷花瓶" --provider hunyuan3d \
    --timeout 300
```

输出默认落在 `outputs/asset_staging/gen3d/`：

```
tripo_<task_id>.glb     # 模型文件
tripo_<task_id>.json    # provenance（provider/prompt/source_url/created_at）
```

## Python API

```python
from cloud_robotics_sim.asset_gen import AssetGenClient, GenerationRequest

client = AssetGenClient("tripo")  # 不传 provider 则选第一个已配置的
result = client.generate(
    GenerationRequest(prompt="a wooden dining chair", topology="quad"),
    out_dir="outputs/asset_staging/gen3d",
)
print(result.model_file)

# 低层接口：先提交、稍后再收结果
handle = client.submit(GenerationRequest(prompt="a red vase"))
url = client.wait(handle)
```

服务商专有参数经 `GenerationRequest(extra={...})` 透传，例如腾讯极速版
`extra={"rapid": True}`、Rodin 档位 `extra={"tier": "Gen-2.5-High"}`。

自定义服务商：实现 `asset_gen.base.Provider` 协议
（`name`/`env_vars`/`configured()`/`submit()`/`poll()`），然后
`register_provider(MyProvider())` 即可被门面与 CLI 使用。

## 接入对象库

staging 中的 `.glb` 可经现有管线入库：

```bash
python scripts/expand_object_library.py --staging outputs/asset_staging/gen3d \
    --class-name 025_chair --out outputs/asset_expansion
```

## 测试

`tests/asset_gen/` 中的 35 个测试全部零网络（monkeypatch HTTP 层），并用腾讯云
官方签名示例向量校验 TC3-HMAC-SHA256 的正确性。
