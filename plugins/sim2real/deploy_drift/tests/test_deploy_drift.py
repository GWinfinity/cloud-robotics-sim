"""deploy_drift 插件测试:drift 度量(纯 torch)+ ONNX 工件链(deploy extra)。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.optional_deps import deploy_only, torch_only

pytestmark = torch_only


def _mlp(state_dim: int = 6, hidden: int = 16, out_dim: int = 3) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(state_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, out_dim),
    )


def test_drift_metric() -> None:
    """Drift 度量:mean/max 绝对误差正确,形状不匹配报错,可脱离计算图。"""
    from plugins.sim2real.deploy_drift.core.harness import drift

    ref = torch.zeros(4, 3)
    cand = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    metrics = drift(ref, cand)
    assert metrics["mean_abs"] == pytest.approx((0.1 + 0.2 + 0.3 + 0 + 3 + 1.5) / 12)
    assert metrics["max_abs"] == pytest.approx(1.0)

    grad_ref = torch.zeros(2, 2, requires_grad=True)
    grad_cand = torch.ones(2, 2)
    assert drift(grad_ref, grad_cand) == {"mean_abs": 1.0, "max_abs": 1.0}

    with pytest.raises(ValueError):
        drift(torch.zeros(2, 2), torch.zeros(3, 2))


@deploy_only
def test_export_onnx_fidelity_and_dynamic_batch(tmp_path) -> None:
    """ONNX f32 导出保真(~1e-5 级)且动态 batch 轴生效。"""
    import onnx

    from plugins.sim2real.deploy_drift.core.export import export_onnx
    from plugins.sim2real.deploy_drift.core.harness import OrtOperator

    op = _mlp().eval()
    path = export_onnx(op, state_dim=6, path=tmp_path / "model.onnx")
    onnx.checker.check_model(onnx.load(path))

    ort_op = OrtOperator(path)
    x = torch.randn(1, 6)
    err = (ort_op(x) - op(x)).abs().max().item()
    assert err < 1e-5, f"export fidelity drift {err:.2e}"

    x5 = torch.randn(5, 6)
    out5 = ort_op(x5)
    assert out5.shape == (5, 3)
    assert (out5 - op(x5)).abs().max().item() < 1e-5


@deploy_only
def test_fp16_and_int8_artifacts_run(tmp_path) -> None:
    """fp16 与动态 INT8 工件可运行且漂移可度量;INT8 引入非零量化误差。"""
    import onnx

    from plugins.sim2real.deploy_drift.core.export import (
        convert_fp16,
        export_onnx,
        quantize_dynamic_int8,
    )
    from plugins.sim2real.deploy_drift.core.harness import OrtOperator, drift

    op = _mlp().eval()
    f32 = export_onnx(op, 6, tmp_path / "m_f32.onnx")
    fp16 = convert_fp16(f32, tmp_path / "m_fp16.onnx")
    int8 = quantize_dynamic_int8(f32, tmp_path / "m_int8.onnx")

    m16 = onnx.load(fp16)
    io_types = {
        t.type.tensor_type.elem_type
        for t in list(m16.graph.input) + list(m16.graph.output)
    }
    assert io_types == {onnx.TensorProto.FLOAT}, "keep_io_types must preserve f32 I/O"

    x = torch.randn(16, 6)
    ref = op(x)
    d16 = drift(ref, OrtOperator(fp16)(x))
    d8 = drift(ref, OrtOperator(int8)(x))
    assert np.isfinite(d16["mean_abs"]) and np.isfinite(d8["mean_abs"])
    assert d8["max_abs"] > 1e-6, "int8 quantization should perturb a random MLP"
    # 注:微型模型的 QDQ 图( scale/zp + DequantizeLinear )可能比 2x 权值压缩
    # 更占空间,文件大小比较对 MLP 无意义,故只验漂移与可运行性


@deploy_only
def test_closed_loop_drift_compounds(tmp_path) -> None:
    """闭环 K 步:同初值不同精度工件驱动,漂移随步数放大(harness 的核心用途)。

    用放大型线性算子(增益 1.3>1):逐步误差经同一雅可比传播会几何增长;
    tanh 这类收缩非线性反而会把两条轨迹拉到同一片,掩盖复利效应。
    """
    from plugins.sim2real.deploy_drift.core.export import (
        export_onnx,
        quantize_dynamic_int8,
    )
    from plugins.sim2real.deploy_drift.core.harness import OrtOperator, drift

    op = torch.nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        op.weight.copy_(1.3 * torch.eye(6) + 0.05 * torch.randn(6, 6))
    op = op.eval()
    f32 = export_onnx(op, 6, tmp_path / "loop_f32.onnx")
    int8 = quantize_dynamic_int8(f32, tmp_path / "loop_int8.onnx")

    def rollout(operator, x0, steps):
        states = [x0]
        x = x0
        with torch.no_grad():
            for _ in range(steps):
                x = operator(x)
                states.append(x)
        return torch.stack(states)

    x0 = torch.randn(4, 6)
    ref_full = rollout(op, x0, 8)
    int8_full = rollout(OrtOperator(int8), x0, 8)

    d_early = drift(ref_full[:2], int8_full[:2])
    d_late = drift(ref_full, int8_full)
    assert d_late["mean_abs"] > d_early["mean_abs"] >= 0.0
