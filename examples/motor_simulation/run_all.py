"""
电机仿真 Demo 合集
====================
运行所有电机仿真示例并生成结果。
"""
import sys
import time
from pathlib import Path

demos = [
    ("demo_01_magnetic_field.py", "电机磁场分布仿真"),
    ("demo_02_eddy_current.py",    "涡流与集肤效应仿真"),
    ("demo_03_joule_heating.py",   "绕组焦耳热与温升仿真"),
    ("demo_04_motor_performance.py", "电机性能曲线仿真"),
]

out_dir = Path("examples/motor_simulation")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  电机仿真 Demo 合集")
print("=" * 60)

for script, name in demos:
    print(f"\n▶ [{name}]")
    print(f"  运行 {script} ...")
    t0 = time.time()
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        script.replace('.py', ''), 
        str(out_dir / script)
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print(f"  ✓ 完成 (耗时 {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("  所有 Demo 运行完毕!")
print(f"  结果保存在: {out_dir.resolve()}")
print("=" * 60)
