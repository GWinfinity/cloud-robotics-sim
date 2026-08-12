"""
Demo 3: 绕组焦耳热与温升仿真 (Joule Heating & Thermal)
=========================================================
计算电机绕组中的 I²R 损耗和温度分布。
等效于 JouleHeatingSolver + ThermalSolver 的核心功能。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 物理参数
# ============================================================
# 铜绕组参数
RHO_CU = 1.68e-8           # 铜电阻率 [Ω·m] (20°C)
ALPHA_CU = 0.00393         # 铜电阻温度系数 [1/K]
T_AMB = 25.0                # 环境温度 [°C]

# 绕组几何
WIRE_DIAMETER = 0.0008      # 线径 [mm]
WIRE_AREA = np.pi * (WIRE_DIAMETER/2)**2  # 导线截面积 [m²]
TURNS = 50                   # 每线圈匝数
COIL_LENGTH = 0.05           # 线圈平均匝长 [m]
TOTAL_WIRE_LENGTH = TURNS * COIL_LENGTH  # 总导线长度 [m]
NUM_COILS = 12               # 线圈数量

# 热参数
K_INSULATION = 0.2           # 绝缘层导热系数 [W/m·K]
K_COPPER = 400               # 铜导热系数 [W/m·K]
H_AIR = 10                   # 空气自然对流换热系数 [W/m²·K]
H_FORCED = 50                # 强制风冷换热系数 [W/m²·K]
H_WATER = 500                # 水冷换热系数 [W/m²·K]

# 材料密度和比热
RHO_CU_DENSITY = 8960        # 铜密度 [kg/m³]
CP_CU = 385                   # 铜比热 [J/kg·K]
RHO_INSULATION = 1400        # 绝缘密度 [kg/m³]
CP_INSULATION = 1500         # 绝缘比热 [J/kg·K]

# ============================================================
# 2. 电流与损耗计算
# ============================================================
R_coil_20 = RHO_CU * TOTAL_WIRE_LENGTH / WIRE_AREA  # 20°C时的电阻
print(f"   线圈电阻 (20°C): {R_coil_20:.4f} Ω")

# 不同电流水平下的损耗
I_currents = np.linspace(0.5, 20, 50)  # 电流范围 [A]
P_loss_20 = NUM_COILS * I_currents**2 * R_coil_20  # 总铜损 (20°C)
P_loss_hot = NUM_COILS * I_currents**2 * R_coil_20 * (1 + ALPHA_CU * 60)  # 80°C时的损耗

# ============================================================
# 3. 稳态温升计算
# ============================================================
# 热阻网络模型: R_th = R_cond + R_conv
# 温升: ΔT = P * R_th

# 热阻计算
# 传导热阻: R_cond = L / (k*A)
INSULATION_THICKNESS = 0.0002  # 绝缘层厚度 [m]
COIL_SURFACE_AREA = TOTAL_WIRE_LENGTH * np.pi * WIRE_DIAMETER  # 散热面积 [m²]
R_cond = INSULATION_THICKNESS / (K_INSULATION * COIL_SURFACE_AREA)  # 传导热阻 [K/W]

# 对流热阻: R_conv = 1 / (h*A)
R_conv_air = 1 / (H_AIR * COIL_SURFACE_AREA)  # 自然冷却
R_conv_forced = 1 / (H_FORCED * COIL_SURFACE_AREA)  # 强制风冷
R_conv_water = 1 / (H_WATER * COIL_SURFACE_AREA)  # 水冷

# 总热阻
R_th_air = R_cond + R_conv_air
R_th_forced = R_cond + R_conv_forced
R_th_water = R_cond + R_conv_water

# 单线圈在不同电流下的温升 (热平衡: P_loss * R_th = ΔT)
I_range = np.linspace(0.5, 15, 100)
P_single = I_range**2 * R_coil_20

delta_T_air = P_single * R_th_air
delta_T_forced = P_single * R_th_forced
delta_T_water = P_single * R_th_water

# ============================================================
# 4. 瞬态温升曲线 (考虑热容)
# ============================================================
# T(t) = T_amb + P*R_th * (1 - exp(-t/τ))
# τ = R_th * C_th (热时间常数)
# C_th = ρ * V * cp (热容)

# 单线圈热容
V_wire = WIRE_AREA * TOTAL_WIRE_LENGTH
V_insulation = COIL_SURFACE_AREA * INSULATION_THICKNESS
C_th_wire = RHO_CU_DENSITY * V_wire * CP_CU
C_th_insulation = RHO_INSULATION * COIL_SURFACE_AREA * INSULATION_THICKNESS * CP_INSULATION
C_th_total = C_th_wire + C_th_insulation

# 热时间常数 (不同冷却方式)
tau_air = R_th_air * C_th_total
tau_forced = R_th_forced * C_th_total
tau_water = R_th_water * C_th_total

# 阶跃响应 (5A, 10A 电流阶跃)
t = np.linspace(0, 3000, 500)  # 时间 [s]

def temp_transient(t, I, R_th, tau):
    """瞬态温升"""
    P = I**2 * R_coil_20
    return P * R_th * (1 - np.exp(-t / tau))

T_5A_air = T_AMB + temp_transient(t, 5, R_th_air, tau_air)
T_10A_air = T_AMB + temp_transient(t, 10, R_th_air, tau_air)
T_5A_forced = T_AMB + temp_transient(t, 5, R_th_forced, tau_forced)
T_10A_forced = T_AMB + temp_transient(t, 10, R_th_forced, tau_forced)

# ============================================================
# 5. 2D 温度分布 (简化: 圆导线截面)
# ============================================================
NX, NY = 100, 100
r_wire = WIRE_DIAMETER / 2
x = np.linspace(-r_wire*1.5, r_wire*1.5, NX)
y = np.linspace(-r_wire*1.5, r_wire*1.5, NY)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# 通10A电流时的温度分布 (简化: 抛物线分布)
I_10A = 10.0
q_dot = I_10A**2 * R_coil_20 / (np.pi * r_wire**2 * TOTAL_WIRE_LENGTH)  # 体积发热率 [W/m³]

# 温度分布: T(r) = T_surface + q_dot/(4*k_copper) * (r_wire² - r²)
T_surface = T_AMB + I_10A**2 * R_coil_20 * R_th_air  # 表面温度
T_distribution = np.zeros_like(R)
inside = R <= r_wire
T_distribution[inside] = T_surface + q_dot/(4*K_COPPER) * (r_wire**2 - R[inside]**2)
T_distribution[~inside] = T_AMB + (T_surface - T_AMB) * np.exp(-(R[~inside]-r_wire)/0.0002)

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 铜损 vs 电流
ax1 = axes[0, 0]
ax1.plot(I_currents, P_loss_20, 'b-', linewidth=2, label='20°C')
ax1.plot(I_currents, P_loss_hot, 'r--', linewidth=2, label='80°C')
ax1.set_xlabel('电流 [A]')
ax1.set_ylabel('总铜损 [W]')
ax1.set_title('绕组铜损 vs 电流 (12线圈)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 标注
ax1.annotate(f'I²R (20°C)\n{R_coil_20:.4f}Ω/线圈',
             xy=(10, 10**2*R_coil_20*12), fontsize=9,
             xytext=(12, 50), arrowprops=dict(arrowstyle='->'))

# 6.2 稳态温升 vs 电流 (不同冷却方式)
ax2 = axes[0, 1]
ax2.plot(I_range, delta_T_air, 'r-', linewidth=2, label='自然冷却')
ax2.plot(I_range, delta_T_forced, 'b-', linewidth=2, label='强制风冷')
ax2.plot(I_range, delta_T_water, 'c-', linewidth=2, label='水冷')
ax2.axhline(y=105, color='gray', linestyle='--', alpha=0.5, label='绝缘等级F (155°C)')
ax2.axhline(y=80, color='gray', linestyle=':', alpha=0.5, label='绝缘等级B (130°C)')
ax2.set_xlabel('电流 [A]')
ax2.set_ylabel('温升 ΔT [°C]')
ax2.set_title('稳态温升 vs 电流 (不同冷却方式)', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 6.3 瞬态温升曲线
ax3 = axes[0, 2]
ax3.plot(t/60, T_5A_air, 'r-', linewidth=2, label='5A 自然冷却')
ax3.plot(t/60, T_10A_air, 'r--', linewidth=2, label='10A 自然冷却')
ax3.plot(t/60, T_5A_forced, 'b-', linewidth=2, label='5A 强制风冷')
ax3.plot(t/60, T_10A_forced, 'b--', linewidth=2, label='10A 强制风冷')
ax3.axhline(y=155, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('时间 [min]')
ax3.set_ylabel('温度 [°C]')
ax3.set_title('瞬态温升曲线 (电流阶跃响应)', fontsize=14)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 50)

# 6.4 温度分布 (导线截面)
ax4 = axes[1, 0]
im4 = ax4.imshow(T_distribution, extent=[-r_wire*1000*1.5, r_wire*1000*1.5,
                                          -r_wire*1000*1.5, r_wire*1000*1.5],
                 origin='lower', cmap='hot')
ax4.set_title(f'导线截面温度分布 (10A, 自然冷却)', fontsize=14)
plt.colorbar(im4, ax=ax4, label='温度 [°C]')
ax4.set_xlabel('x [mm]')
ax4.set_ylabel('y [mm]')

circle = plt.Circle((0, 0), r_wire*1000, fill=False, color='white', linewidth=1.5, linestyle='--')
ax4.add_patch(circle)

# 6.5 热阻网络图
ax5 = axes[1, 1]
ax5.text(0.1, 0.95, '热阻网络模型', fontsize=14, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.1, 0.80, 'T_core ── R_cond ── T_surface ── R_conv ── T_amb', fontsize=12,
         fontfamily='monospace', transform=ax5.transAxes, color='#333')
ax5.text(0.1, 0.65, '', fontsize=12, transform=ax5.transAxes)
ax5.text(0.1, 0.55, f'传导热阻 R_cond: {R_cond:.2f} K/W', fontsize=11,
         transform=ax5.transAxes)
ax5.text(0.1, 0.45, f'对流热阻 R_conv (自然): {R_conv_air:.2f} K/W', fontsize=11,
         transform=ax5.transAxes)
ax5.text(0.1, 0.35, f'对流热阻 R_conv (风冷): {R_conv_forced:.2f} K/W', fontsize=11,
         transform=ax5.transAxes)
ax5.text(0.1, 0.25, f'热时间常数 τ (自然): {tau_air:.0f} s', fontsize=11,
         transform=ax5.transAxes)
ax5.text(0.1, 0.15, f'热时间常数 τ (风冷): {tau_forced:.0f} s', fontsize=11,
         transform=ax5.transAxes)

ax5.text(0.1, 0.02, '热平衡: P_loss = ΔT / R_th\nτ = R_th · C_th', fontsize=11,
         fontweight='bold', transform=ax5.transAxes, color='#333')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

# 6.6 允许电流 vs 冷却方式 (绝缘等级限制)
ax6 = axes[1, 2]
# 绝缘等级对应的允许温升
insulation_classes = {
    'A (105°C)': 80,
    'E (120°C)': 95,
    'B (130°C)': 105,
    'F (155°C)': 130,
    'H (180°C)': 155,
}

# 计算每种绝缘等级下的最大允许电流
def max_current(R_th, delta_T_max):
    return np.sqrt(delta_T_max / (R_coil_20 * R_th))

cooling_methods = ['自然冷却', '强制风冷', '水冷']
R_th_list = [R_th_air, R_th_forced, R_th_water]

x_pos = np.arange(len(insulation_classes))
width = 0.25

for i, (cooling, R_th) in enumerate(zip(cooling_methods, R_th_list)):
    I_max = [max_current(R_th, dt) for dt in insulation_classes.values()]
    bars = ax6.bar(x_pos + (i-1)*width, I_max, width, label=cooling, alpha=0.8)
    # 添加数值
    for j, (bar, im) in enumerate(zip(bars, I_max)):
        if im > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{im:.1f}', ha='center', fontsize=7, rotation=90)

ax6.set_xticks(x_pos)
ax6.set_xticklabels(list(insulation_classes.keys()), rotation=45, fontsize=9)
ax6.set_ylabel('最大允许电流 [A]')
ax6.set_title('不同绝缘等级与冷却方式下的\n最大允许电流', fontsize=14)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_dir = Path('examples/motor_simulation')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_dir / 'demo_03_joule_heating.png'), dpi=150, bbox_inches='tight')
plt.close()

print("✅ Demo 3: 绕组焦耳热与温升仿真完成")
print(f"   线圈电阻 (20°C): {R_coil_20:.4f} Ω")
print(f"   10A时铜损 (12线圈): {10**2*R_coil_20*12:.1f} W")
print(f"   热时间常数 (自然冷却): {tau_air:.0f} s")
print(f"   热时间常数 (强制风冷): {tau_forced:.0f} s")
