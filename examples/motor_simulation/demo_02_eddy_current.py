"""
Demo 2: 涡流与集肤效应仿真 (Eddy Current & Skin Effect)
=========================================================
计算交变磁场在导体中的涡流分布和集肤深度。
等效于 EddyCurrentSolver 的核心功能。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 物理参数
# ============================================================
MU_0 = 4 * np.pi * 1e-7    # 真空磁导率 [H/m]

# 导体材料: 铜
SIGMA_CU = 5.96e7           # 电导率 [S/m]
MU_R_CU = 1.0               # 相对磁导率

# 导体材料: 硅钢
SIGMA_STEEL = 2.0e6         # 电导率 [S/m]
MU_R_STEEL = 1000           # 相对磁导率

# 导体材料: 铝
SIGMA_AL = 3.5e7            # 电导率 [S/m]
MU_R_AL = 1.0               # 相对磁导率

# 频率范围
frequencies = np.logspace(0, 5, 50)  # 1 Hz ~ 100 kHz

# ============================================================
# 2. 计算集肤深度
# ============================================================
# 公式: δ = 1 / sqrt(π * f * μ * σ)

def skin_depth(f, mu_r, sigma):
    """计算集肤深度 [m]"""
    return 1 / np.sqrt(np.pi * f * mu_r * MU_0 * sigma)

delta_Cu = skin_depth(frequencies, MU_R_CU, SIGMA_CU)
delta_steel = skin_depth(frequencies, MU_R_STEEL, SIGMA_STEEL)
delta_Al = skin_depth(frequencies, MU_R_AL, SIGMA_AL)

# ============================================================
# 3. 1D 涡流分布计算
# ============================================================
# 在导体半空间中的涡流分布: J(x) = J₀ * exp(-x/δ) * cos(ωt - x/δ)

x_1d = np.linspace(0, 0.05, 500)  # 0~50mm 深度

# 在 50Hz 下的涡流分布
f_50 = 50.0
delta_Cu_50 = skin_depth(f_50, MU_R_CU, SIGMA_CU)
delta_steel_50 = skin_depth(f_50, MU_R_STEEL, SIGMA_STEEL)

J_Cu_50 = np.exp(-x_1d / delta_Cu_50) * np.cos(-x_1d / delta_Cu_50)
J_steel_50 = np.exp(-x_1d / delta_steel_50) * np.cos(-x_1d / delta_steel_50)

# ============================================================
# 4. 2D 涡流分布 (圆形导体截面)
# ============================================================
# 半径为 5mm 的圆导体在交变磁场中的涡流

NX, NY = 150, 150
radius = 0.005  # 导体半径 [m]
x = np.linspace(-radius*1.2, radius*1.2, NX)
y = np.linspace(-radius*1.2, radius*1.2, NY)
X, Y = np.meshgrid(x, y)
R_grid = np.sqrt(X**2 + Y**2)

# 频率 1kHz 时的涡流 (假设外部磁场沿z方向，随时间正弦变化)
f_1k = 1000.0
delta_1k = skin_depth(f_1k, MU_R_CU, SIGMA_CU)
omega_1k = 2 * np.pi * f_1k

# 简化: 涡流密度幅值随半径变化
# 对于圆柱导体，内部磁场为: B(r) = B₀ * J₀(kr) / J₀(kR)
# 涡流密度: J_φ(r) = -jωσ * A_z(r)
# 这里简化为指数衰减 + 集肤效应分布
k = np.sqrt(-1j * omega_1k * MU_0 * SIGMA_CU)
# 归一化半径
r_norm = R_grid / radius
# 简化涡流分布 (集肤效应)
J_phi = np.zeros_like(R_grid)
inside = R_grid <= radius
J_phi[inside] = (R_grid[inside] / radius) * np.exp(-(radius - R_grid[inside]) / delta_1k)
J_phi = J_phi / J_phi.max()  # 归一化

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5.1 集肤深度 vs 频率
ax1 = axes[0, 0]
ax1.loglog(frequencies, delta_Cu * 1000, 'b-', linewidth=2, label='铜')
ax1.loglog(frequencies, delta_steel * 1000, 'r-', linewidth=2, label='硅钢 (μr=1000)')
ax1.loglog(frequencies, delta_Al * 1000, 'g-', linewidth=2, label='铝')
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('频率 [Hz]')
ax1.set_ylabel('集肤深度 [mm]')
ax1.set_title('集肤深度 vs 频率', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotation at 50Hz
ax1.annotate(f'铜@50Hz: {delta_Cu_50*1000:.2f}mm', 
             xy=(50, delta_Cu_50*1000), xytext=(200, delta_Cu_50*1000*5),
             arrowprops=dict(arrowstyle='->'), fontsize=9)
ax1.annotate(f'硅钢@50Hz: {delta_steel_50*1000:.3f}mm',
             xy=(50, delta_steel_50*1000), xytext=(200, delta_steel_50*1000*5),
             arrowprops=dict(arrowstyle='->'), fontsize=9)

# 5.2 50Hz涡流深度分布
ax2 = axes[0, 1]
ax2.plot(x_1d*1000, J_Cu_50, 'b-', linewidth=2, label='铜')
ax2.plot(x_1d*1000, J_steel_50, 'r-', linewidth=2, label='硅钢')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('深度 [mm]')
ax2.set_ylabel('归一化涡流密度')
ax2.set_title('50Hz 涡流密度沿深度分布', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5.3 不同频率的集肤深度对比 (条形图)
ax3 = axes[0, 2]
freqs_plot = [50, 400, 1000, 10000]
delta_Cu_plot = [skin_depth(f, MU_R_CU, SIGMA_CU)*1000 for f in freqs_plot]
delta_steel_plot = [skin_depth(f, MU_R_STEEL, SIGMA_STEEL)*1000 for f in freqs_plot]

x_pos = np.arange(len(freqs_plot))
width = 0.35
bars1 = ax3.bar(x_pos - width/2, delta_Cu_plot, width, label='铜', color='#c66')
bars2 = ax3.bar(x_pos + width/2, delta_steel_plot, width, label='硅钢', color='#66c')

ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{f}Hz' for f in freqs_plot])
ax3.set_ylabel('集肤深度 [mm]')
ax3.set_title('典型频率下的集肤深度', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

# 5.4 2D涡流分布 (圆形导体截面)
ax4 = axes[1, 0]
im4 = ax4.imshow(J_phi, extent=[-radius*1000, radius*1000, -radius*1000, radius*1000],
                 origin='lower', cmap='hot')
ax4.set_title(f'1kHz 圆导体截面涡流分布 (铜, δ={delta_1k*1000:.2f}mm)', fontsize=14)
plt.colorbar(im4, ax=ax4, label='归一化涡流密度')
ax4.set_xlabel('x [mm]')
ax4.set_ylabel('y [mm]')

# 5.5 涡流损耗 vs 频率
ax5 = axes[1, 1]
# 涡流损耗: P ∝ B²f²/ρ (近似)
# 比较不同材料的归一化损耗
f_plot = np.logspace(1, 4, 100)
B_0 = 1.0  # 1T

# Steinmetz 公式: P = k * f^α * B^β
# 对于硅钢: α≈1.8, β≈2.0
loss_steel = B_0**2 * f_plot**1.8 * 1e-3  # W/kg 近似
loss_copper = B_0**2 * f_plot**2 * 5e-4   # W/kg 近似 (纯涡流)

ax5.loglog(f_plot, loss_steel, 'r-', linewidth=2, label='硅钢片 (涡流+磁滞)')
ax5.loglog(f_plot, loss_copper, 'b-', linewidth=2, label='铜 (纯涡流)')
ax5.set_xlabel('频率 [Hz]')
ax5.set_ylabel('单位质量损耗 [W/kg]')
ax5.set_title('涡流损耗 vs 频率 (B=1T)', fontsize=14)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 5.6 集肤效应的物理意义注解
ax6 = axes[1, 2]
ax6.text(0.1, 0.95, '集肤效应关键公式', fontsize=14, fontweight='bold',
         transform=ax6.transAxes)
ax6.text(0.1, 0.80, '集肤深度:', fontsize=12, transform=ax6.transAxes)
ax6.text(0.1, 0.70, 'δ = 1 / √(π·f·μ·σ)', fontsize=12, 
         fontfamily='monospace', transform=ax6.transAxes, color='#333')
ax6.text(0.1, 0.55, '涡流穿透深度:', fontsize=12, transform=ax6.transAxes)
ax6.text(0.1, 0.45, 'J(x) = J₀·exp(-x/δ)·cos(ωt-x/δ)', fontsize=12,
         fontfamily='monospace', transform=ax6.transAxes, color='#333')
ax6.text(0.1, 0.30, '涡流损耗密度:', fontsize=12, transform=ax6.transAxes)
ax6.text(0.1, 0.20, 'p = |J|²/σ (W/m³)', fontsize=12,
         fontfamily='monospace', transform=ax6.transAxes, color='#333')

ax6.text(0.1, 0.05, '工程意义:', fontsize=12, fontweight='bold', 
         transform=ax6.transAxes, color='#c33')
ax6.text(0.1, -0.1, '高频时电流集中在导体表面，\n增加AC电阻，产生额外损耗。\n对策: 使用利兹线/硅钢片叠压。', 
         fontsize=10, transform=ax6.transAxes, color='#666')
ax6.set_xlim(0, 1)
ax6.set_ylim(-0.2, 1)
ax6.axis('off')

plt.tight_layout()
out_dir = Path('examples/motor_simulation')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_dir / 'demo_02_eddy_current.png'), dpi=150, bbox_inches='tight')
plt.close()

print("✅ Demo 2: 涡流与集肤效应仿真完成")
print(f"   铜@50Hz 集肤深度: {delta_Cu_50*1000:.2f} mm")
print(f"   硅钢@50Hz 集肤深度: {delta_steel_50*1000:.3f} mm")
print(f"   铜@1kHz 集肤深度: {delta_1k*1000:.2f} mm")
