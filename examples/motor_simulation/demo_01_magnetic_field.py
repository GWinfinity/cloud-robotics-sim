"""
Demo 1: 电机磁场分布仿真 (Magnetic Field Simulation)
=====================================================
使用解析法和数值法计算电机横截面的磁场分布。
等效于 MagnetostaticsSolver 的核心功能。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyBboxPatch
from pathlib import Path

# ============================================================
# 1. 电机几何参数
# ============================================================
R_STATOR_OUTER = 0.08      # 定子外径 [m]
R_STATOR_INNER = 0.05      # 定子内径 [m]
R_ROTOR = 0.045            # 转子外径 [m]
R_SHAFT = 0.01             # 转轴半径 [m]
NUM_SLOTS = 12             # 定子槽数
NUM_POLES = 4              # 极数 (2对极)
AIR_GAP = R_STATOR_INNER - R_ROTOR  # 气隙

# 材料属性
MU_0 = 4 * np.pi * 1e-7    # 真空磁导率
MU_R_STEEL = 1000          # 硅钢片相对磁导率
MU_R_MAGNET = 1.05         # 永磁体相对磁导率
BR_MAGNET = 1.2            # 永磁体剩磁 [T] (NdFeB)

# 绕组参数
CURRENT_PER_TURN = 5.0     # 每匝电流 [A]
TURNS_PER_COIL = 20        # 每线圈匝数

# ============================================================
# 2. 建立计算网格 (2D 横截面)
# ============================================================
NX, NY = 200, 200
x = np.linspace(-R_STATOR_OUTER, R_STATOR_OUTER, NX)
y = np.linspace(-R_STATOR_OUTER, R_STATOR_OUTER, NY)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
THETA = np.arctan2(Y, X)

# ============================================================
# 3. 定义电机几何区域
# ============================================================
# 定子区域 (硅钢片)
stator_region = (R >= R_STATOR_INNER) & (R <= R_STATOR_OUTER)

# 转子区域 (硅钢片)
rotor_region = (R <= R_ROTOR) & (R > R_SHAFT)

# 气隙区域
airgap_region = (R > R_ROTOR) & (R < R_STATOR_INNER)

# 转轴区域
shaft_region = (R <= R_SHAFT)

# 磁导率分布
mu_r_map = np.ones((NY, NX), dtype=np.float64)
mu_r_map[stator_region] = MU_R_STEEL
mu_r_map[rotor_region] = MU_R_STEEL
mu_r_map[airgap_region] = 1.0    # 空气
mu_r_map[shaft_region] = MU_R_STEEL
mu_map = mu_r_map * MU_0

# ============================================================
# 4. 永磁体建模 (表贴式)
# ============================================================
# 在转子表面放置永磁体 (4极)
magnet_thickness = 0.003  # 磁钢厚度 [m]
magnet_angle = np.pi / NUM_POLES  # 每极角度跨距

magnet_region = np.zeros((NY, NX), dtype=bool)
for p in range(NUM_POLES):
    pole_angle = p * 2 * np.pi / NUM_POLES
    angle_start = pole_angle - magnet_angle / 2
    angle_end = pole_angle + magnet_angle / 2
    
    # 处理角度环绕
    if angle_start < -np.pi:
        mask = ((THETA >= angle_start + 2*np.pi) | (THETA <= angle_end)) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    elif angle_end > np.pi:
        mask = ((THETA >= angle_start) | (THETA <= angle_end - 2*np.pi)) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    else:
        mask = (THETA >= angle_start) & (THETA <= angle_end) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    magnet_region |= mask

# 磁化方向 (交替NS极)
magnet_M = np.zeros((NY, NX, 2), dtype=np.float64)
for p in range(NUM_POLES):
    pole_angle = p * 2 * np.pi / NUM_POLES
    angle_start = pole_angle - magnet_angle / 2
    angle_end = pole_angle + magnet_angle / 2
    
    if angle_start < -np.pi:
        mask = ((THETA >= angle_start + 2*np.pi) | (THETA <= angle_end)) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    elif angle_end > np.pi:
        mask = ((THETA >= angle_start) | (THETA <= angle_end - 2*np.pi)) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    else:
        mask = (THETA >= angle_start) & (THETA <= angle_end) & \
               (R >= R_ROTOR) & (R <= R_ROTOR + magnet_thickness)
    
    # 沿径向磁化，NS交替
    direction = 1 if p % 2 == 0 else -1
    magnet_M[mask, 0] = direction * BR_MAGNET / MU_0 * np.cos(THETA[mask])
    magnet_M[mask, 1] = direction * BR_MAGNET / MU_0 * np.sin(THETA[mask])

# ============================================================
# 5. 定子绕组电流
# ============================================================
# 简化: 在三相绕组中通入电流，产生旋转磁场
slot_angles = np.linspace(0, 2*np.pi, NUM_SLOTS, endpoint=False)
slot_radius = (R_STATOR_INNER + R_STATOR_OUTER) / 2

# 三相电流 (A相在0度, B相在120度, C相在240度)
# 三相绕组分布: A+/A- B+/B- C+/C-
winding_MMF = np.zeros((NY, NX), dtype=np.float64)

# 计算每槽安匝数
for i, angle in enumerate(slot_angles):
    # 三相分布: A相在槽0,3,6,9; B相在槽1,4,7,10; C相在槽2,5,8,11
    phase = i % 3
    direction = 1 if (i // 3) % 2 == 0 else -1
    
    if phase == 0:  # A相
        current = CURRENT_PER_TURN * TURNS_PER_COIL * direction
    elif phase == 1:  # B相
        current = -0.5 * CURRENT_PER_TURN * TURNS_PER_COIL * direction
    else:  # C相
        current = -0.5 * CURRENT_PER_TURN * TURNS_PER_COIL * direction
    
    # 将槽电流映射到网格 (高斯涂抹)
    dx = X - slot_radius * np.cos(angle)
    dy = Y - slot_radius * np.sin(angle)
    sigma = 0.005  # 涂抹宽度
    winding_MMF += current * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

# ============================================================
# 6. 求解标量磁势 (简化: 磁路法 + 叠加)
# ============================================================
# 这里简化处理，直接计算磁通密度:
# 永磁体贡献: B_magnet = μ₀ * M (在永磁体位置)
# 绕组贡献: B_winding = μ₀ * J × r / (2πr²)  (毕奥-萨伐尔)

# 6.1 永磁体产生的磁场
B_magnet_x = np.zeros((NY, NX), dtype=np.float64)
B_magnet_y = np.zeros((NY, NX), dtype=np.float64)
B_magnet_x[magnet_region] = MU_0 * magnet_M[magnet_region, 0]
B_magnet_y[magnet_region] = MU_0 * magnet_M[magnet_region, 1]

# 6.2 绕组产生的磁场 (基于电流的Biot-Savart)
B_winding_x = np.zeros((NY, NX), dtype=np.float64)
B_winding_y = np.zeros((NY, NX), dtype=np.float64)

# 对每个槽位置计算其磁场贡献
for i, angle in enumerate(slot_angles):
    phase = i % 3
    direction = 1 if (i // 3) % 2 == 0 else -1
    I = CURRENT_PER_TURN * TURNS_PER_COIL * direction
    
    sx = slot_radius * np.cos(angle)
    sy = slot_radius * np.sin(angle)
    
    # Biot-Savart: dB = μ₀/4π * Idl × r̂ / r²
    # 对于无限长直导线: B = μ₀*I/(2πr) * direction
    rx = X - sx
    ry = Y - sy
    r = np.sqrt(rx**2 + ry**2)
    r[r < 0.001] = 0.001  # 避免奇点
    
    # 电流沿z轴方向，磁场在xy平面内
    B_winding_x += -MU_0 * I / (2 * np.pi) * ry / r**2
    B_winding_y += MU_0 * I / (2 * np.pi) * rx / r**2

# 6.3 总磁场 (叠加)
B_total_x = B_magnet_x + B_winding_x
B_total_y = B_magnet_y + B_winding_y
B_total = np.sqrt(B_total_x**2 + B_total_y**2)

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# 7.1 电机几何结构
ax1 = axes[0, 0]
ax1.set_aspect('equal')

# 定子
stator = Circle((0, 0), R_STATOR_OUTER, fill=False, color='#555', linewidth=2, label='定子外径')
ax1.add_patch(stator)
stator_inner = Circle((0, 0), R_STATOR_INNER, fill=False, color='#888', linewidth=2, linestyle='--', label='定子内径')
ax1.add_patch(stator_inner)

# 定子槽
for angle in slot_angles:
    slot = Arc((0, 0), 2*slot_radius, 2*slot_radius, angle=np.degrees(angle)-5, 
               theta1=0, theta2=10, color='#333', linewidth=4)
    ax1.add_patch(slot)

# 转子
rotor = Circle((0, 0), R_ROTOR, fill=False, color='#c44', linewidth=2, label='转子')
ax1.add_patch(rotor)

# 永磁体
for p in range(NUM_POLES):
    pole_angle = p * 2 * np.pi / NUM_POLES
    color = '#d44' if p % 2 == 0 else '#44d'
    for theta_offset in np.linspace(-magnet_angle/2, magnet_angle/2, 10):
        r = R_ROTOR + magnet_thickness/2
        ax1.plot(r*np.cos(pole_angle+theta_offset), r*np.sin(pole_angle+theta_offset), 
                'o', color=color, markersize=2)

# 转轴
shaft = Circle((0, 0), R_SHAFT, color='#999', alpha=0.5)
ax1.add_patch(shaft)

ax1.set_xlim(-R_STATOR_OUTER*1.1, R_STATOR_OUTER*1.1)
ax1.set_ylim(-R_STATOR_OUTER*1.1, R_STATOR_OUTER*1.1)
ax1.set_title('电机横截面几何结构 (4极12槽)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

# 7.2 磁导率分布
ax2 = axes[0, 1]
im2 = ax2.imshow(mu_r_map, extent=[-R_STATOR_OUTER, R_STATOR_OUTER, -R_STATOR_OUTER, R_STATOR_OUTER],
                 origin='lower', cmap='viridis', norm='log')
ax2.set_title('相对磁导率分布', fontsize=14)
plt.colorbar(im2, ax=ax2, label='μr')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')

# 7.3 磁通密度幅值分布
ax3 = axes[1, 0]
# 只显示电机内部区域
B_plot = np.copy(B_total)
B_plot[R > R_STATOR_OUTER] = np.nan
im3 = ax3.imshow(B_plot, extent=[-R_STATOR_OUTER, R_STATOR_OUTER, -R_STATOR_OUTER, R_STATOR_OUTER],
                 origin='lower', cmap='hot', vmin=0, vmax=2.0)
ax3.set_title('磁通密度 |B| 分布 [T]', fontsize=14)
plt.colorbar(im3, ax=ax3, label='|B| [T]')
ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')

# 7.4 磁力线 (流线图)
ax4 = axes[1, 1]
skip = 4
ax4.streamplot(X[::skip, ::skip], Y[::skip, ::skip],
               B_total_x[::skip, ::skip], B_total_y[::skip, ::skip],
               color=np.log(B_total[::skip, ::skip] + 1e-10),
               cmap='plasma', linewidth=1.2, density=1.5)
ax4.set_xlim(-R_STATOR_OUTER, R_STATOR_OUTER)
ax4.set_ylim(-R_STATOR_OUTER, R_STATOR_OUTER)
ax4.set_title('磁力线分布 (流线)', fontsize=14)
ax4.set_aspect('equal')
ax4.set_xlabel('x [m]')
ax4.set_ylabel('y [m]')

plt.tight_layout()
out_dir = Path('examples/motor_simulation')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_dir / 'demo_01_magnetic_field.png'), dpi=150, bbox_inches='tight')
plt.close()

print("✅ Demo 1: 电机磁场分布仿真完成")
print(f"   最大磁通密度: {B_total.max():.3f} T")
print(f"   气隙平均磁密: {np.mean(B_total[airgap_region]):.3f} T")
print(f"   定子轭部磁密: {np.mean(B_total[stator_region & (R > (R_STATOR_INNER+R_STATOR_OUTER)/2)]):.3f} T")
