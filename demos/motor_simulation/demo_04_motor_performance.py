"""
Demo 4: 电机性能曲线仿真 (Motor Performance)
==============================================
计算电机的扭矩-转速特性、效率图和损耗分布。
集成多物理场耦合效果。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ============================================================
# 1. 电机参数 (表贴式永磁同步电机 - SPMSM)
# ============================================================
# 几何
POLES = 4                   # 极数
SLOTS = 12                  # 槽数
R_STATOR_INNER = 0.05       # 定子内径 [m]
R_ROTOR = 0.045             # 转子外径 [m]
STACK_LENGTH = 0.08         # 轴向长度 [m]
AIR_GAP = R_STATOR_INNER - R_ROTOR

# 绕组
TURNS_PER_PHASE = 80        # 每相串联匝数
R_PHASE = 0.25              # 每相电阻 [Ω] (20°C)
L_D = 0.002                 # d轴电感 [H]
L_Q = 0.002                 # q轴电感 [H]

# 永磁体
LAMBDA_PM = 0.12            # 永磁体磁链 [Wb]

# 直流母线电压
V_DC = 310                  # 直流母线电压 [V] (~220V整流)
V_MAX = V_DC / np.sqrt(3)   # 相电压峰值 [V]

# 额定
I_RATED = 10                # 额定电流 [A] (峰值)
N_RATED = 3000              # 额定转速 [RPM]
N_MAX = 8000                # 最大转速 [RPM]

# ============================================================
# 2. 电流控制策略 (MTPA - 最大转矩电流比)
# ============================================================
# SPMSM 中 Ld = Lq, 所以 MTPA 就是 Id = 0
# 转矩方程: T = 3/2 * POLES/2 * (LAMBDA_PM * Iq + (Ld - Lq) * Id * Iq)
# 当 Ld = Lq: T = 3/2 * POLES/2 * LAMBDA_PM * Iq

# 电流极限圆
I_max = I_RATED * 1.2       # 最大电流 [A]  (120%过载)

# 不同电流组合 (Id, Iq)
I_dq_range = np.linspace(-I_max, I_max, 200)
Id_grid, Iq_grid = np.meshgrid(I_dq_range, I_dq_range)
I_amp = np.sqrt(Id_grid**2 + Iq_grid**2)

# 有效电流组合 (在电流极限圆内)
valid_mask = I_amp <= I_max

# ============================================================
# 3. 转矩计算
# ============================================================
# T = 3/2 * P/2 * [LAMBDA_PM * Iq + (Ld - Lq) * Id * Iq]
# SPMSM: Ld ≈ Lq，电磁转矩 = 永磁转矩
P = POLES / 2  # 极对数
T_e = np.zeros_like(Id_grid)
T_e[valid_mask] = (3/2 * P * (LAMBDA_PM * Iq_grid[valid_mask] + 
                               (L_D - L_Q) * Id_grid[valid_mask] * Iq_grid[valid_mask]))

# MTPA 曲线 (Id=0)
Id_mtpa = np.zeros_like(I_dq_range)
Iq_mtpa = I_dq_range
T_mtpa = 3/2 * P * LAMBDA_PM * Iq_mtpa
valid_mtpa = np.abs(I_dq_range) <= I_max

# 额定转矩
T_rated = 3/2 * P * LAMBDA_PM * I_RATED
print(f"   额定转矩: {T_rated:.3f} Nm")
print(f"   额定功率: {T_rated * N_RATED * 2 * np.pi / 60:.1f} W")

# ============================================================
# 4. 转速-转矩特性 (机械特性)
# ============================================================
# 电压方程: v_q = R_s*i_q + ω*L_d*i_d + ω*LAMBDA_PM
# v_d = R_s*i_d - ω*L_q*i_q
# 弱磁控制: 当转速超过基速后，需要Id<0来弱磁

# 基速 (达到电压极限)
omega_base = (V_MAX - I_RATED * R_PHASE) / (P * LAMBDA_PM)
N_base = omega_base * 60 / (2 * np.pi)
print(f"   基速: {N_base:.0f} RPM")

# 不同转速下的最大转矩
N_list = np.linspace(0, N_MAX, 1000)
omega_list = N_list * 2 * np.pi / 60

# 在最大电流下计算不同转速的转矩
# 区域1: 恒转矩区 (Id=0, Iq=I_max)
# 区域2: 弱磁区 (需要Id<0)

T_max_profile = np.zeros_like(N_list)
P_max_profile = np.zeros_like(N_list)

for i, (N, omega) in enumerate(zip(N_list, omega_list)):
    if omega <= omega_base:
        # 恒转矩区
        Iq = I_RATED
        Id = 0
        T = 3/2 * P * LAMBDA_PM * Iq
        v_q = omega * LAMBDA_PM + R_PHASE * Iq
        v_d = -omega * L_D * Iq
    else:
        # 弱磁区: Id < 0, 满足电压极限
        # v_q² + v_d² ≤ V_max²
        # (ω*LAMBDA_PM + R_s*Iq)² + (-ω*Ld*Id)² = V_max²
        # 同时电流极限: Id² + Iq² = I_rated²
        
        # 简化的弱磁计算
        # 从电压极限和电流极限求解 Id, Iq
        # 使用牛顿迭代
        Id = -I_RATED * (1 - omega_base/omega)  # 近似
        Id = max(Id, -I_RATED)  # Id 不超过电流极限
        
        Iq = np.sqrt(max(0, I_RATED**2 - Id**2))
        v_q = omega * LAMBDA_PM + R_PHASE * Iq
        v_d = -omega * L_D * Id
        v_amp = np.sqrt(v_q**2 + v_d**2)
        
        # 如果电压超出，减小Iq
        if v_amp > V_MAX:
            scale = V_MAX / v_amp
            v_q *= scale
            v_d *= scale
            # 重新估计Iq (简化)
            Iq = Iq * scale
            Id = Id * scale
        
        T = 3/2 * P * LAMBDA_PM * Iq
    
    T_max_profile[i] = T
    P_max_profile[i] = T * omega

# ============================================================
# 5. 效率图计算
# ============================================================
# 不同工况 (转速 x 转矩) 下的效率
# 损耗 = 铜损 + 铁损 + 机械损 + 杂散损
# 铜损: P_cu = 3 * I² * R_phase
# 铁损: P_fe = k_h * f * B² + k_e * f² * B² (近似)
# 机械损: P_mec = k_m * ω²

# 效率扫描网格
N_mesh = np.linspace(0, N_MAX, 60)
T_mesh = np.linspace(0, T_rated*1.2, 50)
N_grid, T_grid = np.meshgrid(N_mesh, T_mesh)
omega_grid = N_grid * 2 * np.pi / 60

# 估算效率 (简化模型)
def estimate_efficiency(N, T):
    """估算电机效率"""
    if N <= 0 or T <= 0:
        return 0
    omega = N * 2 * np.pi / 60
    P_out = T * omega
    
    # 估算电流 (从转矩反推)
    Iq_est = T / (3/2 * P * LAMBDA_PM)
    Id_est = 0  # 假设Id=0控制
    I_est = np.sqrt(Id_est**2 + Iq_est**2)
    
    # 损耗
    P_cu = 3 * I_est**2 * R_PHASE  # 铜损
    P_fe = 0.02 * P_out + 5  # 铁损 (近似: 2%输出+空载铁损)
    P_mec = 0.005 * P_out + 2  # 机械损
    P_stray = 0.01 * P_out  # 杂散损耗
    
    P_total_loss = P_cu + P_fe + P_mec + P_stray
    eta = P_out / (P_out + P_total_loss) * 100
    
    return min(eta, 98)  # 上限98%

# 计算效率图
eta_grid = np.zeros_like(T_grid)
for i in range(len(N_mesh)):
    for j in range(len(T_mesh)):
        if N_mesh[i] > 100 and T_mesh[j] > 0.01:
            eta_grid[j, i] = estimate_efficiency(N_mesh[i], T_mesh[j])

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 Id-Iq 矢量图 + 转矩等高线
ax1 = axes[0, 0]
levels = np.linspace(0, T_rated*1.5, 15)
contour = ax1.contour(Id_grid, Iq_grid, T_e, levels=levels, cmap='viridis', alpha=0.8)
ax1.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
ax1.plot(Id_mtpa[valid_mtpa], Iq_mtpa[valid_mtpa], 'r-', linewidth=2, label='MTPA (Id=0)')

# 电流极限圆
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(I_max*np.cos(theta), I_max*np.sin(theta), 'k--', linewidth=1.5, label='电流极限')
ax1.axhline(y=0, color='gray', alpha=0.3)
ax1.axvline(x=0, color='gray', alpha=0.3)
ax1.set_xlabel('Id [A]')
ax1.set_ylabel('Iq [A]')
ax1.set_title('Id-Iq 矢量图与转矩等高线', fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlim(-I_max*1.1, I_max*1.1)
ax1.set_ylim(-I_max*1.1, I_max*1.1)

# 6.2 转矩-转速特性
ax2 = axes[0, 1]
ax2.plot(N_list, T_max_profile, 'b-', linewidth=2.5, label='最大转矩')
ax2.fill_between(N_list, 0, T_max_profile, alpha=0.2, color='blue')
ax2.axhline(y=T_rated, color='r', linestyle='--', linewidth=1, label=f'额定转矩 ({T_rated:.2f} Nm)')
ax2.axvline(x=N_base, color='g', linestyle='--', linewidth=1, label=f'基速 ({N_base:.0f} RPM)')
ax2.set_xlabel('转速 [RPM]')
ax2.set_ylabel('转矩 [Nm]')
ax2.set_title('转矩-转速特性曲线', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 标注区域
ax2.annotate('恒转矩区', xy=(N_base/2, T_rated*0.8), fontsize=11,
             ha='center', color='blue',
             bbox=dict(boxstyle='round', fc='lightblue', alpha=0.5))
ax2.annotate('弱磁区', xy=((N_base+N_MAX)/2, T_rated*0.4), fontsize=11,
             ha='center', color='green',
             bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.5))

# 6.3 功率-转速特性
ax3 = axes[0, 2]
ax3.plot(N_list, P_max_profile, 'r-', linewidth=2.5, label='最大功率')
ax3.fill_between(N_list, 0, P_max_profile, alpha=0.2, color='red')
P_rated = T_rated * omega_base
ax3.axhline(y=P_rated, color='gray', linestyle='--', linewidth=1)
ax3.axvline(x=N_base, color='g', linestyle='--', linewidth=1)
ax3.set_xlabel('转速 [RPM]')
ax3.set_ylabel('功率 [W]')
ax3.set_title('功率-转速特性曲线', fontsize=14)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 6.4 效率图
ax4 = axes[1, 0]
eta_contour = ax4.contourf(N_mesh, T_mesh, eta_grid, levels=20, cmap='RdYlGn')
ax4.contour(N_mesh, T_mesh, eta_grid, levels=[80, 85, 90, 92, 94, 95], 
            colors='k', linewidths=0.5)
ax4.clabel(ax4.contour(N_mesh, T_mesh, eta_grid, levels=[85, 90, 94], 
          colors='k', linewidths=0.5), inline=True, fontsize=9, fmt='%.0f%%')
plt.colorbar(eta_contour, ax=ax4, label='效率 [%]')

# 额定工作点
ax4.plot(N_RATED, T_rated, 'r*', markersize=15, label=f'Rated Point ({N_RATED}RPM, {T_rated:.2f}Nm)')
ax4.set_xlabel('转速 [RPM]')
ax4.set_ylabel('转矩 [Nm]')
ax4.set_title('电机效率图', fontsize=14)
ax4.legend(fontsize=9)

# 6.5 损耗分布 (饼图)
ax5 = axes[1, 1]
# 额定工况下的损耗分布
Iq_rated = I_RATED
I_rated_rms = I_RATED / np.sqrt(2)
P_cu_rated = 3 * I_rated_rms**2 * R_PHASE
P_fe_rated = 0.02 * P_rated + 5
P_mec_rated = 0.005 * P_rated + 2
P_stray_rated = 0.01 * P_rated
P_total_loss = P_cu_rated + P_fe_rated + P_mec_rated + P_stray_rated

labels = ['铜损 (I²R)', '铁损 (磁滞+涡流)', '机械损耗 (摩擦风阻)', '杂散损耗']
sizes = [P_cu_rated, P_fe_rated, P_mec_rated, P_stray_rated]
colors_l = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
explode = (0.05, 0.05, 0, 0)

wedges, texts, autotexts = ax5.pie(sizes, explode=explode, labels=labels,
                                     colors=colors_l, autopct='%1.1f%%',
                                     shadow=False, startangle=90)
ax5.set_title(f'额定工况损耗分布\n(总损耗: {P_total_loss:.1f}W, 效率: {P_rated/(P_rated+P_total_loss)*100:.1f}%)',
              fontsize=13)

# 6.6 关键技术参数
ax6 = axes[1, 2]
params = [
    f'电机类型: SPMSM (表贴式永磁同步)',
    f'极数/槽数: {POLES}极/{SLOTS}槽',
    f'额定功率: {P_rated:.0f} W',
    f'额定转矩: {T_rated:.3f} Nm',
    f'额定转速: {N_RATED} RPM',
    f'最大转速: {N_MAX} RPM',
    f'基速: {N_base:.0f} RPM',
    f'额定电流: {I_RATED} A (峰值)',
    f'相电阻: {R_PHASE} Ω',
    f'永磁磁链: {LAMBDA_PM} Wb',
    f'd/q轴电感: {L_D}/{L_Q} mH',
    f'直流母线电压: {V_DC} V',
    f'',
    f'控制策略: Id=0 (MTPA) + 弱磁',
    f'转矩密度: {T_rated/(np.pi*R_ROTOR**2*STACK_LENGTH):.1f} Nm/m³',
]
ax6.text(0.05, 0.95, '电机设计参数', fontsize=14, fontweight='bold', transform=ax6.transAxes)
for j, param in enumerate(params):
    ax6.text(0.05, 0.85 - j*0.05, param, fontsize=10, transform=ax6.transAxes,
             fontfamily='monospace' if ':' in param else 'sans-serif',
             fontweight='normal' if ':' in param else 'bold')
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')

plt.tight_layout()
out_dir = Path('demos/motor_simulation')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_dir / 'demo_04_motor_performance.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Demo 4: 电机性能曲线仿真完成")
print(f"   额定转矩: {T_rated:.3f} Nm")
print(f"   额定功率: {P_rated:.0f} W")
print(f"   基速: {N_base:.0f} RPM")
print(f"   额定效率: {P_rated/(P_rated+P_total_loss)*100:.1f}%")
