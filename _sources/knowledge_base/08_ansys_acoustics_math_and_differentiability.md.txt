# Ansys Mechanical + Fluent / LS-DYNA 声学仿真：数学原理推导与可微性判定

> 知识库条目 08。衔接前一篇《主流 3D 资产生成服务调研》与声学求解器插件实现
> （`plugins/solvers/acoustics`），给出三大商业求解器声学链路的完整数学推导，
> 并对每个算子做可微性（autodiff）判定——既解释商业软件的梯度能力边界，
> 也对照本项目 quadrants/Taichi AD 的实际限制（见
> `scripts/quadrants_issue_thermal_source_autodiff.md`）。

## 1. 控制方程：从守恒律到声学方程

### 1.1 出发点是可压缩 Navier–Stokes

质量、动量守恒（忽略体力）：

```
∂ρ/∂t + ∇·(ρu) = 0                                    (1)
∂(ρu_i)/∂t + ∂(ρu_i u_j)/∂x_j = -∂p/∂x_i + ∂τ_ij/∂x_j  (2)
```

声学量 = 均值 + 涨落：`p = p₀ + p'`，`ρ = ρ₀ + ρ'`，`u = u₀ + u'`。
对**静止均匀介质**（u₀ = 0, ρ₀, c₀ 常数）做一阶线性化（保留 ρ'u' 二阶项为零）：

```
∂ρ'/∂t + ρ₀∇·u' = 0                                    (3)
ρ₀ ∂u'/∂t = -∇p'                                       (4)
```

配合等熵状态方程 `p' = c₀²ρ'`（`c₀² = (∂p/∂ρ)_s`），(3) 对时间求导代入 (4)：

```
∂²p'/∂t² − c₀²∇²p' = 0                                 (5)  线性波动方程
```

这就是 **Mechanical 声学单元、LS-DYNA `*MAT_ACOUSTIC`、我们 leapfrog 插件**
共同求解的主方程。频域下取 `p'(x,t) = Re{p̂(x)e^{iωt}}` 得 **Helmholtz 方程**：

```
∇²p̂ + k²p̂ = 0,   k = ω/c₀                              (6)
```

这是 Mechanical 谐响应声学、LS-DYNA `*FREQUENCY_DOMAIN_ACOUSTIC_BEM/FEM` 的主方程。

### 1.2 有流动时：Lighthill 声学类比

有平均流时线性化得到 Lilley/Goldstein 型方程，工程上更常用 **Lighthill 类比**：
把 (1)(2) 精确重排成以波动算子作用在 ρ' 上的非齐次方程（**不做任何近似**，只是恒等变形）：

```
∂²ρ'/∂t² − c₀²∇²ρ' = ∂²T_ij/∂x_i∂x_j,   T_ij = ρu_i u_j + (p' − c₀²ρ')δ_ij − τ_ij   (7)
```

右端 `T_ij`（Lighthill 应力张量）在湍流剪切层内显著、在其外可忽略——这就是
"近场 CFD 算源、远场波动方程传播"的**混合法**（hybrid approach）的理论基础。

### 1.3 Ffowcs Williams–Hawkings（FWH）：把固体边界变成面源

对控制面 `f(x,t)=0`（`f>0` 为流体域）用广义函数延拓 (7)，分部积分后固体运动
产生的声归结为两类**面源**（Ffowcs Williams & Hawkings 1969）：

```
□²p' = ∂²/∂x_i∂x_j [T_ij H(f)]            − ∂/∂x_i [L_i δ(f)]  + ∂/∂t [Q δ(f)]   (8)
                          四极体源(湍流)      偶极面源(载荷)      单极面源(厚度/位移)
L_i = (p' δ_ij − τ_ij)n_j + ρu_i(u_n − v_n),   Q = ρ(u_n − v_n) + ρ₀v_n
```

自由场 Green 函数 `G = δ(t − τ − R/c₀)/(4πR)`（`R = |x − y|`）卷积得**时域积分公式**：

```
p'(x,t) = 1/(4π) ∫_S [ (1/R) ∂Q/∂t |τ* + (1/R²)·... dipole terms ... ] dS        (9)
```

`τ*` 为推迟时间（retarded time）。Fluent 的 FWH 模型 + 远场接收器就是 (9) 的数值实现
（forward-time projection 避免存储全部源历史）。

### 1.4 Kirchhoff 积分：从近场解外推远场

若近场已求得波动方程解 `p', ∂p'/∂n` 于封闭面 S 上，Kirchhoff–Helmholtz 积分直接给出
面外任意点压力：

```
p'(x,t) = ∫_S [ G·∂p'/∂n − p'·∂G/∂n ] dS                                            (10)
```

Fluent 波动方程法的远场工具、LS-DYNA BEM 的 Kirchhoff 选项都源于此。

## 2. 各求解器的离散化

### 2.1 Ansys Mechanical（APDL）：FEM 声学单元

对 (6) 加权余量 + Green 第一恒等式得弱式（`δp` 为权函数）：

```
∫_Ω ∇δp·∇p̂ dV − k² ∫_Ω δp·p̂ dV − ∫_Γ δp·(∂p̂/∂n) dS = 0                              (11)
```

单元离散 `p̂ = N p̃`（`N` 形函数）代入得单元矩阵：

```
(K_e − k²M_e) p̃ = f_e,   K_e = ∫∇Nᵀ∇N dV,  M_e = ∫NᵀN dV                            (12)
```

- `FLUID30`（3D 8 节点线性）、`FLUID220/221`（20/10 节点高阶）：就是 (12) 的实现，
  高阶单元在同样网格密度下相位误差更小（5 格/波长 vs 10 格/波长）；
- **FSI 界面**：结构振动速度作为 Neumann 边界 `∂p̂/∂n = −ρ₀ ω² u_n` 进入 `f_e`，
  声压反作用为结构载荷（强耦合）；
- **无限域截断**：`FLUID129/130` 无限元（径向呈指数衰减插值）或 PML（复坐标伸展，
  `FLUID243/244`），等价于在 (11) 的 Γ 上施加无反射条件；
- 求解器：谐响应直接/模态叠加（`HARFRQ`），或瞬态 Newmark。

### 2.2 Ansys Fluent：气动声学

| 模型 | 数学对象 | 关键点 |
|---|---|---|
| 直接法 | 可压缩瞬态 N–S (1)(2)，低耗散格式 | 分辨声波需网格/时间步满足声学 CFL，代价极高 |
| **FWH** | (9) 的时域积分 | 需瞬态流场；仅开放空间；面网格分辨率决定最高可信频率 |
| 宽频噪声源 | 稳态 RANS → Proudman/Lighthill 源强云图 | 统计量，无相位 → 不能合成远场 SPL |
| **波动方程法** | Ewert & Schröder 声学扰动方程（APE）：`∂p'/∂t + ρc∇·(c u'/?)` 类线性欧拉展开 + 海绵层 | 不可压缩流场算源、同网格传播；Kirchhoff 积分 (10) 外推远场 |

### 2.3 LS-DYNA：两条路线

**(a) 显式 FEM 声学（时域，强耦合）**：(5) 的空间 FEM + 中心差分（与结构显式积分
共用时间步，FSI 在单元层面耦合）——冲击、爆炸、水下（USA 模块）。

**(b) 频域卡片族（NVH 弱耦合）**：

```
结构瞬态响应 (*FREQUENCY_DOMAIN_SSD) → 表面速度 v(t) --FFT--> ṽ(f)
   → BEM 边界积分:  C(x)p̂(x) = ∫_S [ p̂ ∂G/∂n − G ∂p̂/∂n ] dS     (13)
   → 远场点声压；FEM 变体 (*FREQUENCY_DOMAIN_ACOUSTIC_FEM) 解腔体 (6)
```

BEM 只需表面网格且天然满足无穷远条件；`Rayleigh`（每单元独立活塞近似）与
`Kirchhoff`（FEM 过渡层 + 积分）是无需求解线性系统的快速近似。

## 3. 可微性判定

记号：● = 天然可微（AD 直接适用）；◐ = 可微但需特殊处理（伴随法/隐函数/
checkpointing/平滑化）；○ = 实践中不可微或需另辟蹊径。

### 3.1 按算子判定

| 算子 | 数学结构 | 梯度路径 | 判定 |
|---|---|---|---|
| FEM 组装 (12) | K、M 为材料参数（c、ρ）与**网格坐标**的显式函数 | ∂K/∂θ、∂M/∂θ 解析可得；形状导数需网格灵敏度 | ● |
| Helmholtz 求解 | 稀疏线性系统 A(k)p̂ = f | 线性：∂p̂/∂θ = A⁻¹(∂f/∂θ − ∂A/∂θ·p̂) | ● |
| 模态/特征值分析 | (K − ω²M)φ = 0 | 特征值导数经典结果 ∂ω/∂θ = φᵀ(∂K/∂θ − ω²∂M/∂θ)φ/(2ωM̃) | ● |
| 时域显式步进（leapfrog/Newmark） | 线性递推 x_{n+1} = A x_n + b_n | 递推可微；内存爆炸 → checkpointing（revolve） | ●（配 checkpointing） |
| **FFT** | 线性正交变换 | 逐项可微（幅度谱在零处不可微，工程无碍） | ● |
| FWH 积分 (9) | 对源历史的线性泛函 + 推迟时间 | 对源项/接收器位置可微；`R→0` 奇点、跨音速 τ* 多值处除外 | ●（远场） |
| Kirchhoff 积分 (10) | 线性边界积分 | 同 FWH | ● |
| Jacobi/不动点迭代到收敛 | x* = g(x*,θ) | **隐函数定理**：∂x*/∂θ = (I − J_g)⁻¹ ∂g/∂θ；有限步截断则对迭代次数分段可微 | ◐ |
| **稳态 RANS 收敛解** | 非线性方程组残差为零的点 | 离散伴随（adjoint）标准做法；**非 AD 磁带适用**（收敛判据、湍流切换点不光滑） | ◐（走伴随） |
| LES/瞬态对流 | 强非线性、多尺度 | 理论上可微（连续伴随 / 离散 AD + checkpointing）；数值耗散格式引入阶数依赖梯度 | ◐ |
| 接触/冲击（ penalty / 事件检测） | 法向间隙 g_n 的分段切换 | 接触状态翻转处梯度不连续 → 需平滑惩罚或平滑激活函数 | ○（需平滑化） |
| 网格重生/自适应 | 离散拓扑变化 | 拓扑变化处不可微 | ○ |
| 单元消亡/侵蚀（LS-DYNA erosion） | 材料删除事件 | 事件驱动，不可微 | ○ |

### 3.2 对照本项目的 AD 限制（quadrants/Taichi）

我们插件层实测过的限制，与上述判定一致：

1. **`atomic_add` / 非确定性归约**：FWH (9) 若用面单元并行累加到接收器时间bin，
   quadrants 反向不支持 → 与商业软件把 FWH 做成**后处理**（非梯度路径）的选择一致；
2. **内核内 in-place 读写**（`p[f+1] = f(p[f+1])`）：热耦合源项已绕过（前向注入
   宿主驱动，`TODO(MUSA/autodiff)`），等价于把源参数当作**常数输入**；
3. **循环内 `break` / 数据依赖早退**：Jacobi 残差早停（◐ 行）在 AD 磁带中必须
   固定迭代数——`joule_heating` 的 backward 固定回放 `max_iter` 次正是此法；
4. **宿主预计算常量断链**：`joule_heating` 的 `_r_field` 原为 host 端 from_numpy 常量，
   梯度到不了 ρ/cp/k；内联进内核后 AD 图接通（本项目已修复）。这对商业软件同样成立：
   **任何"前处理算好、求解器当常量读"的参数都没有梯度**。

### 3.3 结论

- **结构声学链路（振动→辐射）天然是 AD 友好区**：线性 FEM/BEM + 线性传播 +
  FFT 后处理，全链路可微（Mechanical 的声压对结构尺寸/材料、LS-DYNA BEM 对表面
  速度的梯度都有解析/半解析路径）；
- **气动声学链路的瓶颈在流场**：RANS 走伴随、LES 走 AD+checkpointing；FWH/ Kirchhoff
  传播段本身可微，但商业软件将其定位为后处理（无梯度接口）；
- 冲击类问题（接触、侵蚀）在所有平台上都是可微性的硬边界，需平滑化技术
  （penalty 光滑化、sigmoid 激活、事件时刻连续化）。

## 参考

- Ffowcs Williams, J. E. & Hawkings, D. L. (1969). *Sound generation by turbulence and
  surfaces in arbitrary motion*. Proc. R. Soc. A 264.
- Lighthill, M. J. (1952). *On sound generated aerodynamically*. Proc. R. Soc. A 211.
- Ewert, R. & Schröder, W. (2003). *Acoustic perturbation equations based on flow
  decomposition via source filtering*. J. Comput. Phys. 188.
- ANSYS Mechanical APDL *Acoustic Analysis Guide*（FLUID30/129/220/221、FSI、PML）。
- ANSYS Fluent *Theory Guide* §11（FWH、broadband、wave-equation acoustics）。
- LS-DYNA *Keyword User's Manual*：`*MAT_ACOUSTIC`、`*FREQUENCY_DOMAIN_ACOUSTIC_BEM/FEM`、
  `*FREQUENCY_DOMAIN_SSD`；Hang, Souli & Perez, *Simulation of acoustic and vibroacoustic
  problems in LS-DYNA using boundary element method*（9th European LS-DYNA Conference）。
- 本项目：`plugins/solvers/acoustics`（leapfrog 实现）、
  `scripts/quadrants_issue_thermal_source_autodiff.md`（AD 限制实录）。
