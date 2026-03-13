"""
MPC (Model Predictive Control) Controller

来源: openloong-dyn-control
核心算法: 基于简化单刚体模型的模型预测控制

状态: [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz] (12维)
输入: [FL_x, FL_y, FL_z, τL_x, τL_y, τL_z, FR_x, FR_y, FR_z, τR_x, τR_y, τR_z] (12维)
"""

import numpy as np
from scipy.linalg import expm
from typing import Optional, Dict


class MPCController:
    """
    MPC 模型预测控制器
    
    用于人形机器人质心轨迹跟踪，输出期望接触力。
    """
    
    def __init__(
        self,
        dt: float = 0.005,
        horizon: int = 10,
        control_horizon: int = 3,
        mass: float = 77.35,
        inertia: Optional[np.ndarray] = None
    ):
        """
        Args:
            dt: 控制周期 (秒)
            horizon: 预测步数 N
            control_horizon: 控制时域
            mass: 机器人质量 (kg)
            inertia: 3x3 惯性张量 (kg·m²)
        """
        self.dt = dt
        self.N = horizon
        self.ch = control_horizon
        
        # 系统维度
        self.nx = 12  # 状态维度
        self.nu = 13  # 输入维度 (左6 + 右6 + 1)
        
        # 机器人参数
        self.m = mass
        self.g = -9.81
        
        # 惯性矩阵
        if inertia is None:
            self.Ic = np.array([
                [12.61, 0, 0.37],
                [0, 11.15, 0.01],
                [0.37, 0.01, 2.15]
            ])
        else:
            self.Ic = inertia
        self.Ic_inv = np.linalg.inv(self.Ic)
        
        # 输入限制
        self.max_force = np.array([1000.0, 1000.0, -3.0 * self.m * self.g, 20.0, 80.0, 100.0])
        self.min_force = np.array([-1000.0, -1000.0, 0.0, -20.0, -80.0, -100.0])
        
        # 权重参数
        self.alpha = 1e-6
        self.L_diag = np.array([1.0, 1.0, 1.0,      # 欧拉角
                                1.0, 200.0, 1.0,     # 位置 (高度权重高)
                                1e-7, 1e-7, 1e-7,    # 角速度
                                100.0, 10.0, 1.0])   # 线速度
        self.K_diag = np.array([1.0] * 13)
        
        # 初始化
        self._build_weight_matrices()
        self._build_system_matrices()
        
        # 状态
        self.X_cur = np.zeros(self.nx)
        self.Xd = np.zeros(self.nx * self.N)
        self.Ufe = np.zeros(self.nu * self.ch)
        self.Fr_ff = np.zeros(12)
        self.pf2com = np.zeros(6)
        
        self.EN = False  # 启用标志
        self.debug = False
    
    def _build_weight_matrices(self):
        """构建权重矩阵"""
        self.L = np.zeros((self.nx * self.N, self.nx * self.N))
        for i in range(self.N):
            diag_values = np.array(self.L_diag)
            if len(diag_values) != self.nx:
                diag_values = np.ones(self.nx)
            self.L[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = np.diag(diag_values)
        
        self.K = np.zeros((self.nu * self.ch, self.nu * self.ch))
        for i in range(self.ch):
            diag_values = np.array(self.K_diag)
            if len(diag_values) != self.nu:
                diag_values = np.ones(self.nu) * 0.01
            self.K[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = np.diag(diag_values)
    
    def _build_system_matrices(self):
        """构建线性化系统矩阵 A 和 B"""
        Ac = np.zeros((self.nx, self.nx))
        Bc = np.zeros((self.nx, self.nu))
        
        Ac[0:3, 6:9] = np.eye(3)    # 欧拉角变化率
        Ac[3:6, 9:12] = np.eye(3)   # 位置变化率
        
        Bc[9:12, 0:3] = np.eye(3) / self.m   # 力影响线速度
        Bc[9:12, 6:9] = np.eye(3) / self.m
        
        # 离散化
        M = np.zeros((self.nx + self.nu, self.nx + self.nu))
        M[:self.nx, :self.nx] = Ac
        M[:self.nx, self.nx:] = Bc
        Md = expm(M * self.dt)
        
        self.A = Md[:self.nx, :self.nx]
        self.B = Md[:self.nx, self.nx:]
    
    def set_weight(self, u_weight: float, L_diag: np.ndarray, K_diag: np.ndarray):
        """设置 MPC 权重"""
        self.alpha = u_weight
        
        L_diag = np.array(L_diag)
        if len(L_diag) == self.nx:
            self.L_diag = L_diag.copy()
        
        K_diag = np.array(K_diag)
        if len(K_diag) == self.nu:
            self.K_diag = K_diag.copy()
        
        self._build_weight_matrices()
    
    def set_state(self, base_pos: np.ndarray, base_rpy: np.ndarray,
                  base_lin_vel: np.ndarray, base_ang_vel: np.ndarray):
        """设置当前状态"""
        self.X_cur[:3] = base_rpy
        self.X_cur[3:6] = base_pos
        self.X_cur[6:9] = base_ang_vel
        self.X_cur[9:12] = base_lin_vel
    
    def set_foot_positions(self, left_foot_pos: np.ndarray, right_foot_pos: np.ndarray,
                           com_pos: np.ndarray):
        """设置足端相对于质心的位置"""
        self.pf2com[:3] = left_foot_pos - com_pos
        self.pf2com[3:] = right_foot_pos - com_pos
    
    def set_desired_trajectory(self, Xd: np.ndarray):
        """设置期望轨迹"""
        if len(Xd) == self.nx * self.N:
            self.Xd = Xd.copy()
        else:
            # 调整长度
            Xd_new = np.zeros(self.nx * self.N)
            min_len = min(len(Xd), len(Xd_new))
            Xd_new[:min_len] = Xd[:min_len]
            self.Xd = Xd_new
    
    def compute(self) -> np.ndarray:
        """
        计算 MPC
        
        Returns:
            12维接触力/力矩 [FL(6), FR(6)]
        """
        if not self.EN:
            # 默认支撑力
            self.Fr_ff[2] = -self.m * self.g / 2
            self.Fr_ff[8] = -self.m * self.g / 2
            return self.Fr_ff.copy()
        
        try:
            # 更新时变动力学矩阵
            self._update_dynamics()
            
            # 构建预测模型
            Phi, Theta = self._build_prediction_model()
            
            # 求解 QP
            U_opt = self._solve_qp(Phi, Theta)
            
            # 提取接触力
            self.Fr_ff[:6] = U_opt[:6]
            self.Fr_ff[6:12] = U_opt[6:12]
            
        except Exception as e:
            if self.debug:
                print(f"MPC error: {e}")
            self._use_default_forces()
        
        return self.Fr_ff.copy()
    
    def _update_dynamics(self):
        """更新时变动力学矩阵"""
        r_l = self.pf2com[:3]
        r_r = self.pf2com[3:]
        
        # 叉积矩阵
        B_omega_l = self.Ic_inv @ np.array([
            [0, -r_l[2], r_l[1]],
            [r_l[2], 0, -r_l[0]],
            [-r_l[1], r_l[0], 0]
        ])
        
        B_omega_r = self.Ic_inv @ np.array([
            [0, -r_r[2], r_r[1]],
            [r_r[2], 0, -r_r[0]],
            [-r_r[1], r_r[0], 0]
        ])
        
        self.B[6:9, 0:3] = B_omega_l
        self.B[6:9, 6:9] = B_omega_r
    
    def _build_prediction_model(self):
        """构建预测模型矩阵"""
        Phi = np.zeros((self.nx * self.N, self.nx))
        Theta = np.zeros((self.nx * self.N, self.nu * self.ch))
        
        A_power = np.eye(self.nx)
        for i in range(self.N):
            Phi[i*self.nx:(i+1)*self.nx, :] = A_power @ self.A
            A_power = A_power @ self.A
            
            for j in range(min(i+1, self.ch)):
                A_power_j = np.linalg.matrix_power(self.A, i-j)
                Theta[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = A_power_j @ self.B
        
        return Phi, Theta
    
    def _solve_qp(self, Phi: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        """求解 QP 问题"""
        H_temp = Theta.T @ self.L @ Theta
        
        # 确保维度匹配
        if H_temp.shape != self.K.shape:
            min_dim = min(H_temp.shape[0], self.K.shape[0])
            H_temp = H_temp[:min_dim, :min_dim]
            K_adj = self.K[:min_dim, :min_dim]
        else:
            K_adj = self.K
        
        H = 2 * (H_temp + self.alpha * K_adj)
        
        error_term = Phi @ self.X_cur - self.Xd
        g = 2 * Theta.T @ self.L @ error_term
        
        if len(g) != H.shape[0]:
            g = g[:H.shape[0]]
        
        U_min = np.tile(self.min_force, self.ch)[:H.shape[0]]
        U_max = np.tile(self.max_force, self.ch)[:H.shape[0]]
        
        # 解析解
        H_reg = H + 0.001 * np.eye(H.shape[0])
        U_opt = np.linalg.solve(H_reg, -g)
        U_opt = np.clip(U_opt, U_min, U_max)
        
        return U_opt
    
    def _use_default_forces(self):
        """使用默认力"""
        self.Ufe[:6] = np.array([0, 0, -self.m * self.g / 2, 0, 0, 0])
        self.Ufe[6:12] = np.array([0, 0, -self.m * self.g / 2, 0, 0, 0])
        self.Fr_ff[:6] = self.Ufe[:6]
        self.Fr_ff[6:12] = self.Ufe[6:12]
    
    def enable(self):
        """启用 MPC"""
        self.EN = True
    
    def disable(self):
        """禁用 MPC"""
        self.EN = False
