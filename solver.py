import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

def solve_ode(f, t_span, y0, method='RK45', t_eval=None):
    """你原来的函数，保留不动"""
    sol = solve_ivp(f, t_span, y0, method=method, t_eval=t_eval)
    return sol


# ========== 新增：文档第一个例题（双体问题 BVP）==========

def two_body_dynamics(t, state):
    """双体问题 ODE
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    r3 = r**3 if r**3 > 1e-10 else 1e-10
    return [vx, vy, -x / r3, -y / r3]


def solve_example_26_1(guess_vx0=-0.5, guess_vy0=0.5, method='RK45', eps=1e-8):
    """
    例题 26.1：双体问题边值问题（打靶法）
    已知：x(0)=2, y(0)=0, x(7)=1.0738644361, y(7)=-1.0995343576
    未知：vx(0), vy(0)
    """
    a1, a2 = 2.0, 0.0
    b1, b2 = 1.0738644361, -1.0995343576
    T = 7.0

    def residual(v0):
        vx0, vy0 = v0
        sol = solve_ivp(two_body_dynamics, [0, T], [a1, a2, vx0, vy0],
                        method=method, dense_output=True, rtol=eps, atol=eps/10)
        return [sol.y[0, -1] - b1, sol.y[1, -1] - b2]

    # 自动搜索初始速度
    result = root(residual, [guess_vx0, guess_vy0], method='hybr', tol=eps)
    if not result.success:
        raise RuntimeError(f"Shooting failed: {result.message}")

    vx0_opt, vy0_opt = result.x

    # 最终积分
    t_eval = np.linspace(0, T, 200)
    sol = solve_ivp(two_body_dynamics, [0, T], [a1, a2, vx0_opt, vy0_opt],
                    method=method, t_eval=t_eval, rtol=eps, atol=eps/10)

    info = {'vx0': vx0_opt, 'vy0': vy0_opt, 'iter': result.nfev}
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], info