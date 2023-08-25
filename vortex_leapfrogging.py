import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

eps = 0.01  #   
dt = 0.1  # 两帧之间的时间差

n_vortex = 4   # 涡流的数量
n_tracer = 200000  # 

pos = ti.Vector.field(2, ti.f32, shape=n_vortex)   # 这是一个二维向量数组
new_pos = ti.Vector.field(2, ti.f32, shape=n_vortex)# 这是一个二维向量数组
vort = ti.field(ti.f32, shape=n_vortex) # 涡流的旋转速度
tracer = ti.Vector.field(2, ti.f32, shape=n_tracer)  # 每一个元素由两个float组成 


@ti.func
def compute_u_single(p, i):
    r2 = (p - pos[i]).norm_sqr()
    uv = ti.Vector([pos[i].y - p.y, p.x - pos[i].x])  # 这个vector 和 p-pos[i] 垂直,这是切线方向
    return vort[i] * uv / (r2 * math.pi) * 0.5 #* (1.0 - ti.exp(-r2 / eps**2))
# uv 的长度就是 r  ，


@ti.func
def compute_u_full(p):
    u = ti.Vector([0.0, 0.0])
    for i in range(n_vortex):
        u += compute_u_single(p, i)
    return u


@ti.kernel
def integrate_vortex():
    for i in range(n_vortex):
        v = ti.Vector([0.0, 0.0])
        for j in range(n_vortex):
            if i != j:
                v += compute_u_single(pos[i], j)
        new_pos[i] = pos[i] + dt * v

    for i in range(n_vortex):
        pos[i] = new_pos[i]


@ti.kernel  # 通过taichi 进行gpu 加速
def advect():
    for i in range(n_tracer):   # tracer的遍历
        # Ralston's third-order method
        p = tracer[i]
        v1 = compute_u_full(p)
        v2 = compute_u_full(p + v1 * dt * 0.5)
        v3 = compute_u_full(p + v2 * dt * 0.75)
        tracer[i] += (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3) * dt


pos[0] = [0, 1]
pos[1] = [0, -1]
pos[2] = [0, 0.3]
pos[3] = [0, -0.3]
vort[0] = 1
vort[1] = -1
vort[2] = 1
vort[3] = -1


@ti.kernel
def init_tracers():
    for i in range(n_tracer):
        tracer[i] = [ti.random() - 0.5, ti.random() * 3 - 1.5]   # 以 (0,0)为半径为1中心的分布


init_tracers()

gui = ti.GUI("Vortex", (1024, 512), background_color=0xFFFFFF)

for T in range(1000):
    for i in range(4):  # substeps
        advect()     # advect 就是简单的平流输送
        integrate_vortex()  # 集成 涡流

    gui.circles(   
        tracer.to_numpy() * np.array([[0.05, 0.1]]) + np.array([[0.0, 0.5]]),
        radius=0.5, 
        color=0x0)
    # 这里在画固定半径的圆
    gui.show()
