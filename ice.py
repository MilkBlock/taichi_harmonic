import taichi as ti
#  0.75拉格朗日 0.25欧拉法
# 就是说我们在记录粒子的同时，在网格中记录一些辅助信息

# 欧拉法
# 首先把空间化成方格，然后遍历画布，对于画布中每个像素点的坐标，通过方格上的速度向量回溯到上一帧
# 这个像素所在的位置
# 然后只需要复制上个位置的像素颜色就行了

# 计算散度特别容易
# projection 也就是根据某一点找到它附近有数据的邻居    

# 拉格朗日法 
# for every particle ,store and calculate its speed per frame and predict the position in next frame
# then draw every particle in every frame
# 可以很好的满足动量守恒，还有能量守恒，碰撞计算等等

ti.init(arch=ti.gpu,default_fp=ti.f64)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations    resolution 
n_particles, n_grid = 6000 * quality**2, 200 * quality   # 
dx, inv_dx = 1 / n_grid, float(n_grid)  # 是单个cell 的大小
dt = 1e-4 / quality   # 两帧时间间距
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho   #  计算质量   of particle
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio  杨氏模量  
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
# lame & naive 

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient 形变梯度
# transformer   deformation deformation transform
material = ti.field(dtype=int, shape=n_particles)  # material id  
#  triangle_indices 是类似的东西 
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation   
# elastic deformation    
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass     质量
gravity = ti.Vector.field(2, dtype=float, shape=())    # 二维向量  代表重力方向
attractor_strength = ti.field(dtype=float, shape=())   # 引力强度
attractor_pos = ti.Vector.field(2, dtype=float, shape=())   # 引力的中心

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]  # 网格上记录的速度
        grid_m[i, j] = 0   # 网格上记录的质量
    for p in x:  # Particle state update and scatter to grid (P2G)   Particle to grid 
        if(not 0<x[p][0]<1 or not 0<x[p][1]<1 ):
            print(x[p])
        base = (x[p] * inv_dx - 0.5).cast(int)   # 计算它的基   这里是在网格的中点记录数据
        if x[p][0]>1.0 or x[p][0]<0.0:
            print(x[p])
        fx = x[p] * inv_dx - base.cast(float)  # 到网格节点的距离
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]   
        # deformation gradient update 形变梯度的更新
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))  # Clamp 函数 max( xxx, min())
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    group_size = n_particles // 3
    for i in range(n_particles):
        x[i] = [    # i = 9000     # group_size = 3000
            ti.random() * 0.3 + 0.3 + 0.10 * (i // group_size),     # 液体以(0.3,0.05)为中心
            ti.random() * 0.3 + 0.05 + 0.32 * (i // group_size),   
        ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])  # 形变梯度  此时为 无
        Jp[i] = 1     # 塑性形变
        C[i] = ti.Matrix.zero(float, 2, 2)  # 仿射速度  这只是一个常量


print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset.")
gui = ti.GUI("Taichi MLS-MPM-128", res=1024, background_color=0x112F41)
reset()
gravity[None] = [0, -1]

for frame in range(20000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == "r":
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.event is not None:
        gravity[None] = [0, 0]  # if had any event
    if gui.is_pressed(ti.GUI.LEFT, "a"):
        gravity[None][0] = -1
    if gui.is_pressed(ti.GUI.RIGHT, "d"):
        gravity[None][0] = 1
    if gui.is_pressed(ti.GUI.UP, "w"):
        gravity[None][1] = 1
    if gui.is_pressed(ti.GUI.DOWN, "s"):
        gravity[None][1] = -1
    mouse = gui.get_cursor_pos()
    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
    attractor_pos[None] = [mouse[0], mouse[1]]
    attractor_strength[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        attractor_strength[None] = 1
    if gui.is_pressed(ti.GUI.RMB):  # right mouse button
        attractor_strength[None] = -1  # 反向的斥力
    for s in range(int(2e-3 // dt)):   #    连续计算其中的物理帧后 再显示一帧
        substep()
    gui.circles(
        x.to_numpy(),
        radius=1.5,
        palette=[0x068587, 0xED553B, 0xEEEEF0],
        palette_indices=material,
    )

    # Change to gui.show(f'{frame:06d}.png') to write images to disk
    gui.show()
