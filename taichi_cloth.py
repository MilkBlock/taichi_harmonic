import taichi as ti
ti.init(default_fp=ti.f32,arch=ti.cuda) # Alternatively, ti.init(arch=ti.cpu)
N = 70  # 如果是N=500这里一共是 25w个弹簧节点， 大概耗时10s编译
cube_N = 50
cell_size = 1./ N
gravity = 1.5
stiffness = 1200
damping = 2 # 这个模型并没有考虑非相邻块布料与布料之间的力所以可能会出现相连的现象
dt = 5e-4

block_radius = 0.22
block_center = ti.Vector.field(3, ti.f32, (1,))

x = ti.Vector.field(3, ti.f32, (N, N))
v = ti.Vector.field(3, ti.f32, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, ti.f32, N * N)
cube_head_vertices = ti.Vector.field(3, ti.f32, 8)
cube_vertices = ti.Vector.field(3, ti.f32, 6*cube_N*cube_N)
cube_indices = ti.field(int, 6*(cube_N-1)**2*2*3)

def init_scene():
    print("init started")
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([i * cell_size ,
                             j * cell_size / ti.sqrt(2),
                             (N - j) * cell_size / ti.sqrt(2)])
        x[i,j] += ti.Vector([0,1,0])
        # x[i,j] = ti.Vector([i*cell_size,  0., j*cell_size])
    block_center[0] = ti.Vector([0.5, -0.5, -0.0])
    for k,v in ti.static(enumerate([[1,1,1],[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1],[1,-1,1]])):
        cube_head_vertices[k] = ti.Vector([block_center[0][i]+v[i]*block_radius for i in range(3)])
        # print([block_center[0][i]+v[i]*block_radius for i in range(3)])
    import itertools as it
    for n,c in ti.static(enumerate(it.combinations([cube_head_vertices[i] for i in range(0,8,2)],r=2))):
        # print(c)
        # print("n : ",n)
        v1,v2 = c[0],c[1]
        dv = (v2 - v1)/(cube_N-1)
        if dv[0] == 0:
            # print("0")
            for i,j in ti.ndrange(cube_N,cube_N):
                # print(i,j)
                cube_vertices[n*cube_N**2+i*cube_N+j] = ti.Vector([v1[r]+(0,i,j)[r]*dv[r] for r in range(3)])
        if dv[1] == 0:
            # print("1")
            for i,j in ti.ndrange(cube_N,cube_N):
                cube_vertices[n*cube_N**2+i*cube_N+j] = ti.Vector([v1[r]+(i,0,j)[r]*dv[r] for r in range(3)])
        if dv[2] == 0:
            # print("2")
            for i,j in ti.ndrange(cube_N,cube_N):
                cube_vertices[n*cube_N**2+i*cube_N+j] = ti.Vector([v1[r]+(i,j,0)[r]*dv[r] for r in range(3)])
    print("init ended")
            

@ti.kernel
def set_indices():
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j
    for n in ti.static(range(6)):
        for i, j in ti.ndrange(cube_N, cube_N):
            if i < cube_N - 1 and j < cube_N - 1:
                square_id = (i * (cube_N - 1)) + j
                # 1st triangle of the square
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 0)] = n*(cube_N)**2 +i * cube_N + j
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 1)] = n*(cube_N)**2 +(i + 1) * cube_N + j
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 2)] = n*(cube_N)**2 +i * cube_N + (j + 1)
                # 2nd triangln*be_N*(c-1ube_N)**2the square
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 3)] = n*(cube_N)**2 +(i + 1) * cube_N + j + 1
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 4)] = n*(cube_N)**2 +i * cube_N + (j + 1)
                cube_indices[n*(cube_N-1)**2*6+(square_id * 6 + 5)] = n*(cube_N)**2 +(i + 1) * cube_N + j

links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = [ti.Vector(v) for v in links]

@ti.kernel
def step():
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
    for i in ti.grouped(x):
        force = ti.Vector([0.0,0.0,0.0])
        for d in ti.static(links):
            j = min(max(i + d, 0), [N-1,N-1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i-j).norm()
            if original_length != 0:
                force +=  stiffness * relative_pos.normalized() * (current_length - original_length) / original_length
        v[i] +=  force * dt
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)  #这个是v速度的衰减量 也就是我们的ti.exp(-damping*dt )
        if (x[i]-block_center[0]).norm() <= block_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])
        if ti.abs(x[i].x-block_center[0].x)<=block_radius+0.03 and ti.abs(x[i].y-block_center[0].y)<=block_radius+0.03 and ti.abs(x[i].z-block_center[0].z)<=block_radius+0.03:
            v[i] = ti.Vector([0.0, 0.0, 0.0])
        x[i] += dt * v[i]   # explicit  version   

@ti.kernel
def set_vertices():  # 由于我们这里使用了多线程(窗体与计算是两个)所以必须要在开启绘制线程之前拷贝一份 顶点
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]

# set images
from PIL import Image as IMG
from numpy import asarray
import numpy as np

np.ndarray([2,2,2,3])
img:IMG.Image = IMG.open("./rubato.png")
img = img.resize((N,N))
# img.thumbnail((N,N),resample=IMG.Resampling.LANCZOS)
img.save("./rubato_tmp.png")
if img.mode!="RGB":
    print("not rgb transform")
    img.convert(mode="RGB")
v_color:ti.MatrixField= ti.Vector.field(3,ti.f32,shape=(N*N))
color = asarray(img,dtype=np.float32)
assert color.shape==(N,N,3)
# color = np.transpose(color,(1,0,2))
color = np.rot90(color,3,axes=(0,1))
color = color.reshape((-1,3))/255.0
assert color.shape==(N*N,3)
v_color.from_numpy(color)
n_color = v_color.to_numpy()
# v_color.from_numpy(np.ones((N*N,3),dtype=ti.f32)/2)
img:IMG.Image = IMG.open("./flag.png")
img.resize( (cube_N,cube_N) )
cube_color:np.ndarray= asarray(img,dtype=np.float32)
cube_color = cube_color.reshape((-1,3))/255.
cube_color = np.vstack([cube_color for i in range(6)])
print(cube_color.shape)
# assert cube_color.shape == (cube_N*cube_N*6,3) ," cube_color not suit "+str(cube_color.shape)
c_color:ti.MatrixField = ti.Vector.field(3,float,shape=(cube_N*cube_N*6))
c_color.from_numpy(cube_color)

print("images loaded ")
init_scene()
set_indices()
window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene:ti.ui.Scene = ti.ui.Scene()
camera = ti.ui.Camera()

n = 0 
import sys
while window.running :
    # n+=1
    sys.stdout.flush()
    # print(n)
    for i in range(30):
        step()
    set_vertices()
    camera.position(1.5, -1, 2.2)
    camera.lookat(0.5, -0.5, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.mesh(cube_vertices, indices=cube_indices, per_vertex_color=c_color, two_sided = True)
    scene.mesh(vertices, indices=indices, per_vertex_color=v_color, two_sided = True)
    # scene.mesh(vertices, indices=indices, color=(1,1,1), two_sided = True)
    # scene.particles(block_center, radius=block_radius, color=(0.5, 0, 0))  # ? particles 居然不需要自己手动模拟
    canvas.scene(scene)
    window.show()