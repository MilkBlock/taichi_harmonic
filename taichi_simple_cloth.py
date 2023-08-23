import taichi as ti
ti.init(debug=True) # Alternatively, ti.init(arch=ti.cpu)
N = 10   # 一条边一共N个节点

dampling =0.7

cloth_N = 13 
cloth_vertices = ti.Vector.field(3,ti.f32,cloth_N**2)
cloth_indices = ti.field(int,(cloth_N-1)**2*6)
cloth_cell_size = 5/cloth_N

cloth_v = ti.Vector.field(3,ti.f32,cloth_N*cloth_N)
gravity = 5
delta_time = 0.016

cell_size = 3/N
k = 500

c_vertices = ti.Vector.field(3,ti.f32,6*N*N)
c_indices = ti.field(int,6*(N-1)**2*6)
@ti.kernel
def generate_points():
    # 和 x y平面平行
    for i,j in ti.ndrange(N,N):
        c_vertices[0*(N)*(N)+i*(N)+j] = ti.Vector([i*cell_size,j*cell_size,0*cell_size])       
    for i,j in ti.ndrange(N,N):
        c_vertices[1*(N)*(N)+i*(N)+j] = ti.Vector([i*cell_size,j*cell_size,(N-1)*cell_size])    
    # 和x z 平面
    for i,j in ti.ndrange(N,N):
        c_vertices[2*(N)*(N)+i*(N)+j] = ti.Vector([i*cell_size,0*cell_size,j*cell_size])       
    for i,j in ti.ndrange(N,N):
        c_vertices[3*(N)*(N)+i*(N)+j] = ti.Vector([i*cell_size,(N-1)*cell_size,j*cell_size])       
    # y  z 
    for i,j in ti.ndrange(N,N):
        c_vertices[4*(N)*(N)+i*(N)+j] = ti.Vector([0*cell_size,i*cell_size,j*cell_size])       
    for i,j in ti.ndrange(N,N):
        c_vertices[5*(N)*(N)+i*(N)+j] = ti.Vector([(N-1)*cell_size,i*cell_size,j*cell_size])       
    
    for i,j,k in ti.ndrange(N,N,(0,6)):
        c_vertices[k*(N)*(N)+i*(N)+j] = c_vertices[k*(N)*(N)+i*(N)+j] + ti.Vector([2,2,2])

    for i,j in ti.ndrange(cloth_N,cloth_N):
        cloth_vertices[i*(cloth_N)+j] = ti.Vector([ i*cloth_cell_size, 0,j*cloth_cell_size])
        cloth_vertices[i*(cloth_N)+j]= cloth_vertices[i*(cloth_N)+j] + ti.Vector([3.5,8,1])
        cloth_v[i*cloth_N+ j] = ti.Vector([0,0,0])

    
directions = [[1,0],[0,1],[-1,0],[0,-1]]
@ti.kernel
def step():
    for i , j in ti.ndrange(cloth_N,cloth_N):
        for di,dj in ti.static(directions):
            ei = i + di 
            ej = j + dj 
            if i + di >=0  and i+di < cloth_N:
                if j + dj >=0  and j+dj < cloth_N:
                    x= cloth_vertices[ei*cloth_N+ej]  - cloth_vertices[i*cloth_N+ j]
                    f = k*(x.norm()-cloth_cell_size) * (x/x.norm())
                    cloth_v[i*cloth_N+ j] += f * delta_time 
    
    cloth_v[0]  =ti.Vector([0,0,0])
    cloth_v[cloth_N-1]  =ti.Vector([0,0,0])
    # 这是重力
    for i , j in ti.ndrange(cloth_N,cloth_N):
        cloth_v[i*cloth_N+ j] += ti.Vector([0,-gravity*delta_time,0])
    
    # 对速度进行迭代
    for i , j in ti.ndrange(cloth_N,cloth_N):
        cloth_v[i*cloth_N +j] *= ti.exp(-dampling* delta_time)
        cloth_vertices[i*cloth_N + j] += cloth_v[i*cloth_N +j]*delta_time
    
        

@ti.kernel
def generate_triangles():
    for n,i,j in ti.ndrange((0,6),N-1,N-1):
        square_id = (i * (N - 1)) + j
        c_indices[n*((N-1)**2*6)+square_id*6 +0] = n*(N)*(N)+i*(N)+j
        c_indices[n*((N-1)**2*6)+square_id*6 +1] = n*(N)*(N)+(i+1)*(N)+j
        c_indices[n*((N-1)**2*6)+square_id*6 +2] = n*(N)*(N)+i*(N)+j+1
        c_indices[n*((N-1)**2*6)+square_id*6+3] = n*(N)*(N)+(i+1)*(N)+j+1
        c_indices[n*((N-1)**2*6)+square_id*6+4] = n*(N)*(N)+i*(N)+j+1
        c_indices[n*((N-1)**2*6)+square_id*6+5] = n*(N)*(N)+(i+1)*(N)+j
    for i , j in ti.ndrange(cloth_N-1,cloth_N-1):
        square_id = i*(cloth_N-1) + j
        # 1st 斜边朝右下的三角形
        cloth_indices[square_id*6+0] = (i)*(cloth_N)+j
        cloth_indices[square_id*6+1] = (i+1)*(cloth_N)+j
        cloth_indices[square_id*6+2] = i*(cloth_N)+j+1
        # 2nd 斜边朝左上的三角形
        cloth_indices[square_id*6+3] = (i+1)*(cloth_N)+j+1
        cloth_indices[square_id*6+5] = i*(cloth_N)+j+1
        cloth_indices[square_id*6+4] = (i+1)*(cloth_N)+j
        

    # print((N-1)**2*6," 一个面中的三角形中的点")
    # for i,j,p in ti.ndrange(N-1,N-1,6):
    # for i in range(N-1):
    #     for j in range(N-1):
    #         for p in range(6):
    #             square_id = i*(N-1) + j
    #             print(c_indices[square_id*6 + p])
    #         print()



generate_points()
generate_triangles()
import sys
# sys.stdout.flush()
window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene:ti.ui.Scene = ti.ui.Scene()
camera = ti.ui.Camera()
# print(c_indices[60:66])

for i in range(6):
    print(c_indices[4*6+i])
    print(c_vertices[c_indices[4*6+i]])

n=0
while window.running :
    # n+=1
    n +=0
    scene.point_light(pos=(10,10, 10), color=(1, 1, 1))
    camera.position(8+n*0.003,8, 8)
    camera.lookat(0, 0, 0)
    step()
    # camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    scene.set_camera(camera)
    # scene.mesh(c_vertices, indices=c_indices,color=(0.5,0.5,0.5),index_count=54, two_sided = True,show_wireframe=False)
    # scene.mesh(c_vertices, indices=c_indices,color=(0.5,0.5,0.5), two_sided = True,show_wireframe=True)
    # scene.mesh(c_vertices, indices=c_indices,index_count=6,index_offset=0*6,color=(1,1,1), two_sided = True,show_wireframe=False)
    # scene.mesh(c_vertices, indices=c_indices,index_count=6,index_offset=1*6,color=(0.5,1,1), two_sided = True,show_wireframe=False)
    # scene.mesh(c_vertices, indices=c_indices,index_count=6,index_offset=2*6,color=(1,0.5,1), two_sided = True,show_wireframe=False)
    # scene.mesh(c_vertices, indices=c_indices,index_count=6,index_offset=3*6,color=(1,1,0.5), two_sided = True,show_wireframe=False)
    # scene.mesh(c_vertices, indices=c_indices,index_count=6,index_offset=4*6,color=(1,1,0.5), two_sided = True,show_wireframe=False)
    # scene.particles(c_vertices,0.05)
    scene.particles(cloth_vertices,0.05)
    # scene.mesh(c_vertices, indices=c_indices,color=(1,1,1), two_sided = True)
    scene.mesh(cloth_vertices, indices=cloth_indices,color=(1,1,1),  two_sided = True)
    # scene.mesh(vertices, indices=indices, per_vertex_color=v_color, two_sided = True)
    # scene.mesh(vertices, indices=indices, color=(1,1,1), two_sided = True)
    # scene.particles(block_center, radius=block_radius, color=(0.5, 0, 0))  # ? particles 居然不需要自己手动模拟
    
    canvas.scene(scene)
    window.show()