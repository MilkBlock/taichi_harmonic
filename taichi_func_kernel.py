# from PIL import Image
# import taichi as ti
# Image.open()
# ti.init(debug=True,arch=ti.gpu)
# N = 4
# dt = 0.0002
# gravity = 2
# x=  ti.Vector.field(3,float,shape=(N,N)) #coordinate 
# v=  ti.Vector.field(3,float,shape=(N,N)) # velocity 
# indices = ti.field(int,)
# ball_radius = 0.2
# ball_center = ti.Vector.field(3, float,shape = (1))
# stiff = 2
# gravity = 0.5
# dampling = 0.5

# cell_size = 3
# def init_scene():
#     for i, j in ti.ndrange(N,N):
#         x[i,j] = ti.Vector([[i*cell_size],
#                             [j*cell_size/ti.sqrt(2)],
#                             [(N-j)*cell_size/ti.sqrt(2)]])

# links = [[i,j] for i,j in ti.ndrange((-1,2),(-1,2))]
# links.remove([0,0])
# links = [ti.Vector(v) for v in links ]

# @ti.kernel
# def step():
#     for i in ti.grouped(v):  # grouped function can't be called in pythonscope 
#         v[i].y -= gravity*dt 
#         # ti.grouped()
# # ti.Vector.diag
# # 
# @ti.kernel
# def step():
#     for i in ti.grouped(v):
#         v[i].y -= gravity*dt
#     for i in ti.grouped(x):   # what's the type of 
#         for d in ti.static(links):
#             force:ti.Vector = ti.Vector([0.0,0.0,0.0])
#             a=min(max(i+d,0),[N-1,N-1])  # 推断max函数是在编译的时候进行了某些修改
#             force = x[a] - x[i]
#             relative_pos = force.norm()
#             original_pos = d.norm()
#             force =  stiff * force.normalized() *(relative_pos - original_pos) / original_pos
#             v[i] += force*dt
#     for i in ti.grouped(x):   # dampling 
#         v[i] = v[i] * ti.exp( -dampling * dt)
#         x[i] += v[i]*dt

# window = ti.ui.Window("cloth",vsync=True,show_window=True,pos=(800,800))
# camera = ti.ui.Camera()
# canvas = window.get_canvas()
# step()


# def init_indices():
    


# # step()