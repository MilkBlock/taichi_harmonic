import taichi as ti
ti.init()
N = 30
@ti.kernel
def p():
    for i,j in ti.ndrange((0,5),(0,3)):
        print(i,j)
@ti.kernel
def generate_triangles():
    for k,i,j in ti.ndrange((0,6),N,N):
        print(k,i,j)
        print(i*(N+1)+j+1+k*(N+1)*(N+1)  ,  
              (i+1)*(N+1)+j+k*(N+1)*(N+1),
              (i+1)*(N+1)+j+1+k*(N+1)*(N+1))
    print("tir init ended")
generate_triangles()