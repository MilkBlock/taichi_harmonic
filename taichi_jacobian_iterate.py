import taichi as ti
n = 10
A = ti.var(dt=ti.f32,shape=(n,n))
x = ti.var(dt=ti.f32,shape = n)
new_x = ti.var(dt=ti.f32,shape= n)
b = ti.var(dt=ti.f32,shape= n)

@ti.kernel
def iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if j!=i:
                r-= A[i,j]*x[j]
        new_x = r/A[i,i]

ti.GUI()
