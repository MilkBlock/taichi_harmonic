import taichi as ti
ti.init(debug=True)
k = 4
N = 1024
p_list = [[0.1,0.1],[0.2,0.5],[0.3,0.7],[0.4,0.5],[0.5,0.1],[0,0]]
n = len(p_list)
dx = (n+1-k+1)/N 
# particles = ti.Vector.field(2,float,N)
particles_list = []
def B_spline():
	"""
	:param p_list: (list of list of int:[[x0, y0], [x1, y1], ...])point set of p
	result: (list of list of int:[[x0, y0], [x1, y1], ...])point on curve
	绘制三次(四阶)均匀B样条曲线
	"""
	result = []
	u = k-1
	while (u < n+1):
		x, y = 0, 0
		#calc P(u)
		for i in range(0, n):
			B_ik = deBoor_Cox(u, k, i)
			x += B_ik * p_list[i][0]  # for every points in the list 
			y += B_ik * p_list[i][1]
		# result.append((int(x+0.5), int(y+0.5)))
		result.append([x,y])
		u += dx
	return result

def deBoor_Cox(u, k, i):
	if k==1:
		if i <= u and u <= i+1:
			return 1
		else:
			return 0
	else:
		coef_1, coef_2 = 0, 0
		if (u-i == 0) and (i+k-1-i == 0):
			coef_1 = 0
		else:
			coef_1 = (u-i) / (i+k-1-i)
		if (i+k-u == 0) and (i+k-i-1 == 0):
			coef_2 = 0
		else:
			coef_2 = (i+k-u) / (i+k-i-1)
	return coef_1 * deBoor_Cox(u, k-1, i) + coef_2 * deBoor_Cox(u, k-1, i+1)
gui = ti.GUI("B-spline", (512, 512), background_color=0xFFFFFF)
import numpy as np
particles_list = B_spline()
def draw():
    # for i,v in enumerate(l):
    pass
        # print(v)
        # particles[i] = np.array(v)
        # print(particles[i])
    
for T in range(1000000):
    draw()

    gui.circles(   
        np.array(p_list),
        radius=8, 
        color=0xff
        )
    gui.circles(   
        np.array(particles_list),
        radius=2, 
        color=0xff
        )
    # 这里在画固定半径的圆
    gui.show()
    mouse = gui.get_cursor_pos()


# 如果只想要使用taichi的绘图功能的话那么一般是不需要使用taichi的gpu计算使用字段
# 直接使用numpy 更加划算一点
# 注意二维渲染gui的坐标系是以左下角为(0,0)，并且整个窗体的右上角被视作 (1,1)