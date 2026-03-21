import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# 1. 构建三维几何体：立方体有 8 个顶点
# 题目要求边长为 2，中心在原点，即坐标在 [-1, 1] 之间
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 定义立方体的 12 条边，每条边由两个顶点的索引组成
# 例如 (0, 1) 表示连接 vertices[0] 和 vertices[1]
indices = ti.field(ti.i32, shape=24) 

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    模型变换矩阵：为了更好的 3D 效果，这里改为绕 Y 轴旋转
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c,   0.0,  s,   0.0],
        [0.0, 1.0,  0.0, 0.0],
        [-s,  0.0,  c,   0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho = ti.Matrix([
        [2.0/(r-l), 0.0, 0.0, -(r+l)/(r-l)],
        [0.0, 2.0/(t-b), 0.0, -(t+b)/(t-b)],
        [0.0, 0.0, 2.0/(n-f), -(n+f)/(n-f)],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    # 将相机放远一点 (z=5)，以便观察立方体
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(8):
        v4 = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def init_cube():
    # 定义 8 个顶点 (x, y, z)
    v_data = [
        [-1, -1,  1], [1, -1,  1], [1,  1,  1], [-1,  1,  1], # 前面 4 个点
        [-1, -1, -1], [1, -1, -1], [1,  1, -1], [-1,  1, -1]  # 后面 4 个点
    ]
    for i in range(8):
        vertices[i] = v_data[i]

    # 定义 12 条边的索引对 (共 24 个数值)
    e_data = [
        0, 1, 1, 2, 2, 3, 3, 0, # 前面四条边
        4, 5, 5, 6, 6, 7, 7, 4, # 后面四条边
        0, 4, 1, 5, 2, 6, 3, 7  # 前后连接的四条边
    ]
    for i in range(24):
        indices[i] = e_data[i]

def main():
    init_cube()
    gui = ti.GUI("3D Rotating Cube", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE: gui.running = False
        
        # 每一帧自动旋转，增强视觉效果
        angle += 1.0
        compute_transform(angle)
        
        # 2. 修改渲染逻辑：遍历 12 条边进行绘制
        # 索引数组中每两个值代表一条线的起点和终点
        coords_np = screen_coords.to_numpy()
        indices_np = indices.to_numpy()
        
        for i in range(12):
            idx1 = indices_np[i * 2]
            idx2 = indices_np[i * 2 + 1]
            gui.line(coords_np[idx1], coords_np[idx2], radius=2, color=0x00FF00)
        
        gui.show()

if __name__ == '__main__':
    main()