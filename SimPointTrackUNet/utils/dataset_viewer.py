import tkinter as tk
import numpy as np

# ==============================================================================
# --- 用户可配置参数 ---
# ==============================================================================

# 1. 数据文件路径
NPZ_FILE_PATH = r'D:\Admin-Ender\net_learn\data\lslidar_points_A\test\22.npz'  # 替换为您的 .npz 文件路径

# 2. 画布尺寸
CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 700

# 3. 标签颜色映射
#    键是标签(例如 0, 1, ...)，值是颜色名称
LABEL_COLORS = {
    0: 'blue',
    1: 'green',
    2: 'red',
    # 您可以根据需要添加更多标签和颜色
}
DEFAULT_COLOR = 'gray'  # 如果标签在上面字典中找不到，则使用此默认颜色

# 4. 可视化参数
POINT_RADIUS = 5  # 数据点的半径（像素）
SCALE_FACTOR = 120 # 比例尺：数据中的 1.0 单位对应多少像素
AXIS_COLOR = 'white' # 坐标轴颜色
GRID_COLOR = '#444'  # 网格线颜色

# ==============================================================================
# --- 主程序 ---
# ==============================================================================

class DataVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("数据集可视化")

        # 创建画布
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 加载数据
        try:
            self.targets, self.labels = self.load_data(NPZ_FILE_PATH)
        except Exception as e:
            self.canvas.create_text(
                CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2,
                text=f"错误: {e}", fill="red", font=("Arial", 12)
            )
            return
        
        # 绘制所有内容
        self.draw_all()

    def load_data(self, file_path):
        """从.npz文件中加载数据"""
        with np.load(file_path) as data:
            if 'targets' not in data or 'labels' not in data:
                raise FileNotFoundError("文件中未找到 'targets' 或 'labels' 数组")
            targets = data['targets']
            labels = data['labels']
            print(f"成功加载数据: {len(targets)} 个点")
            return targets, labels

    def transform_coords(self, data_x, data_y):
        """
        将数据坐标转换为画布坐标。
        数据坐标系: 标准笛卡尔坐标系 (x向右, y向上)
        要求的屏幕坐标系: X正方向朝上, Y正方向朝左
        画布坐标系 (Tkinter): 原点在左上角, x向右, y向下
        """
        # 数据原点 (0,0) 映射到画布中心
        center_x = CANVAS_WIDTH / 2
        center_y = CANVAS_HEIGHT / 2

        # 转换逻辑：
        # - 我们的X轴正向是屏幕上方，对应画布的 Y 轴负方向。
        # - 我们的Y轴正向是屏幕左方，对应画布的 X 轴负方向。
        # 所以，画布坐标 (canvas_x, canvas_y) 的计算如下：
        canvas_x = center_x - data_y * SCALE_FACTOR
        canvas_y = center_y - data_x * SCALE_FACTOR
        
        return canvas_x, canvas_y

    def draw_axes_and_grid(self):
        """绘制坐标轴、网格和标签"""
        center_x = CANVAS_WIDTH / 2
        center_y = CANVAS_HEIGHT / 2

        # 绘制网格线 (每隔一个单位)
        max_dist = max(CANVAS_WIDTH, CANVAS_HEIGHT) / (2 * SCALE_FACTOR)
        step = 1.0
        for i in np.arange(-max_dist, max_dist, step):
            # 垂直于X轴的网格线
            x1, y1 = self.transform_coords(i, -max_dist)
            x2, y2 = self.transform_coords(i, max_dist)
            self.canvas.create_line(x1, y1, x2, y2, fill=GRID_COLOR, dash=(2, 2))
            
            # 垂直于Y轴的网格线
            x1, y1 = self.transform_coords(-max_dist, i)
            x2, y2 = self.transform_coords(max_dist, i)
            self.canvas.create_line(x1, y1, x2, y2, fill=GRID_COLOR, dash=(2, 2))

        # 绘制主坐标轴
        # Y 轴 (对应数据中的 x=0)
        x1, y1 = self.transform_coords(0, -1000)
        x2, y2 = self.transform_coords(0, 1000)
        self.canvas.create_line(x1, y1, x2, y2, fill=AXIS_COLOR, width=1.5, arrow=tk.FIRST)
        
        # X 轴 (对应数据中的 y=0)
        x1, y1 = self.transform_coords(-1000, 0)
        x2, y2 = self.transform_coords(1000, 0)
        self.canvas.create_line(x1, y1, x2, y2, fill=AXIS_COLOR, width=1.5, arrow=tk.FIRST)

        # 添加坐标轴标签
        self.canvas.create_text(center_x + 15, 15, text="X+", fill="cyan", font=("Arial", 12))
        self.canvas.create_text(15, center_y - 15, text="Y+", fill="cyan", font=("Arial", 12))
        
        # 绘制比例尺
        p1_x, p1_y = self.transform_coords(0, 0)
        p2_x, p2_y = self.transform_coords(1, 0) # 1个单位长度
        self.canvas.create_line(p1_x, p1_y + 20, p2_x, p2_y + 20, fill='yellow', width=2)
        self.canvas.create_text(
            (p1_x + p2_x) / 2, p1_y + 35,
            text=f"1.0 unit ({SCALE_FACTOR} px)", fill='yellow'
        )

    def draw_points(self):
        """绘制所有数据点"""
        for i in range(len(self.targets)):
            data_point = self.targets[i]
            label = self.labels[i]
            
            # 获取坐标和颜色
            data_x, data_y = data_point[0], data_point[1]
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            
            # 转换为画布坐标
            canvas_x, canvas_y = self.transform_coords(data_x, data_y)
            
            # 计算椭圆的边界框
            x1 = canvas_x - POINT_RADIUS
            y1 = canvas_y - POINT_RADIUS
            x2 = canvas_x + POINT_RADIUS
            y2 = canvas_y + POINT_RADIUS
            
            # 绘制点
            self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)

    def draw_all(self):
        """按顺序绘制所有画布元素"""
        self.draw_axes_and_grid()
        self.draw_points()

if __name__ == '__main__':
    root = tk.Tk()
    app = DataVisualizer(root)
    root.mainloop()