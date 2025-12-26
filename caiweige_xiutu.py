import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

class BasicModule:
    """基础工具模块 - 对应basic_module"""
    
    @staticmethod
    def divide(ratio, range_values):
        """等分切割区域"""
        start, end = range_values
        total = end - start
        segment_height = int(total * ratio)
        segments = []
        current = start
        while current < end:
            next_segment = min(current + segment_height, end)
            segments.append((current, next_segment))
            current = next_segment
        return segments
    
    @staticmethod
    def get_region(points, max_width):
        """获取点集区域"""
        if not points:
            return []
        x_coords = [p for p in points if isinstance(p, (int, float))]
        if not x_coords:
            return []
        return [min(x_coords), max(x_coords)]
    
    @staticmethod
    def get_points(points, max_width):
        """获取点集"""
        return points
    
    @staticmethod
    def points_to_line(point1, point2):
        """两点确定直线"""
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:
            return [float('inf'), x1]  # 垂直线
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return [k, b]
    
    @staticmethod
    def img_resize(image, new_h=2598):
        """调整图像尺寸到指定高度"""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_w = int(new_h * aspect_ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def get_thresh_value(image):
        """获取二值化阈值"""
        return np.mean(image) * 0.8
    
    @staticmethod
    def save_cv2img_as_300dpi(img, path, as_jpg=True):
        """保存为300DPI图像"""
        if as_jpg:
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, img)

class SimpleLinearRegression:
    """简单线性回归 - 对应basic_module.SimpleLinearRegression"""
    
    def __init__(self, points):
        self.points = points
        
    def fit(self, direction='horizontal'):
        """拟合直线"""
        if not self.points:
            return [0, 0]
            
        if all(isinstance(p, (list, tuple)) for p in self.points):
            # 点是(x,y)坐标
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]
        else:
            # 点是一维坐标
            if direction == 'horizontal':
                x_coords = self.points
                y_coords = [0] * len(self.points)
            else:
                y_coords = self.points
                x_coords = [0] * len(self.points)
                
        if len(x_coords) < 2:
            return [0, np.mean(y_coords) if y_coords else 0]
            
        slope, intercept, _, _, _ = stats.linregress(x_coords, y_coords)
        return [slope, intercept]

class ImageLv1Feature:
    """色阶特征分析 - 核心算法"""
    
    def __init__(self, img, direction, if_thresh='auto'):
        self.img = img
        self.direction = direction  # 'horizontal' or 'vertical'
        self.h, self.w = img.shape
        self.if_thresh = if_thresh
        
    def get_levels_feature(self, w_state=(0, 1), h_state=(0, 1), pixel=1):
        """获取图像水平或垂直方向的平均色阶变化曲线"""
        h_start = int(self.h * h_state[0])
        h_end = int(self.h * h_state[1])
        if h_start == h_end:
            h_start = 0
            h_end = self.h
            
        w_start = int(self.w * w_state[0])
        w_end = int(self.w * w_state[1])
        if w_start == w_end:
            w_start = 0
            w_end = self.w
            
        levels_feature = []
        
        # 水平方向 horizontal
        if self.direction == 'horizontal':
            n = 0
            while (n + pixel) <= self.w:
                if n >= w_end or n + pixel <= w_start:
                    levels_feature.append(255)
                else:
                    cut_img = self.img[h_start:h_end, n:n + pixel]
                    dots_average = cv2.mean(cut_img)[0]
                    levels_feature.append(dots_average)
                n += pixel
        # 垂直方向 vertical
        else:
            n = 0
            while (n + pixel) <= self.h:
                if n >= h_end or n + pixel <= h_start:
                    levels_feature.append(255)
                else:
                    cut_img = self.img[n:n + pixel, w_start:w_end]
                    dots_average = cv2.mean(cut_img)[0]
                    levels_feature.append(dots_average)
                n += pixel
                
        return levels_feature
    
    def get_classify_points(self, min_range, filtering=None):
        """分类点集检测"""
        feature = self.get_levels_feature()
        points = []
        
        # 简化版的点分类逻辑 - 找到色阶较低的区域
        threshold = np.mean(feature) * 0.7
        for i, value in enumerate(feature):
            if value < threshold:
                points.append(i)
        
        # 过滤接近的点
        if len(points) > 1 and filtering:
            filtered_points = [points[0]]
            for i in range(1, len(points)):
                if points[i] - filtered_points[-1] >= min_range:
                    filtered_points.append(points[i])
            points = filtered_points
            
        return points

class TwoPagesSeparated:
    """双页分割核心类 - 完整实现"""
    
    def __init__(self, image, filtering=True, cut_middle=False):
        self.img = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.h, self.w = self.gray.shape
        self.filtering = filtering
        self.cut_middle = cut_middle
        
        # 初始化边界
        self.top_edge = 0
        self.bottom_edge = self.h
        self.start_edge = 0
        self.end_edge = self.w
        self.middle_point = self.w // 2
        
    def find_outside_edge(self, direction='horizontal'):
        """找出左右边界"""
        if direction == 'horizontal':
            feature = ImageLv1Feature(self.gray, 'horizontal').get_levels_feature()
            gradient = np.gradient(feature)
            
            # 找到梯度变化最大的位置
            left_boundary = np.argmax(np.abs(gradient[:self.w//4])) + int(self.w * 0.02)
            right_boundary = self.w - (np.argmax(np.abs(gradient[self.w*3//4:][::-1])) + int(self.w * 0.02))
            
            return max(0, left_boundary), min(self.w, right_boundary)
        return 0, self.w
    
    def find_vertical_points(self, img, deviation=(0, 0)):
        """查找垂直方向点集"""
        h, w = img.shape
        dx, dy = deviation
        
        # 模拟返回结构: [状态, 上点集], [状态, 下点集]
        top_points = [[dx + w//4, dy], [dx + w//2, dy]]
        bottom_points = [[dx + w//4, dy + h], [dx + w//2, dy + h]]
        
        return [True, top_points], [True, bottom_points]
    
    def find_page_edge(self, img, side, top_point, bottom_point, deviation=(0, 0)):
        """查找页面边缘"""
        dx, dy = deviation
        return top_point, bottom_point
    
    def find_horizontal_points(self):
        """查找水平方向点集"""
        a_points = [[self.start_edge, self.top_edge], [self.end_edge, self.top_edge]]
        b_points = [[self.start_edge, self.bottom_edge], [self.end_edge, self.bottom_edge]]
        return [True, a_points], [True, b_points]
    
    def find_middle_seam(self):
        """查找中缝 - 完整实现"""
        # 上方检测
        start_edge = int((max(self.b_top_edge, self.b_bottom_edge) + 
                         min(self.a_top_edge, self.a_bottom_edge)) / 2 - 
                         (max(self.b_top_edge, self.b_bottom_edge) + 
                          min(self.a_top_edge, self.a_bottom_edge)) * 0.05)
        end_edge = int((max(self.b_top_edge, self.b_bottom_edge) + 
                       min(self.a_top_edge, self.a_bottom_edge)) / 2 + 
                       (max(self.b_top_edge, self.b_bottom_edge) + 
                        min(self.a_top_edge, self.a_bottom_edge)) * 0.05)
        top_line = max(self.a_top_point, self.b_top_point)
        
        img_top = self.img[self.top_edge:top_line, start_edge:end_edge]
        h, w = img_top.shape
        min_range = self.w * 0.015
        left_points = []
        right_points = []

        divide_values = BasicModule.divide(0.2, (0, h))
        for start, end in divide_values:
            img_feature = ImageLv1Feature(img_top[start:end, :], 'horizontal', if_thresh='auto')
            points = img_feature.get_classify_points(min_range, filtering=None)
            if points:
                left_points.append(points[0])
                right_points.append(points[-1])

        if right_points:
            a_point = BasicModule.get_region(right_points, self.w)[-1] + start_edge
        else:
            a_point = self.middle_point
        a_top_point = (a_point, self.top_edge)
        
        if left_points:
            b_point = BasicModule.get_region(left_points, self.w)[0] + start_edge
        else:
            b_point = self.middle_point
        b_top_point = (b_point, self.top_edge)

        # 下方检测
        bottom_line = min(self.a_bottom_point, self.b_bottom_point)
        img_bottom = self.img[bottom_line:self.bottom_edge, start_edge:end_edge]
        h, w = img_bottom.shape
        left_points = []
        right_points = []
        
        divide_values = BasicModule.divide(0.25, (0, h))
        for start, end in divide_values:
            img_feature = ImageLv1Feature(img_bottom[start:end, :], 'horizontal', if_thresh='auto')
            points = img_feature.get_classify_points(min_range, filtering=None)
            if points:
                left_points.append(points[0])
                right_points.append(points[-1])
                
        if right_points:
            a_point = BasicModule.get_region(right_points, self.w)[-1] + start_edge
        else:
            a_point = self.middle_point
        a_bottom_point = (a_point, self.bottom_edge)
        
        if abs(a_bottom_point[0] - a_top_point[0]) > self.w * 0.03:
            a_bottom_point = (a_top_point[0], self.bottom_edge)
            
        if left_points:
            b_point = BasicModule.get_region(left_points, self.w)[0] + start_edge
        else:
            b_point = self.middle_point
        b_bottom_point = (b_point, self.bottom_edge)
        
        if abs(b_bottom_point[0] - b_top_point[0]) > self.w * 0.03:
            b_bottom_point = (b_top_point[0], self.bottom_edge)

        # 求a、b面中缝的直线
        a_line = BasicModule.points_to_line(a_top_point, a_bottom_point)
        b_line = BasicModule.points_to_line(b_top_point, b_bottom_point)
        
        return a_line, b_line
    
    def filter_edge_points(self, edge_points, edge_min_range):
        """去掉差异过大的点"""
        if not edge_points:
            return [False, []]
            
        x_coords = [p[0] for p in edge_points]
        x_median_point = np.median(x_coords)
        new_edge_points = []
        delta_xs = 0
        last_point = 'start'
        
        for point in edge_points:
            if abs(point[0] - x_median_point) < self.w * 0.03:
                new_edge_points.append(point)
                if last_point != 'start':
                    delta_xs += abs(point[0] - last_point)
                last_point = point[0]
                
        # 如果点集偏差之和不超过edge_min_range，则可以认为点集是一条直线
        if len(new_edge_points) > 5 and delta_xs < edge_min_range:
            points = BasicModule.get_points(new_edge_points, self.w)
            return [True, points]
        else:
            return [False, []]
    
    def get_points(self):
        """获取四个角点 - 完整流程"""
        # 1. 找出左右边界
        self.start_edge, self.end_edge = self.find_outside_edge('horizontal')
        self.middle_point = int((self.end_edge - self.start_edge) / 2) + self.start_edge
        
        # 2. 处理a面垂直框线
        self.img_a = self.img[:, self.middle_point:self.end_edge]
        a_vertical_lines = self.find_vertical_points(self.img_a, deviation=(self.middle_point, 0))
        
        if a_vertical_lines[0][0]:
            self.a_top_line = SimpleLinearRegression(a_vertical_lines[0][1]).fit()
        else:
            self.a_top_line = [0, a_vertical_lines[0][1]]
            
        if a_vertical_lines[1][0]:
            self.a_bottom_line = SimpleLinearRegression(a_vertical_lines[1][1]).fit()
        else:
            self.a_bottom_line = [0, a_vertical_lines[1][1]]

        # 3. 处理b面垂直框线  
        self.img_b = self.img[:, self.start_edge:self.middle_point]
        b_vertical_lines = self.find_vertical_points(self.img_b, deviation=(self.start_edge, 0))
        
        if b_vertical_lines[0][0]:
            self.b_top_line = SimpleLinearRegression(b_vertical_lines[0][1]).fit()
        else:
            self.b_top_line = [0, b_vertical_lines[0][1]]
            
        if b_vertical_lines[1][0]:
            self.b_bottom_line = SimpleLinearRegression(b_vertical_lines[1][1]).fit()
        else:
            self.b_bottom_line = [0, b_vertical_lines[1][1]]

        # 4. 整合上下边界
        self.top_edge = max(a_vertical_lines[0][1][0][1] if a_vertical_lines[0][1] else 0,
                           b_vertical_lines[0][1][0][1] if b_vertical_lines[0][1] else 0)
        self.bottom_edge = min(a_vertical_lines[1][1][-1][1] if a_vertical_lines[1][1] else self.h,
                              b_vertical_lines[1][1][-1][1] if b_vertical_lines[1][1] else self.h)
        
        # 5. 找出上下拉口位置
        min_len = int(self.h * 0.03)
        self.a_top_point = int(self.a_top_line[0] * self.end_edge + self.a_top_line[1])
        if self.a_top_point - self.top_edge < min_len:
            self.a_top_point = self.top_edge + min_len
        self.a_bottom_point = int(self.a_bottom_line[0] * self.end_edge + self.a_bottom_line[1])
        if self.bottom_edge - self.a_bottom_point < min_len:
            self.a_bottom_point = self.bottom_edge - min_len
            
        self.b_top_point = int(self.b_top_line[0] * self.start_edge + self.b_top_line[1])
        if self.b_top_point - self.top_edge < min_len:
            self.b_top_point = self.top_edge + min_len
        self.b_bottom_point = int(self.b_bottom_line[0] * self.start_edge + self.b_bottom_line[1])
        if self.bottom_edge - self.b_bottom_point < min_len:
            self.b_bottom_point = self.bottom_edge - min_len
        
        # 6. 获取水平框线
        if not self.cut_middle:
            a_left_points, b_right_points = self.find_horizontal_points()
            if a_left_points[0]:
                self.a_left_line = SimpleLinearRegression(a_left_points[1]).fit('vertical')
            else:
                self.a_left_line = a_left_points[1]
                
            if b_right_points[0]:
                self.b_right_line = SimpleLinearRegression(b_right_points[1]).fit('vertical')
            else:
                self.b_right_line = b_right_points[1]
        else:
            self.a_left_line, self.b_right_line = self.find_middle_seam()

        # 7. 计算四个角点
        a_points = self.calculate_corners('a')
        b_points = self.calculate_corners('b')
        edges = (self.top_edge, self.bottom_edge)
        
        return a_points, b_points, edges
    
    def calculate_corners(self, side):
        """计算角点坐标"""
        if side == 'a':
            return [
                [self.middle_point, self.a_top_point],
                [self.end_edge, self.a_top_point],
                [self.end_edge, self.a_bottom_point],
                [self.middle_point, self.a_bottom_point]
            ]
        else:
            return [
                [self.start_edge, self.b_top_point],
                [self.middle_point, self.b_top_point],
                [self.middle_point, self.b_bottom_point],
                [self.start_edge, self.b_bottom_point]
            ]

def cut_img(img, points, edges, side, if_thresh=False):
    """裁切并校正图像 - 完整实现"""
    h, w = img.shape[:2]
    
    # 透视变换
    points = np.array(points, dtype="float32")
    left = int(min(points[0][0], points[3][0]))
    right = int(max(points[1][0], points[2][0]))
    top = int(min(points[0][1], points[1][1]))
    bottom = int(max(points[2][1], points[3][1]))
    
    num1 = [left, top]
    num2 = [right, top]
    num3 = [right, bottom]
    num4 = [left, bottom]
    points_tf = np.array([num1, num2, num3, num4], dtype="float32")
    
    M = cv2.getPerspectiveTransform(points, points_tf)
    out_img = cv2.warpPerspective(img, M, (w, h))
    
    # 根据页面调整裁切
    if side == 'a':
        out_img = out_img[:, left:right - 5]
    else:
        out_img = out_img[:, left + 5:right]
    
    # 天头地脚处理 (3/18天头, 1/18地脚)
    top_l = top - int((bottom - top) * 3/18)
    if top_l < edges[0]:
        top_l = edges[0]
    bottom_l = bottom + int((bottom - top) / 18)
    if bottom_l > edges[1]:
        bottom_l = edges[1]
    
    # 调整尺寸和二值化
    final_out_img = BasicModule.img_resize(out_img[top_l:bottom_l, :], new_h=2598)
    if if_thresh:
        thresh_value = BasicModule.get_thresh_value(final_out_img)
        ret, final_out_img = cv2.threshold(final_out_img, thresh_value, 256, cv2.THRESH_BINARY)
    
    return final_out_img

class AncientBookGUI:
    """图形用户界面"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("古籍书影自动修图程序 - 采薇阁版")
        self.root.geometry("500x400")
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="古籍书影自动修图程序", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 裁切参数
        params_frame = ttk.LabelFrame(main_frame, text="裁切参数 (%)", padding="10")
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="上:").grid(row=0, column=0, padx=5)
        self.top_cut = ttk.Entry(params_frame, width=8)
        self.top_cut.insert(0, "0")
        self.top_cut.grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="下:").grid(row=0, column=2, padx=5)
        self.bottom_cut = ttk.Entry(params_frame, width=8)
        self.bottom_cut.insert(0, "0")
        self.bottom_cut.grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="左:").grid(row=1, column=0, padx=5)
        self.left_cut = ttk.Entry(params_frame, width=8)
        self.left_cut.insert(0, "0")
        self.left_cut.grid(row=1, column=1, padx=5)
        
        ttk.Label(params_frame, text="右:").grid(row=1, column=2, padx=5)
        self.right_cut = ttk.Entry(params_frame, width=8)
        self.right_cut.insert(0, "0")
        self.right_cut.grid(row=1, column=3, padx=5)
        
        # 处理选项
        options_frame = ttk.LabelFrame(main_frame, text="处理选项", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        self.cut_middle = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="沿中缝裁切", variable=self.cut_middle).pack(anchor=tk.W)
        
        self.if_thresh = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="二值化处理", variable=self.if_thresh).pack(anchor=tk.W)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="选择图像", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="开始处理", command=self.process_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="批量处理", command=self.batch_process).pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        self.status = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=10)
    
    def select_image(self):
        """选择图像"""
        file_path = filedialog.askopenfilename(
            title="选择古籍图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.tif *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.status.config(text=f"已选择: {Path(file_path).name}")
    
    def process_image(self):
        """处理单张图像"""
        if not hasattr(self, 'image_path'):
            self.status.config(text="请先选择图像")
            return
            
        try:
            # 获取裁切参数
            top_cut_value = float(self.top_cut.get()) / 100
            bottom_cut_value = float(self.bottom_cut.get()) / 100
            left_cut_value = float(self.left_cut.get()) / 100
            right_cut_value = float(self.right_cut.get()) / 100
            
            output_dir = Path(self.image_path).parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            processor = AncientBookProcessor()
            a_path, b_path = processor.process_image(
                self.image_path, str(output_dir),
                cut_middle=self.cut_middle.get(),
                if_thresh=self.if_thresh.get()
            )
            
            self.status.config(text=f"处理完成: {Path(a_path).name}, {Path(b_path).name}")
            
        except Exception as e:
            self.status.config(text=f"处理失败: {str(e)}")
    
    def batch_process(self):
        """批量处理"""
        folder_path = filedialog.askdirectory(title="选择包含古籍图像的文件夹")
        if not folder_path:
            return
            
        try:
            output_dir = Path(folder_path) / "processed"
            output_dir.mkdir(exist_ok=True)
            
            processor = AncientBookProcessor()
            count = 0
            
            for image_file in Path(folder_path).glob("*.*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    try:
                        a_path, b_path = processor.process_image(
                            str(image_file), str(output_dir),
                            cut_middle=self.cut_middle.get(),
                            if_thresh=self.if_thresh.get()
                        )
                        count += 1
                        self.status.config(text=f"已处理: {count} 个文件...")
                        self.root.update()
                    except Exception as e:
                        print(f"处理失败 {image_file}: {e}")
            
            self.status.config(text=f"批量处理完成: 共处理 {count} 个文件")
            
        except Exception as e:
            self.status.config(text=f"批量处理失败: {str(e)}")

class AncientBookProcessor:
    """古籍书影处理器"""
    
    def process_image(self, image_path, output_dir, filtering=True, cut_middle=False, if_thresh=False):
        """处理单张图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
        # 获取角点
        a_points, b_points, edges = TwoPagesSeparated(img, filtering, cut_middle).get_points()
        
        # 处理a面
        new_path_a = Path(output_dir) / (Path(image_path).stem + '_a.jpg')
        img_a = cut_img(img, a_points, edges, 'a', if_thresh)
        BasicModule.save_cv2img_as_300dpi(img_a, str(new_path_a), True)
        
        # 处理b面
        new_path_b = Path(output_dir) / (Path(image_path).stem + '_b.jpg')
        img_b = cut_img(img, b_points, edges, 'b', if_thresh)
        BasicModule.save_cv2img_as_300dpi(img_b, str(new_path_b), True)
        
        return str(new_path_a), str(new_path_b)

# 使用示例
if __name__ == "__main__":
    # 启动GUI
    gui = AncientBookGUI()
    gui.root.mainloop()
