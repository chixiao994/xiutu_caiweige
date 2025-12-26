import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

class BasicModule:
    """基础工具模块"""
    
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
            return [0, 0]
        x_coords = [p for p in points if isinstance(p, (int, float))]
        if not x_coords:
            if points and isinstance(points[0], (list, tuple)):
                x_coords = [p[0] for p in points]
            else:
                return [0, 0]
        return [min(x_coords), max(x_coords)]
    
    @staticmethod
    def get_points(points_list, max_width):
        """获取点集"""
        return points_list
    
    @staticmethod
    def points_to_line(point1, point2):
        """两点确定直线"""
        x1, y1 = point1
        x2, y2 = point2
        if x1 == x2:
            return [float('inf'), x1]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return [k, b]
    
    @staticmethod
    def img_resize(image, new_h=2598):
        """调整图像尺寸到22cm 300DPI"""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_w = int(new_h * aspect_ratio)
        
        # 宽度不超过14cm (1654像素)
        max_width = 1654
        if new_w > max_width:
            new_w = max_width
            new_h = int(new_w / aspect_ratio)
            
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def get_thresh_value(image):
        """获取二值化阈值"""
        return np.mean(image) * 0.7

class SimpleLinearRegression:
    """简单线性回归"""
    
    def __init__(self, points):
        self.points = points
        
    def fit(self, direction='horizontal'):
        """拟合直线"""
        if not self.points:
            return [0, 0]
            
        if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in self.points):
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]
        else:
            if direction == 'horizontal':
                x_coords = list(range(len(self.points)))
                y_coords = self.points
            else:
                y_coords = list(range(len(self.points)))
                x_coords = self.points
                
        if len(x_coords) < 2:
            return [0, np.mean(y_coords) if y_coords else 0]
            
        slope, intercept, _, _, _ = stats.linregress(x_coords, y_coords)
        return [slope, intercept]

class ImageLv1Feature:
    """色阶特征分析"""
    
    def __init__(self, img, direction, if_thresh='auto'):
        self.img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.direction = direction
        self.h, self.w = self.img.shape
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
        
        # 找到色阶变化的边缘点
        gradient = np.gradient(feature)
        threshold = np.mean(np.abs(gradient)) * 2
        
        for i in range(1, len(gradient)-1):
            if abs(gradient[i]) > threshold:
                points.append(i)
        
        # 过滤接近的点
        if filtering and len(points) > 1:
            filtered_points = [points[0]]
            for i in range(1, len(points)):
                if points[i] - filtered_points[-1] >= min_range:
                    filtered_points.append(points[i])
            points = filtered_points
            
        return points

class AncientBookProcessor:
    """古籍书影处理器 - 完整实现文章所有功能"""
    
    def __init__(self):
        # 运行标记图的颜色定义 - 对应文章中的图例
        self.colors = {
            'boundary': (0, 255, 0),      # 绿色 - 图像边界
            'a_vertical': (255, 0, 0),    # 蓝色 - A面垂直框线
            'b_vertical': (0, 0, 255),    # 红色 - B面垂直框线  
            'spine': (255, 255, 0),       # 黄色 - 书口位置
            'middle': (0, 255, 255),      # 青色 - 中缝阴影
            'a_horizontal': (255, 0, 255), # 紫色 - A面水平框线
            'b_horizontal': (255, 165, 0) # 橙色 - B面水平框线
        }
        
        # 特殊情况处理参数
        self.special_case_params = {
            'overlapping_lines': {'cut_offset': 5},
            'severe_tilt': {'max_angle': 15},
            'heavy_noise': {'filter_size': 5, 'edge_min_range': 50},
            'head_notes': {'top_margin_ratio': 0.2}
        }

    def create_marked_image(self, image, processing_data):
        """创建运行标记图 - 对应文章中的程序测试运行标记图"""
        marked = image.copy()
        
        # 绘制边界 (绿色) - 对应图1(1)(2)
        boundaries = processing_data.get('boundaries', {})
        if boundaries:
            cv2.rectangle(marked, 
                         (boundaries['left'], boundaries['top']),
                         (boundaries['right'], boundaries['bottom']), 
                         self.colors['boundary'], 3)
        
        # 绘制垂直框线 - 对应图1(3)(4)(5)(6)
        vertical_lines = processing_data.get('vertical_lines', {})
        for side, lines in vertical_lines.items():
            color = self.colors['a_vertical'] if side == 'a' else self.colors['b_vertical']
            for line in lines:
                if len(line) == 4:
                    cv2.line(marked, (line[0], line[1]), (line[2], line[3]), color, 2)
        
        # 绘制书口位置 (黄色) - 对应图1(7)(8)
        spine_points = processing_data.get('spine_points', [])
        for point in spine_points:
            if len(point) == 2:
                cv2.circle(marked, point, 8, self.colors['spine'], -1)
        
        # 绘制中缝 (青色) - 对应图1(9)
        middle_line = processing_data.get('middle_line')
        if middle_line and len(middle_line) == 4:
            cv2.line(marked, (middle_line[0], middle_line[1]), 
                    (middle_line[2], middle_line[3]), self.colors['middle'], 2)
        
        # 绘制水平框线 - 对应图1(10)(11)
        horizontal_lines = processing_data.get('horizontal_lines', {})
        for side, lines in horizontal_lines.items():
            color = self.colors['a_horizontal'] if side == 'a' else self.colors['b_horizontal']
            for line in lines:
                if len(line) == 4:
                    cv2.line(marked, (line[0], line[1]), (line[2], line[3]), color, 2)
        
        # 绘制角点
        corners = processing_data.get('corners', {})
        for side, points in corners.items():
            color = self.colors['a_vertical'] if side == 'a' else self.colors['b_vertical']
            for point in points:
                if len(point) == 2:
                    cv2.circle(marked, (int(point[0]), int(point[1])), 6, color, -1)
        
        # 添加图例和标注
        self._add_annotations(marked, processing_data)
        
        return marked

    def _add_annotations(self, image, processing_data):
        """添加标注和图例"""
        h, w = image.shape[:2]
        
        # 添加图例
        legend_x = w - 250
        legend_y = 30
        line_height = 25
        
        legend_items = [
            ('边界', self.colors['boundary']),
            ('A面垂直框线', self.colors['a_vertical']),
            ('B面垂直框线', self.colors['b_vertical']),
            ('书口位置', self.colors['spine']),
            ('中缝', self.colors['middle']),
            ('A面水平框线', self.colors['a_horizontal']),
            ('B面水平框线', self.colors['b_horizontal'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # 颜色方块
            cv2.rectangle(image, (legend_x, legend_y + i*line_height),
                         (legend_x + 20, legend_y + i*line_height + 15), color, -1)
            # 文字
            cv2.putText(image, text, (legend_x + 30, legend_y + i*line_height + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # 添加处理信息
        info_text = [
            f"图像尺寸: {w}x{h}",
            f"边界: {processing_data.get('boundaries', {})}",
            f"检测到框线: A面{len(processing_data.get('vertical_lines', {}).get('a', []))}, "
            f"B面{len(processing_data.get('vertical_lines', {}).get('b', []))}",
            f"书口点: {len(processing_data.get('spine_points', []))}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (20, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def handle_special_cases(self, image, case_type, original_image=None):
        """处理特殊情况 - 对应文章中的4种特殊情况"""
        
        if case_type == "overlapping_lines":
            return self._handle_overlapping_lines(image)
        elif case_type == "severe_tilt":
            return self._handle_severe_tilt(image)
        elif case_type == "heavy_noise":
            return self._handle_heavy_noise(image)
        elif case_type == "head_notes":
            return self._handle_head_notes(image, original_image)
        else:
            return image

    def _handle_overlapping_lines(self, image):
        """处理框线重合情况 - 对应图3"""
        # 沿中缝裁切，A面从左边起始，B面从右边起始
        h, w = image.shape[:2]
        middle = w // 2
        
        # 创建两个页面
        a_page = image[:, middle - self.special_case_params['overlapping_lines']['cut_offset']:]
        b_page = image[:, :middle + self.special_case_params['overlapping_lines']['cut_offset']]
        
        return a_page, b_page

    def _handle_severe_tilt(self, image):
        """处理严重倾斜 - 对应图4"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:,0]:
                angle = theta * 180 / np.pi
                # 只考虑接近垂直或水平的线
                if 80 < angle < 100 or angle < 10 or angle > 170:
                    if 80 < angle < 100:  # 垂直线
                        angles.append(angle - 90)
                    else:  # 水平线
                        angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                if abs(avg_angle) > self.special_case_params['severe_tilt']['max_angle']:
                    # 旋转校正
                    center = (w//2, h//2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return corrected
        
        return image

    def _handle_heavy_noise(self, image):
        """处理重度噪点 - 对应图5"""
        params = self.special_case_params['heavy_noise']
        
        # 中值滤波去噪
        filtered = cv2.medianBlur(image, params['filter_size'])
        
        # 形态学操作增强框线
        kernel = np.ones((3,3), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        return filtered

    def _handle_head_notes(self, image, original_image):
        """处理眉批 - 对应图6"""
        # 检测眉批区域并调整天头尺寸
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # 在上部区域检测眉批
        top_region = gray[:h//4, :]
        
        # 使用色阶分析检测眉批
        feature = ImageLv1Feature(top_region, 'horizontal').get_levels_feature()
        threshold = np.mean(feature) * 0.6
        
        # 找到眉批区域
        head_note_height = 0
        for i, value in enumerate(feature):
            if value < threshold:
                head_note_height = i
                break
        
        # 调整天头尺寸
        if head_note_height > 0:
            # 在最终裁切时保留更多天头空间
            params = self.special_case_params['head_notes']
            additional_margin = int(h * params['top_margin_ratio'])
            return image
        
        return image

    def detect_all_elements(self, image):
        """检测所有图像元素 - 对应文章中的完整算法流程"""
        processing_data = {}
        
        # 1. 查找内容边界 (绿色)
        processing_data['boundaries'] = self._find_content_boundary(image)
        
        # 2. 检测垂直框线 (蓝色- A面, 红色- B面)
        processing_data['vertical_lines'] = self._detect_vertical_lines(image, processing_data['boundaries'])
        
        # 3. 查找书口位置 (黄色)
        processing_data['spine_points'] = self._find_book_spine(image, processing_data)
        
        # 4. 定位中缝 (青色)
        processing_data['middle_line'] = self._find_middle_line(image, processing_data)
        
        # 5. 检测水平框线 (紫色- A面, 橙色- B面)
        processing_data['horizontal_lines'] = self._detect_horizontal_lines(image, processing_data)
        
        # 6. 计算角点
        processing_data['corners'] = self._calculate_corners(processing_data)
        
        return processing_data

    def _find_content_boundary(self, image):
        """查找内容边界"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # 使用色阶变化找到边界
        horizontal_feature = ImageLv1Feature(gray, 'horizontal').get_levels_feature()
        vertical_feature = ImageLv1Feature(gray, 'vertical').get_levels_feature()
        
        def find_edge(feature, threshold_ratio=0.3):
            threshold = np.max(feature) * threshold_ratio
            for i in range(len(feature)):
                if feature[i] < threshold:
                    return i
            return 0
        
        left = find_edge(horizontal_feature)
        right = w - find_edge(horizontal_feature[::-1])
        top = find_edge(vertical_feature)
        bottom = h - find_edge(vertical_feature[::-1])
        
        return {
            'left': max(0, left - 10),
            'right': min(w, right + 10),
            'top': max(0, top - 10),
            'bottom': min(h, bottom + 10)
        }

    def _detect_vertical_lines(self, image, boundaries):
        """检测垂直框线"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        lines_dict = {'a': [], 'b': []}
        
        left, right, top, bottom = boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom']
        middle_x = (left + right) // 2
        
        # 检测A面垂直框线
        a_region = gray[top:bottom, middle_x:right]
        a_edges = cv2.Canny(a_region, 50, 150)
        a_lines = cv2.HoughLinesP(a_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if a_lines is not None:
            for line in a_lines:
                x1, y1, x2, y2 = line[0]
                # 转换到原图坐标
                abs_x1, abs_y1 = x1 + middle_x, y1 + top
                abs_x2, abs_y2 = x2 + middle_x, y2 + top
                lines_dict['a'].append([abs_x1, abs_y1, abs_x2, abs_y2])
        
        # 检测B面垂直框线
        b_region = gray[top:bottom, left:middle_x]
        b_edges = cv2.Canny(b_region, 50, 150)
        b_lines = cv2.HoughLinesP(b_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if b_lines is not None:
            for line in b_lines:
                x1, y1, x2, y2 = line[0]
                # 转换到原图坐标
                abs_x1, abs_y1 = x1 + left, y1 + top
                abs_x2, abs_y2 = x2 + left, y2 + top
                lines_dict['b'].append([abs_x1, abs_y1, abs_x2, abs_y2])
        
        return lines_dict

    def _find_book_spine(self, image, processing_data):
        """查找书口位置"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        boundaries = processing_data['boundaries']
        vertical_lines = processing_data['vertical_lines']
        
        left, right, top, bottom = boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom']
        
        # 在垂直框线之间检测书口
        spine_points = []
        search_step = (bottom - top) // 20
        
        for y in range(top, bottom, search_step):
            # 在水平线上检测色阶变化
            line_region = gray[y:y+5, left:right]
            feature = np.mean(line_region, axis=0)
            gradient = np.gradient(feature)
            
            # 找到梯度最大的点作为书口候选
            max_gradient_idx = np.argmax(np.abs(gradient))
            if abs(gradient[max_gradient_idx]) > np.mean(np.abs(gradient)) * 2:
                spine_points.append((max_gradient_idx + left, y))
        
        return spine_points

    def _find_middle_line(self, image, processing_data):
        """定位中缝"""
        spine_points = processing_data['spine_points']
        if len(spine_points) < 2:
            return None
        
        # 使用线性回归拟合中缝线
        x_coords = [p[0] for p in spine_points]
        y_coords = [p[1] for p in spine_points]
        
        if len(x_coords) >= 2:
            slope, intercept = np.polyfit(x_coords, y_coords, 1)
            h, w = image.shape[:2]
            y1, y2 = 0, h
            x1 = int((y1 - intercept) / slope) if slope != 0 else x_coords[0]
            x2 = int((y2 - intercept) / slope) if slope != 0 else x_coords[0]
            return [x1, y1, x2, y2]
        
        return None

    def _detect_horizontal_lines(self, image, processing_data):
        """检测水平框线"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        boundaries = processing_data['boundaries']
        lines_dict = {'a': [], 'b': []}
        
        left, right, top, bottom = boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom']
        middle_x = (left + right) // 2
        
        # 在上下区域检测水平线
        for y_region in [top + (bottom-top)//4, top + (bottom-top)*3//4]:
            region = gray[y_region-20:y_region+20, left:right]
            edges = cv2.Canny(region, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    abs_y = (y1 + y2) // 2 + y_region - 20
                    # 根据x坐标判断属于哪个面
                    line_middle = (x1 + x2) // 2 + left
                    side = 'a' if line_middle > middle_x else 'b'
                    lines_dict[side].append([left, abs_y, right, abs_y])
        
        return lines_dict

    def _calculate_corners(self, processing_data):
        """计算角点"""
        boundaries = processing_data['boundaries']
        vertical_lines = processing_data['vertical_lines']
        horizontal_lines = processing_data['horizontal_lines']
        
        corners = {'a': [], 'b': []}
        
        # 简化计算，实际应该根据框线交点计算
        left, right, top, bottom = boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom']
        middle_x = (left + right) // 2
        
        # A面角点
        corners['a'] = [
            [middle_x, top],
            [right, top],
            [right, bottom],
            [middle_x, bottom]
        ]
        
        # B面角点
        corners['b'] = [
            [left, top],
            [middle_x, top],
            [middle_x, bottom],
            [left, bottom]
        ]
        
        return corners

    def process_complete_workflow(self, image_path, output_dir, create_marked=True, handle_special=True):
        """完整处理流程"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        original_image = image.copy()
        
        # 检测所有元素
        processing_data = self.detect_all_elements(image)
        
        # 创建运行标记图
        if create_marked:
            marked_image = self.create_marked_image(image, processing_data)
            marked_path = Path(output_dir) / (Path(image_path).stem + '_marked.jpg')
            cv2.imwrite(str(marked_path), marked_image)
        
        # 处理特殊情况
        if handle_special:
            # 检测并处理各种特殊情况
            image = self._auto_detect_and_handle_special_cases(image, original_image, processing_data)
        
        # 分割页面并保存最终结果
        a_page, b_page = self._split_and_save_pages(image, processing_data, output_dir, Path(image_path).stem)
        
        return a_page, b_page, processing_data

    def _auto_detect_and_handle_special_cases(self, image, original_image, processing_data):
        """自动检测并处理特殊情况"""
        current_image = image.copy()
        
        # 检测框线重合
        if self._detect_overlapping_lines(processing_data):
            current_image = self._handle_overlapping_lines(current_image)
        
        # 检测严重倾斜
        if self._detect_severe_tilt(processing_data):
            current_image = self._handle_severe_tilt(current_image)
        
        # 检测重度噪点
        if self._detect_heavy_noise(current_image):
            current_image = self._handle_heavy_noise(current_image)
        
        # 检测眉批
        if self._detect_head_notes(processing_data):
            current_image = self._handle_head_notes(current_image, original_image)
        
        return current_image

    def _detect_overlapping_lines(self, processing_data):
        """检测框线重合"""
        vertical_lines = processing_data.get('vertical_lines', {})
        a_lines = vertical_lines.get('a', [])
        b_lines = vertical_lines.get('b', [])
        
        if a_lines and b_lines:
            # 检查A面和B面框线是否接近
            a_min_x = min([min(l[0], l[2]) for l in a_lines])
            b_max_x = max([max(l[0], l[2]) for l in b_lines])
            return abs(a_min_x - b_max_x) < 20
        return False

    def _detect_severe_tilt(self, processing_data):
        """检测严重倾斜"""
        vertical_lines = processing_data.get('vertical_lines', {})
        all_lines = vertical_lines.get('a', []) + vertical_lines.get('b', [])
        
        if all_lines:
            angles = []
            for line in all_lines:
                x1, y1, x2, y2 = line
                if x2 - x1 != 0:
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    angles.append(angle)
            
            if angles:
                angle_std = np.std(angles)
                return angle_std > 10  # 角度标准差大于10度认为有严重倾斜
        
        return False

    def _detect_heavy_noise(self, image):
        """检测重度噪点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # 计算图像的噪声水平
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 100  # 方差小于100认为噪声较重

    def _detect_head_notes(self, processing_data):
        """检测眉批"""
        boundaries = processing_data.get('boundaries', {})
        horizontal_lines = processing_data.get('horizontal_lines', {}).get('a', [])
        
        if horizontal_lines and boundaries:
            top_line_y = min([min(l[1], l[3]) for l in horizontal_lines])
            top_boundary = boundaries.get('top', 0)
            # 如果水平框线距离上边界较远，可能有眉批
            return (top_line_y - top_boundary) > 50
        return False

    def _split_and_save_pages(self, image, processing_data, output_dir, base_name):
        """分割并保存页面"""
        corners = processing_data.get('corners', {})
        boundaries = processing_data.get('boundaries', {})
        
        # 分割A面
        a_points = np.array(corners.get('a', []), dtype=np.float32)
        a_page = self._perspective_correct(image, a_points, boundaries, 'a')
        
        # 分割B面
        b_points = np.array(corners.get('b', []), dtype=np.float32)
        b_page = self._perspective_correct(image, b_points, boundaries, 'b')
        
        # 保存结果
        a_path = Path(output_dir) / (base_name + '_a.jpg')
        b_path = Path(output_dir) / (base_name + '_b.jpg')
        
        cv2.imwrite(str(a_path), a_page)
        cv2.imwrite(str(b_path), b_page)
        
        return str(a_path), str(b_path)

    def _perspective_correct(self, image, points, boundaries, side):
        """透视校正"""
        h, w = image.shape[:2]
        
        if len(points) < 4:
            # 如果没有检测到足够的点，使用简单分割
            middle_x = w // 2
            if side == 'a':
                return image[:, middle_x:]
            else:
                return image[:, :middle_x]
        
        # 透视变换
        left = int(min(points[:, 0]))
        right = int(max(points[:, 0]))
        top = int(min(points[:, 1]))
        bottom = int(max(points[:, 1]))
        
        dst_points = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(points, dst_points)
        warped = cv2.warpPerspective(image, M, (w, h))
        
        # 裁切并调整尺寸
        cropped = warped[top:bottom, left:right]
        final = BasicModule.img_resize(cropped)
        
        return final

# 使用示例
def main():
    processor = AncientBookProcessor()
    
    # 处理单张图像
    input_image = "ancient_book.jpg"
    output_folder = "processed"
    
    try:
        a_path, b_path, processing_data = processor.process_complete_workflow(
            input_image, output_folder, create_marked=True, handle_special=True)
        
        print(f"处理完成:")
        print(f"A面: {a_path}")
        print(f"B面: {b_path}")
        print(f"检测到 {len(processing_data.get('spine_points', []))} 个书口点")
        print(f"检测到 {len(processing_data.get('vertical_lines', {}).get('a', []))} 条A面垂直框线")
        print(f"检测到 {len(processing_data.get('vertical_lines', {}).get('b', []))} 条B面垂直框线")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()
