"""
古籍书影智能处理系统 v4.2 - 优化版图形界面
单文件版本，无需额外依赖
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageOps

# ============================================================================
# 核心处理器类
# ============================================================================

class BookType(Enum):
    """古籍类型枚举"""
    STANDARD = "standard"           # 标准刻本
    COLOR_PRINT = "color_print"     # 套红/朱墨本
    RUBBING = "rubbing"             # 拓本
    ANNOTATED = "annotated"         # 批校本
    MAP_STYLE = "map_style"         # 地图式古籍
    DAMAGED = "damaged"             # 损伤古籍
    MIXED = "mixed"                 # 混合类型

@dataclass
class PageBoundary:
    """页面边界信息"""
    left: int
    right: int
    top: int
    bottom: int
    confidence: float = 1.0
    
    @property
    def width(self):
        return self.right - self.left
    
    @property
    def height(self):
        return self.bottom - self.top
    
    @property
    def area(self):
        return self.width * self.height

@dataclass
class DetectedLine:
    """检测到的线条"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    line_type: str  # 'vertical', 'horizontal', 'diagonal'
    color: Optional[Tuple[int, int, int]] = None

@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    page_type: BookType
    boundaries: PageBoundary
    vertical_lines: List[DetectedLine]
    horizontal_lines: List[DetectedLine]
    spine_points: List[Tuple[int, int]]
    corners: Dict[str, List[Tuple[int, int]]]
    processing_time: float
    confidence: float
    warnings: List[str]
    debug_info: Dict

class AncientBookProcessor:
    """古籍书影智能处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 颜色定义 - 使用更柔和的颜色
        self.colors = {
            'boundary': (0, 200, 0),        # 柔和的绿色 - 边界
            'a_boundary': (200, 100, 0),    # 橙色 - A面边界
            'b_boundary': (0, 100, 200),    # 青色 - B面边界
            'a_vertical': (255, 50, 50),    # 亮蓝色 - A面垂直线
            'b_vertical': (50, 50, 255),    # 亮红色 - B面垂直线
            'spine': (255, 255, 100),       # 淡黄色 - 书口
            'middle': (100, 255, 255),      # 淡青色 - 中缝
            'a_horizontal': (255, 100, 255), # 淡紫色 - A面水平线
            'b_horizontal': (255, 200, 100), # 淡橙色 - B面水平线
        }
        
        # 默认配置
        self.config = {
            'preprocessing': {'dpi': 200, 'max_size': 2000},
            'edge_detection': {'canny_low': 30, 'canny_high': 100},
            'line_detection': {'hough_threshold': 80, 'min_line_length': 50},
            'output': {'create_debug_images': True}
        }
        
        # 内部状态
        self.processing_history = []
        self.statistics = {
            'total_pages': 0,
            'successful_pages': 0,
            'total_time': 0.0,
            'avg_time_per_page': 0.0,
        }
    
    def process_image(self, image_path: str, output_dir: str = "output") -> ProcessingResult:
        """处理单张古籍图像"""
        start_time = time.time()
        
        try:
            # 读取图像 - 保持颜色通道顺序
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_h, original_w = image.shape[:2]
            
            # 简化处理流程
            # 1. 识别古籍类型
            book_type = self._identify_book_type_simple(image)
            
            # 2. 预处理（修正：减少对比度增强，使色调更自然）
            processed = self._simple_preprocess(image, book_type)
            
            # 3. 检测边界（修正：扩大边界检测范围）
            boundaries = self._detect_boundaries_simple(processed)
            
            # 4. 检测线条
            vertical_lines = self._detect_vertical_lines_simple(processed, boundaries)
            horizontal_lines = self._detect_horizontal_lines_simple(processed, boundaries)
            
            # 5. 检测书口点
            spine_points = self._detect_spine_points_simple(processed, boundaries)
            
            # 6. 计算角点
            corners = self._calculate_corners_simple(boundaries, vertical_lines, horizontal_lines)
            
            # 7. 分割页面
            a_page, b_page = self._split_pages_simple(processed, corners)
            
            # 8. 保存结果
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            a_path = str(Path(output_dir) / f"{base_name}_a.jpg")
            b_path = str(Path(output_dir) / f"{base_name}_b.jpg")
            
            # 保存分割后的页面
            cv2.imwrite(a_path, cv2.cvtColor(a_page, cv2.COLOR_RGB2BGR) if len(a_page.shape) == 3 and a_page.shape[2] == 3 else a_page)
            cv2.imwrite(b_path, cv2.cvtColor(b_page, cv2.COLOR_RGB2BGR) if len(b_page.shape) == 3 and b_page.shape[2] == 3 else b_page)
            
            # 9. 创建调试图像（这既是调试图像也是标记预览）
            debug_path = str(Path(output_dir) / f"{base_name}_debug.jpg")
            debug_image = self._create_debug_image_enhanced(processed, boundaries, 
                                                          vertical_lines, horizontal_lines,
                                                          spine_points, corners)
            cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR) if len(debug_image.shape) == 3 and debug_image.shape[2] == 3 else debug_image)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 计算置信度
            confidence = self._calculate_confidence_simple(boundaries, len(vertical_lines), 
                                                          len(horizontal_lines), len(spine_points))
            
            # 创建结果对象
            result = ProcessingResult(
                success=True,
                page_type=book_type,
                boundaries=boundaries,
                vertical_lines=vertical_lines,
                horizontal_lines=horizontal_lines,
                spine_points=spine_points,
                corners=corners,
                processing_time=processing_time,
                confidence=confidence,
                warnings=[],
                debug_info={
                    'original_size': (original_w, original_h),
                    'output_paths': {'a': a_path, 'b': b_path},
                    'debug_image': debug_path,
                    'book_type': book_type.value,
                }
            )
            
            # 更新统计
            self._update_statistics(result)
            
            return result
            
        except Exception as e:
            # 创建失败结果
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                page_type=BookType.STANDARD,
                boundaries=PageBoundary(0, 0, 0, 0),
                vertical_lines=[],
                horizontal_lines=[],
                spine_points=[],
                corners={'a': [], 'b': []},
                processing_time=processing_time,
                confidence=0.0,
                warnings=[f"处理失败: {str(e)}"],
                debug_info={'error': str(e)}
            )
    
    def _identify_book_type_simple(self, image: np.ndarray) -> BookType:
        """简化版古籍类型识别"""
        # 这里简化处理，总是返回标准类型
        return BookType.STANDARD
    
    def _simple_preprocess(self, image: np.ndarray, book_type: BookType) -> np.ndarray:
        """简化版预处理 - 修正：减少对比度增强"""
        # 转换BGR到RGB以保持颜色正确性
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()
            
        h, w = processed.shape[:2]
        
        # 调整尺寸
        max_size = self.config['preprocessing']['max_size']
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 如果图像是灰度图，转换为三通道
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        # 修正：轻微增强对比度，而不是强烈增强
        if len(processed.shape) == 3:
            # 轻微调整亮度和对比度
            alpha = 1.1  # 对比度系数 (1.0表示不变)
            beta = 10    # 亮度增量
            
            # 应用亮度和对比度调整
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
            
            # 轻微的色彩平衡
            # 分离通道
            b, g, r = cv2.split(processed)
            
            # 轻微调整各个通道
            r = cv2.add(r, 5)
            g = cv2.add(g, 5)
            b = cv2.add(b, 5)
            
            # 合并通道
            processed = cv2.merge([b, g, r])
        
        return processed
    
    def _detect_boundaries_simple(self, image: np.ndarray) -> PageBoundary:
        """简化版边界检测 - 修正：扩大边界检测范围"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape
        
        # 使用更宽松的边缘检测参数
        edges = cv2.Canny(gray, 20, 80)
        
        # 形态学操作，连接边缘
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest = max(contours, key=cv2.contourArea)
            x, y, w_rect, h_rect = cv2.boundingRect(largest)
            
            # 计算置信度
            contour_area = cv2.contourArea(largest)
            bbox_area = w_rect * h_rect
            confidence = contour_area / bbox_area if bbox_area > 0 else 0.5
            
            # 扩大边界（修正：扩大更多）
            margin_h = int(h_rect * 0.02)  # 2%的高度作为边距
            margin_w = int(w_rect * 0.02)  # 2%的宽度作为边距
            
            return PageBoundary(
                left=max(0, x - margin_w),
                right=min(w, x + w_rect + margin_w),
                top=max(0, y - margin_h),
                bottom=min(h, y + h_rect + margin_h),
                confidence=min(confidence, 1.0)
            )
        
        # 默认使用整个图像
        return PageBoundary(0, w, 0, h, confidence=0.5)
    
    def _detect_vertical_lines_simple(self, image: np.ndarray, boundaries: PageBoundary) -> List[DetectedLine]:
        """简化版垂直线检测"""
        lines = []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 提取ROI
        roi = gray[boundaries.top:boundaries.bottom, boundaries.left:boundaries.right]
        
        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)
        
        # 霍夫变换检测直线
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                
                # 转换到原图坐标
                abs_x1 = x1 + boundaries.left
                abs_y1 = y1 + boundaries.top
                abs_x2 = x2 + boundaries.left
                abs_y2 = y2 + boundaries.top
                
                # 检查是否为垂直线
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if dx < 10 and dy > 50:  # 接近垂直且足够长
                    # 计算置信度
                    length = np.sqrt(dx**2 + dy**2)
                    confidence = min(length / boundaries.height, 1.0)
                    
                    # 确定颜色（A面或B面）
                    line_center = (abs_x1 + abs_x2) // 2
                    page_center = (boundaries.left + boundaries.right) // 2
                    color = self.colors['a_vertical'] if line_center > page_center else self.colors['b_vertical']
                    
                    lines.append(DetectedLine(
                        x1=abs_x1, y1=abs_y1,
                        x2=abs_x2, y2=abs_y2,
                        confidence=confidence,
                        line_type='vertical',
                        color=color
                    ))
        
        return lines
    
    def _detect_horizontal_lines_simple(self, image: np.ndarray, boundaries: PageBoundary) -> List[DetectedLine]:
        """简化版水平线检测"""
        lines = []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 提取ROI
        roi = gray[boundaries.top:boundaries.bottom, boundaries.left:boundaries.right]
        
        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)
        
        # 霍夫变换检测直线
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                
                # 转换到原图坐标
                abs_x1 = x1 + boundaries.left
                abs_y1 = y1 + boundaries.top
                abs_x2 = x2 + boundaries.left
                abs_y2 = y2 + boundaries.top
                
                # 检查是否为水平线
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if dy < 10 and dx > 50:  # 接近水平且足够长
                    # 计算置信度
                    length = np.sqrt(dx**2 + dy**2)
                    confidence = min(length / boundaries.width, 1.0)
                    
                    # 确定颜色（A面或B面）
                    line_center = (abs_x1 + abs_x2) // 2
                    page_center = (boundaries.left + boundaries.right) // 2
                    color = self.colors['a_horizontal'] if line_center > page_center else self.colors['b_horizontal']
                    
                    lines.append(DetectedLine(
                        x1=abs_x1, y1=abs_y1,
                        x2=abs_x2, y2=abs_y2,
                        confidence=confidence,
                        line_type='horizontal',
                        color=color
                    ))
        
        return lines
    
    def _detect_spine_points_simple(self, image: np.ndarray, boundaries: PageBoundary) -> List[Tuple[int, int]]:
        """简化版书口点检测"""
        points = []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 在边界区域内每隔50像素检测一次
        step = 50
        for y in range(boundaries.top + 50, boundaries.bottom - 50, step):
            # 提取水平线区域
            line_region = gray[y-2:y+2, boundaries.left:boundaries.right]
            
            if line_region.size == 0:
                continue
            
            # 计算水平方向的梯度
            profile = np.mean(line_region, axis=0)
            gradient = np.gradient(profile)
            
            # 寻找梯度最大点
            if len(gradient) > 0:
                max_idx = np.argmax(np.abs(gradient))
                
                # 检查是否显著
                if abs(gradient[max_idx]) > np.mean(np.abs(gradient)) * 1.5:
                    x = boundaries.left + max_idx
                    points.append((x, y))
        
        return points
    
    def _calculate_corners_simple(self, boundaries: PageBoundary,
                                vertical_lines: List[DetectedLine],
                                horizontal_lines: List[DetectedLine]) -> Dict[str, List[Tuple[int, int]]]:
        """简化版角点计算"""
        left, right, top, bottom = boundaries.left, boundaries.right, boundaries.top, boundaries.bottom
        middle_x = (left + right) // 2
        
        # 简化处理：直接基于边界计算
        corners = {
            'a': [  # A面（右页）
                (middle_x, top),      # 左上
                (right, top),         # 右上
                (right, bottom),      # 右下
                (middle_x, bottom)    # 左下
            ],
            'b': [  # B面（左页）
                (left, top),          # 左上
                (middle_x, top),      # 右上
                (middle_x, bottom),   # 右下
                (left, bottom)        # 左下
            ]
        }
        
        return corners
    
    def _split_pages_simple(self, image: np.ndarray, corners: Dict[str, List[Tuple[int, int]]]) -> Tuple[np.ndarray, np.ndarray]:
        """简化版页面分割"""
        h, w = image.shape[:2]
        
        # 简单分割：直接按中缝分割
        middle_x = w // 2
        
        a_page = image[:, middle_x:]
        b_page = image[:, :middle_x]
        
        # 调整尺寸到标准
        target_height = 2598
        a_page = self._resize_to_standard(a_page, target_height)
        b_page = self._resize_to_standard(b_page, target_height)
        
        return a_page, b_page
    
    def _resize_to_standard(self, image: np.ndarray, target_height: int = 2598) -> np.ndarray:
        """调整到标准尺寸"""
        h, w = image.shape[:2]
        
        if h == 0 or w == 0:
            return image
        
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        
        # 限制最大宽度
        max_width = 1654
        if target_width > max_width:
            target_width = max_width
            target_height = int(target_width / aspect_ratio)
        
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized
    
    def _create_debug_image_enhanced(self, image: np.ndarray, boundaries: PageBoundary,
                                   vertical_lines: List[DetectedLine], horizontal_lines: List[DetectedLine],
                                   spine_points: List[Tuple[int, int]], corners: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """创建增强的调试图像（既是调试图像也是标记预览）"""
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 如果图像是RGB格式，直接使用
            debug = image.copy()
        elif len(image.shape) == 2:
            # 灰度图转换为RGB
            debug = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            debug = image.copy()
            
        h, w = debug.shape[:2]
        
        # 绘制左右两个边界框（修正：分别绘制左右两个框）
        middle_x = w // 2
        
        # 绘制整个页面边界（绿色）
        cv2.rectangle(debug, (boundaries.left, boundaries.top),
                     (boundaries.right, boundaries.bottom), self.colors['boundary'], 3)
        
        # 绘制左页面边界（青色）
        left_boundary = PageBoundary(
            left=boundaries.left,
            right=middle_x,
            top=boundaries.top,
            bottom=boundaries.bottom
        )
        cv2.rectangle(debug, (left_boundary.left, left_boundary.top),
                     (left_boundary.right, left_boundary.bottom), self.colors['b_boundary'], 2)
        
        # 绘制右页面边界（橙色）
        right_boundary = PageBoundary(
            left=middle_x,
            right=boundaries.right,
            top=boundaries.top,
            bottom=boundaries.bottom
        )
        cv2.rectangle(debug, (right_boundary.left, right_boundary.top),
                     (right_boundary.right, right_boundary.bottom), self.colors['a_boundary'], 2)
        
        # 绘制垂直线（用虚线表示）
        for line in vertical_lines:
            color = line.color if line.color else self.colors['a_vertical']
            # 绘制虚线
            dash_length = 15
            gap_length = 8
            dx = line.x2 - line.x1
            dy = line.y2 - line.y1
            distance = np.sqrt(dx*dx + dy*dy)
            steps = int(distance / (dash_length + gap_length))
            
            for i in range(steps):
                start_x = int(line.x1 + i * (dash_length + gap_length) * dx / distance)
                start_y = int(line.y1 + i * (dash_length + gap_length) * dy / distance)
                end_x = int(start_x + dash_length * dx / distance)
                end_y = int(start_y + dash_length * dy / distance)
                
                if i < steps:
                    cv2.line(debug, (start_x, start_y), (end_x, end_y), color, 2)
        
        # 绘制水平线（用虚线表示）
        for line in horizontal_lines:
            color = line.color if line.color else self.colors['a_horizontal']
            # 绘制虚线
            dash_length = 15
            gap_length = 8
            dx = line.x2 - line.x1
            dy = line.y2 - line.y1
            distance = np.sqrt(dx*dx + dy*dy)
            steps = int(distance / (dash_length + gap_length))
            
            for i in range(steps):
                start_x = int(line.x1 + i * (dash_length + gap_length) * dx / distance)
                start_y = int(line.y1 + i * (dash_length + gap_length) * dy / distance)
                end_x = int(start_x + dash_length * dx / distance)
                end_y = int(start_y + dash_length * dy / distance)
                
                if i < steps:
                    cv2.line(debug, (start_x, start_y), (end_x, end_y), color, 2)
        
        # 绘制书口点（用大圆圈表示）
        for point in spine_points:
            cv2.circle(debug, point, 10, self.colors['spine'], 3)
            cv2.circle(debug, point, 5, self.colors['spine'], -1)
        
        # 绘制中缝线（粗线）
        cv2.line(debug, (middle_x, 0), (middle_x, h), self.colors['middle'], 3)
        
        # 添加文字标注（更大的字体）
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 页面信息
        info_y = 40
        cv2.putText(debug, f"Page Size: {w}x{h}", 
                   (20, info_y), font, 0.9, (0, 0, 0), 3)
        cv2.putText(debug, f"Page Size: {w}x{h}", 
                   (20, info_y), font, 0.9, (255, 255, 255), 1)
        
        # 边界信息
        cv2.putText(debug, f"Boundary: {boundaries.width}x{boundaries.height}", 
                   (20, info_y + 40), font, 0.8, (0, 0, 0), 3)
        cv2.putText(debug, f"Boundary: {boundaries.width}x{boundaries.height}", 
                   (20, info_y + 40), font, 0.8, (255, 255, 255), 1)
        
        # 线条信息
        cv2.putText(debug, f"Vertical Lines: {len(vertical_lines)}", 
                   (20, info_y + 80), font, 0.8, (0, 0, 0), 3)
        cv2.putText(debug, f"Vertical Lines: {len(vertical_lines)}", 
                   (20, info_y + 80), font, 0.8, (255, 255, 255), 1)
        
        cv2.putText(debug, f"Horizontal Lines: {len(horizontal_lines)}", 
                   (20, info_y + 120), font, 0.8, (0, 0, 0), 3)
        cv2.putText(debug, f"Horizontal Lines: {len(horizontal_lines)}", 
                   (20, info_y + 120), font, 0.8, (255, 255, 255), 1)
        
        # 书口点信息
        cv2.putText(debug, f"Spine Points: {len(spine_points)}", 
                   (20, info_y + 160), font, 0.8, (0, 0, 0), 3)
        cv2.putText(debug, f"Spine Points: {len(spine_points)}", 
                   (20, info_y + 160), font, 0.8, (255, 255, 255), 1)
        
        # 绘制图例（修正：更大更清晰的图例）
        legend_y = h - 200
        legend_x = w - 250
        
        # 图例背景
        cv2.rectangle(debug, (legend_x, legend_y), (legend_x + 240, legend_y + 180), (250, 250, 250), -1)
        cv2.rectangle(debug, (legend_x, legend_y), (legend_x + 240, legend_y + 180), (0, 0, 0), 2)
        
        cv2.putText(debug, "LEGEND:", (legend_x + 10, legend_y + 25), font, 0.7, (0, 0, 0), 2)
        
        # 图例项
        legend_items = [
            ("Overall Boundary", self.colors['boundary'], legend_y + 50),
            ("Left Page (B)", self.colors['b_boundary'], legend_y + 75),
            ("Right Page (A)", self.colors['a_boundary'], legend_y + 100),
            ("Middle Line", self.colors['middle'], legend_y + 125),
            ("Spine Point", self.colors['spine'], legend_y + 150),
        ]
        
        for text, color, y_pos in legend_items:
            # 绘制颜色示例
            if "Boundary" in text or "Page" in text:
                cv2.rectangle(debug, (legend_x + 10, y_pos - 10), (legend_x + 30, y_pos + 5), color, 2)
            elif "Line" in text:
                cv2.line(debug, (legend_x + 10, y_pos), (legend_x + 30, y_pos), color, 3)
            elif "Point" in text:
                cv2.circle(debug, (legend_x + 20, y_pos), 8, color, 2)
                cv2.circle(debug, (legend_x + 20, y_pos), 4, color, -1)
            
            # 绘制文字
            cv2.putText(debug, text, (legend_x + 40, y_pos + 5), font, 0.5, (0, 0, 0), 1)
        
        # 在图像四角添加小标记
        corner_size = 20
        cv2.line(debug, (0, 0), (corner_size, 0), self.colors['boundary'], 3)
        cv2.line(debug, (0, 0), (0, corner_size), self.colors['boundary'], 3)
        cv2.line(debug, (w, 0), (w - corner_size, 0), self.colors['boundary'], 3)
        cv2.line(debug, (w, 0), (w, corner_size), self.colors['boundary'], 3)
        cv2.line(debug, (0, h), (corner_size, h), self.colors['boundary'], 3)
        cv2.line(debug, (0, h), (0, h - corner_size), self.colors['boundary'], 3)
        cv2.line(debug, (w, h), (w - corner_size, h), self.colors['boundary'], 3)
        cv2.line(debug, (w, h), (w, h - corner_size), self.colors['boundary'], 3)
        
        return debug
    
    def _calculate_confidence_simple(self, boundaries: PageBoundary,
                                   vertical_count: int, horizontal_count: int,
                                   spine_count: int) -> float:
        """简化版置信度计算"""
        # 边界置信度
        boundary_conf = boundaries.confidence * 0.3
        
        # 线条数量置信度
        line_conf = min((vertical_count + horizontal_count) / 10, 1.0) * 0.4
        
        # 书口点置信度
        spine_conf = min(spine_count / 5, 1.0) * 0.3
        
        total = boundary_conf + line_conf + spine_conf
        return min(total, 1.0)
    
    def _update_statistics(self, result: ProcessingResult):
        """更新统计信息"""
        self.statistics['total_pages'] += 1
        self.statistics['total_time'] += result.processing_time
        
        if result.success:
            self.statistics['successful_pages'] += 1
        
        self.statistics['avg_time_per_page'] = (
            self.statistics['total_time'] / self.statistics['total_pages']
        )
        
        self.processing_history.append({
            'timestamp': time.time(),
            'success': result.success,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'page_type': result.page_type.value,
        })
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成处理报告"""
        total = self.statistics['total_pages']
        success = self.statistics['successful_pages']
        total_time = self.statistics['total_time']
        
        report = f"""
古籍书影处理系统报告
{'=' * 60}
总处理页数: {total}
成功页数: {success}
成功率: {success/total if total>0 else 0:.2%}
总处理时间: {total_time:.2f}秒
平均每页时间: {total_time/total if total>0 else 0:.2f}秒

处理统计:
"""
        
        # 按类型统计
        type_stats = {}
        for entry in self.processing_history:
            page_type = entry['page_type']
            if page_type not in type_stats:
                type_stats[page_type] = {'total': 0, 'success': 0}
            type_stats[page_type]['total'] += 1
            if entry['success']:
                type_stats[page_type]['success'] += 1
        
        for page_type, stats in type_stats.items():
            success_rate = stats['success'] / max(stats['total'], 1)
            report += f"  {page_type}: {stats['success']}/{stats['total']} ({success_rate:.2%})\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

# ============================================================================
# 图形界面类
# ============================================================================

class AncientBookGUI:
    """古籍处理系统图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("古籍书影智能处理系统 v4.2")
        self.root.geometry("1400x850")
        
        # 设置字体
        try:
            self.root.option_add("*Font", "微软雅黑 10")
        except:
            pass
        
        # 处理器实例
        self.processor = AncientBookProcessor()
        
        # 处理队列
        self.task_queue = queue.Queue()
        self.processing = False
        
        # 当前显示的图像
        self.current_image_tk = None
        self.debug_image_tk = None
        
        # 当前图像文件列表和索引
        self.image_files = []
        self.current_image_index = -1
        
        # 创建界面
        self._create_widgets()
        
        # 开始处理线程
        self._start_processing_thread()
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========== 控制面板内容 ==========
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(control_frame, text="文件选择", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 图像文件
        ttk.Label(file_frame, text="图像文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.image_path_var = tk.StringVar()
        image_entry = ttk.Entry(file_frame, textvariable=self.image_path_var, width=30)
        image_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_image, width=10).grid(row=0, column=2, pady=2)
        
        # 绑定回车键
        image_entry.bind('<Return>', lambda e: self._load_selected_image())
        
        # 文件夹
        ttk.Label(file_frame, text="图像文件夹:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.folder_path_var = tk.StringVar()
        folder_entry = ttk.Entry(file_frame, textvariable=self.folder_path_var, width=30)
        folder_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_folder, width=10).grid(row=1, column=2, pady=2)
        
        # PDF文件
        ttk.Label(file_frame, text="PDF文件:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pdf_path_var = tk.StringVar()
        pdf_entry = ttk.Entry(file_frame, textvariable=self.pdf_path_var, width=30)
        pdf_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_pdf, width=10).grid(row=2, column=2, pady=2)
        
        # 输出目录
        ttk.Label(file_frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.output_dir_var = tk.StringVar(value="out")  # 改为out文件夹
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=30).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_output, width=10).grid(row=3, column=2, pady=2)
        
        # 图像导航区域（新增）
        nav_frame = ttk.LabelFrame(control_frame, text="图像导航", padding="5")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_btn_frame = ttk.Frame(nav_frame)
        nav_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_btn_frame, text="上一张", command=self._prev_image, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_btn_frame, text="下一张", command=self._next_image, width=10).pack(side=tk.LEFT, padx=2)
        
        self.nav_label = ttk.Label(nav_frame, text="0 / 0")
        self.nav_label.pack(pady=2)
        
        # 处理选项区域
        options_frame = ttk.LabelFrame(control_frame, text="处理选项", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理模式
        ttk.Label(options_frame, text="处理模式:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="标准")
        modes = ["标准", "快速", "精确"]
        mode_combo = ttk.Combobox(options_frame, textvariable=self.mode_var, values=modes, width=20, state="readonly")
        mode_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 调试选项
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="生成调试图像", variable=self.debug_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 处理按钮区域
        button_frame = ttk.LabelFrame(control_frame, text="处理控制", padding="10")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 按钮行1
        btn_frame1 = ttk.Frame(button_frame)
        btn_frame1.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame1, text="处理图像", command=self._process_image, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame1, text="处理PDF", command=self._process_pdf, width=15).pack(side=tk.LEFT, padx=2)
        
        # 按钮行2
        btn_frame2 = ttk.Frame(button_frame)
        btn_frame2.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame2, text="批量处理", command=self._batch_process, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="停止处理", command=self._stop_processing, width=15).pack(side=tk.LEFT, padx=2)
        
        # 进度区域
        progress_frame = ttk.LabelFrame(control_frame, text="处理进度", padding="5")
        progress_frame.pack(fill=tk.X)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 进度标签
        self.progress_label = ttk.Label(progress_frame, text="就绪")
        self.progress_label.pack(pady=2)
        
        # 日志区域
        log_frame = ttk.LabelFrame(control_frame, text="处理日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=40, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 日志控制按钮
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_btn_frame, text="清空日志", command=self._clear_log, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_btn_frame, text="保存日志", command=self._save_log, width=10).pack(side=tk.LEFT, padx=2)
        
        # ========== 显示区域内容 ==========
        
        # 创建标签页 - 只保留原始图像和调试图像
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像标签页
        original_frame = ttk.Frame(self.notebook)
        self.notebook.add(original_frame, text="原始图像")
        
        self.original_canvas = tk.Canvas(original_frame, bg='#f0f0f0', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 调试图像标签页（也就是标记预览）
        debug_frame = ttk.Frame(self.notebook)
        self.notebook.add(debug_frame, text="调试图像")
        
        self.debug_canvas = tk.Canvas(debug_frame, bg='#f0f0f0', highlightthickness=0)
        self.debug_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 信息标签页
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="处理信息")
        
        self.info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _browse_image(self):
        """浏览图像文件"""
        filetypes = [
            ('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('所有文件', '*.*')
        ]
        filename = filedialog.askopenfilename(title="选择古籍图像", filetypes=filetypes)
        if filename:
            self.image_path_var.set(filename)
            self._load_selected_image()
    
    def _browse_folder(self):
        """浏览文件夹"""
        directory = filedialog.askdirectory(title="选择图像文件夹")
        if directory:
            self.folder_path_var.set(directory)
            self._load_folder_images(directory)
    
    def _load_folder_images(self, directory):
        """加载文件夹中的图像文件"""
        try:
            # 支持的图像格式
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            # 查找图像文件
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(Path(directory).glob(f'*{ext}'))
                self.image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
            
            # 按文件名排序
            self.image_files.sort()
            
            if not self.image_files:
                messagebox.showwarning("警告", "所选文件夹中没有找到图像文件")
                return
            
            # 显示第一张图像
            self.current_image_index = 0
            self._load_image_from_list()
            
            self._log(f"已加载文件夹: {directory}")
            self._log(f"找到 {len(self.image_files)} 个图像文件")
            
        except Exception as e:
            self._log(f"加载文件夹失败: {str(e)}")
    
    def _load_image_from_list(self):
        """从文件列表中加载当前索引的图像"""
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = str(self.image_files[self.current_image_index])
            self.image_path_var.set(image_path)
            self._load_and_display_image(image_path, self.original_canvas)
            self._update_navigation_label()
        else:
            self.current_image_index = -1
    
    def _prev_image(self):
        """显示上一张图像"""
        if len(self.image_files) > 0:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self._load_image_from_list()
    
    def _next_image(self):
        """显示下一张图像"""
        if len(self.image_files) > 0:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self._load_image_from_list()
    
    def _update_navigation_label(self):
        """更新导航标签"""
        if len(self.image_files) > 0:
            self.nav_label.config(text=f"{self.current_image_index + 1} / {len(self.image_files)}")
        else:
            self.nav_label.config(text="0 / 0")
    
    def _load_selected_image(self):
        """加载选择的图像"""
        image_path = self.image_path_var.get()
        if image_path and os.path.exists(image_path):
            self._load_and_display_image(image_path, self.original_canvas)
    
    def _browse_pdf(self):
        """浏览PDF文件"""
        filetypes = [
            ('PDF文件', '*.pdf'),
            ('所有文件', '*.*')
        ]
        filename = filedialog.askopenfilename(title="选择PDF文件", filetypes=filetypes)
        if filename:
            self.pdf_path_var.set(filename)
    
    def _browse_output(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir_var.set(directory)
    
    def _load_and_display_image(self, image_path, canvas):
        """加载并显示图像到指定画布"""
        try:
            # 使用PIL加载图像，确保颜色正确
            pil_image = Image.open(image_path)
            
            # 获取画布尺寸
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 400
            
            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图像大小
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可显示的图像
            image_tk = ImageTk.PhotoImage(pil_image)
            
            # 保存引用，避免被垃圾回收
            if canvas == self.original_canvas:
                self.current_image_tk = image_tk
            elif canvas == self.debug_canvas:
                self.debug_image_tk = image_tk
            
            # 显示在画布上
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=image_tk, anchor=tk.CENTER
            )
            
            # 记录日志
            if canvas == self.original_canvas:
                self._log(f"已加载图像: {os.path.basename(image_path)}")
                self._log(f"原始尺寸: {img_width} x {img_height}")
            
        except Exception as e:
            self._log(f"加载图像失败: {str(e)}")
    
    def _process_image(self):
        """处理单张图像"""
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showwarning("警告", "请选择有效的图像文件")
            return
        
        output_dir = self.output_dir_var.get()
        
        # 添加到处理队列
        self.task_queue.put(('image', image_path, output_dir))
        self._log(f"已添加到处理队列: {os.path.basename(image_path)}")
    
    def _process_pdf(self):
        """处理PDF文件"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path or not os.path.exists(pdf_path):
            messagebox.showwarning("警告", "请选择有效的PDF文件")
            return
        
        output_dir = self.output_dir_var.get()
        
        # 询问处理页数
        from tkinter import simpledialog
        max_pages = simpledialog.askinteger(
            "处理页数",
            "请输入最大处理页数 (0表示全部):",
            parent=self.root,
            initialvalue=0,
            minvalue=0,
            maxvalue=1000
        )
        
        if max_pages is None:  # 用户取消
            return
        
        # 添加到处理队列
        self.task_queue.put(('pdf', pdf_path, output_dir, max_pages))
        self._log(f"已添加到处理队列: {os.path.basename(pdf_path)} (最多{max_pages}页)")
    
    def _batch_process(self):
        """批量处理文件夹中的图像"""
        if len(self.image_files) == 0:
            directory = filedialog.askdirectory(title="选择包含古籍图像的文件夹")
            if not directory:
                return
            self._load_folder_images(directory)
        
        if len(self.image_files) == 0:
            messagebox.showwarning("警告", "没有找到图像文件")
            return
        
        output_dir = self.output_dir_var.get()
        
        # 确认
        if not messagebox.askyesno("确认", f"找到 {len(self.image_files)} 个图像文件，是否开始批量处理？"):
            return
        
        # 添加到处理队列
        for image_file in self.image_files:
            self.task_queue.put(('image', str(image_file), output_dir))
        
        self._log(f"已添加 {len(self.image_files)} 个文件到处理队列")
    
    def _stop_processing(self):
        """停止处理"""
        self.processing = False
        self._log("正在停止处理...")
    
    def _clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def _save_log(self):
        """保存日志"""
        filename = filedialog.asksaveasfilename(
            title="保存日志",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    log_content = self.log_text.get(1.0, tk.END)
                    f.write(log_content)
                self._log(f"日志已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {str(e)}")
    
    def _log(self, message):
        """记录日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # 在文本框中显示
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # 调试图像信息不输出到控制台
        if "调试图像" not in message and "保存到" not in message:
            print(log_message, end='')
    
    def _update_progress(self, value, message):
        """更新进度"""
        self.progress_var.set(value)
        self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def _display_result(self, result, original_image_path=None):
        """显示处理结果"""
        # 更新信息文本
        info_text = f"【处理结果】\n"
        info_text += f"状态: {'✓ 成功' if result.success else '✗ 失败'}\n"
        info_text += f"古籍类型: {result.page_type.value}\n"
        info_text += f"置信度: {result.confidence:.2%}\n"
        info_text += f"处理时间: {result.processing_time:.2f}秒\n"
        info_text += f"边界检测: {result.boundaries.left}, {result.boundaries.top} - {result.boundaries.right}, {result.boundaries.bottom}\n"
        info_text += f"边界尺寸: {result.boundaries.width} x {result.boundaries.height}\n"
        info_text += f"垂直线: {len(result.vertical_lines)}条\n"
        info_text += f"水平线: {len(result.horizontal_lines)}条\n"
        info_text += f"书口点: {len(result.spine_points)}个\n"
        
        if result.warnings:
            info_text += f"\n【警告信息】\n"
            for warning in result.warnings:
                info_text += f"• {warning}\n"
        
        if result.debug_info:
            info_text += f"\n【调试信息】\n"
            for key, value in result.debug_info.items():
                if key not in ['output_paths', 'debug_image', 'error']:
                    info_text += f"{key}: {value}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
        # 显示原始图像（如果有）
        if original_image_path:
            self._load_and_display_image(original_image_path, self.original_canvas)
        
        # 显示调试图像（标记预览）
        debug_image_path = result.debug_info.get('debug_image')
        if debug_image_path and os.path.exists(debug_image_path):
            self._load_and_display_image(debug_image_path, self.debug_canvas)
            # 切换到调试图像标签页
            self.notebook.select(1)  # 索引1是"调试图像"标签页
    
    def _start_processing_thread(self):
        """启动处理线程"""
        def processing_worker():
            self.processing = True
            
            while self.processing:
                try:
                    # 从队列获取任务
                    task = self.task_queue.get(timeout=0.1)
                    
                    task_type = task[0]
                    
                    if task_type == 'image':
                        image_path, output_dir = task[1], task[2]
                        self._process_single_image(image_path, output_dir)
                    
                    elif task_type == 'pdf':
                        pdf_path, output_dir, max_pages = task[1], task[2], task[3]
                        self._process_pdf_file(pdf_path, output_dir, max_pages)
                    
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self._log(f"处理线程错误: {str(e)}")
        
        # 启动线程
        thread = threading.Thread(target=processing_worker, daemon=True)
        thread.start()
    
    def _process_single_image(self, image_path, output_dir):
        """处理单张图像（在工作线程中）"""
        try:
            self._log(f"开始处理: {os.path.basename(image_path)}")
            self._update_progress(10, "正在预处理...")
            
            # 处理图像
            result = self.processor.process_image(image_path, output_dir)
            
            self._update_progress(100, "处理完成")
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self._display_result(result, image_path))
            
            if result.success:
                self._log(f"✓ 处理成功: {os.path.basename(image_path)}")
                self._log(f"  置信度: {result.confidence:.2%}")
                self._log(f"  处理时间: {result.processing_time:.2f}秒")
                
                output_paths = result.debug_info.get('output_paths', {})
                if output_paths.get('a'):
                    self._log(f"  左页面保存到: {os.path.basename(output_paths['a'])}")
                if output_paths.get('b'):
                    self._log(f"  右页面保存到: {os.path.basename(output_paths['b'])}")
                
                # 调试图像信息只显示在GUI中，不输出到控制台
                debug_image_path = result.debug_info.get('debug_image')
                if debug_image_path and os.path.exists(debug_image_path):
                    self._log(f"  调试图像已生成")
            else:
                self._log(f"✗ 处理失败: {os.path.basename(image_path)}")
                for warning in result.warnings:
                    self._log(f"  警告: {warning}")
            
        except Exception as e:
            self._log(f"处理失败: {str(e)}")
            self._update_progress(0, "处理失败")
    
    def _process_pdf_file(self, pdf_path, output_dir, max_pages):
        """处理PDF文件（简化版，实际需要PyMuPDF）"""
        try:
            self._log(f"开始处理PDF: {os.path.basename(pdf_path)}")
            self._log("注意: PDF处理需要PyMuPDF库，这里只做演示")
            
            # 这里简化处理，只处理第一页
            result = self.processor.process_image(pdf_path, output_dir)
            result.debug_info['page_number'] = 1
            
            self._log(f"PDF处理完成 (演示模式)")
            
            # 显示结果
            self.root.after(0, lambda: self._display_result(result))
            
        except Exception as e:
            self._log(f"PDF处理失败: {str(e)}")
    
    def _on_closing(self):
        """窗口关闭事件"""
        self.processing = False
        self.root.destroy()
    
    def run(self):
        """运行主循环"""
        # 居中显示窗口
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        self.root.mainloop()

# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("古籍书影智能处理系统 v4.2")
    print("=" * 60)
    
    try:
        # 检查OpenCV
        print(f"OpenCV版本: {cv2.__version__}")
        
        # 创建并运行GUI
        app = AncientBookGUI()
        app.run()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请安装必要的依赖库:")
        print("pip install opencv-python pillow numpy")
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()                   
