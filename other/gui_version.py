"""
古籍书影智能处理系统 v4.0 - 完整图形界面版
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
from PIL import Image, ImageTk

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
        # 颜色定义
        self.colors = {
            'boundary': (0, 255, 0),        # 绿色 - 边界
            'a_vertical': (255, 0, 0),      # 蓝色 - A面垂直线
            'b_vertical': (0, 0, 255),      # 红色 - B面垂直线
            'spine': (255, 255, 0),         # 黄色 - 书口
            'middle': (0, 255, 255),        # 青色 - 中缝
            'a_horizontal': (255, 0, 255),  # 紫色 - A面水平线
            'b_horizontal': (255, 165, 0),  # 橙色 - B面水平线
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
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_h, original_w = image.shape[:2]
            
            # 简化处理流程
            # 1. 识别古籍类型
            book_type = self._identify_book_type_simple(image)
            
            # 2. 预处理
            processed = self._simple_preprocess(image, book_type)
            
            # 3. 检测边界
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
            
            cv2.imwrite(a_path, a_page)
            cv2.imwrite(b_path, b_page)
            
            # 9. 创建调试图像
            debug_path = str(Path(output_dir) / f"{base_name}_debug.jpg")
            debug_image = self._create_debug_image_simple(processed, boundaries, 
                                                         vertical_lines, horizontal_lines,
                                                         spine_points, corners)
            cv2.imwrite(debug_path, debug_image)
            
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
                    'book_type': book_type.value
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
        """简化版预处理"""
        processed = image.copy()
        h, w = processed.shape[:2]
        
        # 调整尺寸
        max_size = self.config['preprocessing']['max_size']
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 增强对比度
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def _detect_boundaries_simple(self, image: np.ndarray) -> PageBoundary:
        """简化版边界检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 使用边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
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
            
            # 稍微扩大边界
            margin = 10
            return PageBoundary(
                left=max(0, x - margin),
                right=min(w, x + w_rect + margin),
                top=max(0, y - margin),
                bottom=min(h, y + h_rect + margin),
                confidence=min(confidence, 1.0)
            )
        
        # 默认使用整个图像
        return PageBoundary(0, w, 0, h, confidence=0.5)
    
    def _detect_vertical_lines_simple(self, image: np.ndarray, boundaries: PageBoundary) -> List[DetectedLine]:
        """简化版垂直线检测"""
        lines = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
    
    def _create_debug_image_simple(self, image: np.ndarray, boundaries: PageBoundary,
                                 vertical_lines: List[DetectedLine], horizontal_lines: List[DetectedLine],
                                 spine_points: List[Tuple[int, int]], corners: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """创建调试图像"""
        debug = image.copy()
        h, w = debug.shape[:2]
        
        # 绘制边界
        cv2.rectangle(debug, (boundaries.left, boundaries.top),
                     (boundaries.right, boundaries.bottom), self.colors['boundary'], 3)
        
        # 绘制垂直线
        for line in vertical_lines:
            color = line.color if line.color else self.colors['a_vertical']
            cv2.line(debug, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 绘制水平线
        for line in horizontal_lines:
            color = line.color if line.color else self.colors['a_horizontal']
            cv2.line(debug, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 绘制书口点
        for point in spine_points:
            cv2.circle(debug, point, 8, self.colors['spine'], -1)
        
        # 绘制中缝
        middle_x = w // 2
        cv2.line(debug, (middle_x, 0), (middle_x, h), self.colors['middle'], 2)
        
        # 添加信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        info = f"Size: {w}x{h} | V-lines: {len(vertical_lines)} | H-lines: {len(horizontal_lines)}"
        cv2.putText(debug, info, (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug, info, (10, 30), font, 0.7, (0, 0, 0), 1)
        
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
        self.root.title("古籍书影智能处理系统 v4.0")
        self.root.geometry("1200x800")
        
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
        ttk.Entry(file_frame, textvariable=self.image_path_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_image, width=10).grid(row=0, column=2, pady=2)
        
        # PDF文件
        ttk.Label(file_frame, text="PDF文件:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pdf_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.pdf_path_var, width=30).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_pdf, width=10).grid(row=1, column=2, pady=2)
        
        # 输出目录
        ttk.Label(file_frame, text="输出目录:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=30).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览...", command=self._browse_output, width=10).grid(row=2, column=2, pady=2)
        
        # 处理选项区域
        options_frame = ttk.LabelFrame(control_frame, text="处理选项", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 处理模式
        ttk.Label(options_frame, text="处理模式:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="标准")
        modes = ["标准", "快速", "精确"]
        ttk.Combobox(options_frame, textvariable=self.mode_var, values=modes, width=20, state="readonly").grid(row=0, column=1, padx=5, pady=5)
        
        # 调试选项
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="生成调试图像", variable=self.debug_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        self.intermediate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="保存中间结果", variable=self.intermediate_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
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
        
        # 创建标签页
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像标签页
        original_frame = ttk.Frame(self.notebook)
        self.notebook.add(original_frame, text="原始图像")
        
        self.original_canvas = tk.Canvas(original_frame, bg='#f0f0f0', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理结果标签页
        result_frame = ttk.Frame(self.notebook)
        self.notebook.add(result_frame, text="处理结果")
        
        self.result_canvas = tk.Canvas(result_frame, bg='#f0f0f0', highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 调试图像标签页
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
            ('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('所有文件', '*.*')
        ]
        filename = filedialog.askopenfilename(title="选择古籍图像", filetypes=filetypes)
        if filename:
            self.image_path_var.set(filename)
            self._load_and_display_image(filename, self.original_canvas)
    
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
            # 使用PIL加载图像
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
        directory = filedialog.askdirectory(title="选择包含古籍图像的文件夹")
        if not directory:
            return
        
        output_dir = self.output_dir_var.get()
        
        # 查找图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        if not image_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到图像文件")
            return
        
        # 确认
        if not messagebox.askyesno("确认", f"找到 {len(image_files)} 个图像文件，是否开始批量处理？"):
            return
        
        # 添加到处理队列
        for image_file in image_files:
            self.task_queue.put(('image', str(image_file), output_dir))
        
        self._log(f"已添加 {len(image_files)} 个文件到处理队列")
    
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
        
        # 同时打印到控制台
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
        
        # 显示调试图像（如果有）
        debug_image_path = result.debug_info.get('debug_image')
        if debug_image_path and os.path.exists(debug_image_path):
            self._load_and_display_image(debug_image_path, self.debug_canvas)
        
        # 显示结果图像
        output_paths = result.debug_info.get('output_paths', {})
        if output_paths.get('a') and os.path.exists(output_paths['a']):
            self._load_and_display_image(output_paths['a'], self.result_canvas)
    
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
                    self._log(f"  输出文件A: {os.path.basename(output_paths['a'])}")
                if output_paths.get('b'):
                    self._log(f"  输出文件B: {os.path.basename(output_paths['b'])}")
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
    print("古籍书影智能处理系统 v4.0")
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
