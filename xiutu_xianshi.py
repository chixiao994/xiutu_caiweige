"""
古籍书影智能处理系统 v4.0 - 文件夹批量处理版
支持文件夹批量处理，保持原图色调，实时显示标记
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

# ============================================================================
# 核心处理器类
# ============================================================================

class BookType(Enum):
    """古籍类型枚举"""
    STANDARD = "标准刻本"
    COLOR_PRINT = "套红朱墨本"
    RUBBING = "拓本"
    ANNOTATED = "批校本"
    MAP_STYLE = "地图式古籍"
    DAMAGED = "损伤古籍"

@dataclass
class DetectedLine:
    """检测到的线条"""
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    line_type: str
    side: str  # 'a' 或 'b'

@dataclass
class PageBoundary:
    """页面边界"""
    left: int; right: int; top: int; bottom: int
    confidence: float = 1.0

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
    
    def __init__(self):
        # 标记颜色定义（仅用于界面显示）
        self.colors = {
            'boundary': (0, 255, 0),        # 绿色 - 边界
            'a_vertical': (255, 0, 0),      # 蓝色 - A面垂直线
            'b_vertical': (0, 0, 255),      # 红色 - B面垂直线
            'spine': (255, 255, 0),         # 黄色 - 书口
            'middle': (0, 255, 255),        # 青色 - 中缝
            'a_horizontal': (255, 0, 255),  # 紫色 - A面水平线
            'b_horizontal': (255, 165, 0),  # 橙色 - B面水平线
        }
        
        self.processing_history = []
    
    def process_image_with_markup(self, image_path: str) -> Tuple[Optional[np.ndarray], ProcessingResult]:
        """处理图像并生成标记图（仅用于显示，不保存）"""
        start_time = time.time()
        
        try:
            # 1. 读取图像（保持原色）
            image = cv2.imread(image_path)
            if image is None:
                return None, ProcessingResult(
                    success=False, page_type=BookType.STANDARD,
                    boundaries=PageBoundary(0,0,0,0), vertical_lines=[],
                    horizontal_lines=[], spine_points=[], corners={'a':[],'b':[]},
                    processing_time=0, confidence=0, warnings=["无法读取图像"],
                    debug_info={}
                )
            
            original = image.copy()
            h, w = image.shape[:2]
            
            # 2. 灰度处理用于分析（不影响原图）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 3. 检测边界
            boundaries = self._detect_boundaries(gray)
            
            # 4. 检测线条
            vertical_lines = self._detect_lines(gray, boundaries, 'vertical')
            horizontal_lines = self._detect_lines(gray, boundaries, 'horizontal')
            
            # 5. 检测书口点
            spine_points = self._detect_spine_points(gray, boundaries, vertical_lines)
            
            # 6. 计算角点
            corners = self._calculate_corners(boundaries, vertical_lines, horizontal_lines)
            
            # 7. 在原始图像上绘制标记（不影响原图颜色）
            marked_image = original.copy()
            self._draw_markup(marked_image, boundaries, vertical_lines, 
                             horizontal_lines, spine_points, corners)
            
            # 8. 创建处理结果
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(boundaries, len(vertical_lines), 
                                                   len(horizontal_lines), len(spine_points))
            
            result = ProcessingResult(
                success=True,
                page_type=BookType.STANDARD,
                boundaries=boundaries,
                vertical_lines=vertical_lines,
                horizontal_lines=horizontal_lines,
                spine_points=spine_points,
                corners=corners,
                processing_time=processing_time,
                confidence=confidence,
                warnings=[],
                debug_info={
                    'original_size': (w, h),
                    'image_path': image_path,
                    'processing_time': processing_time
                }
            )
            
            return marked_image, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return None, ProcessingResult(
                success=False, page_type=BookType.STANDARD,
                boundaries=PageBoundary(0,0,0,0), vertical_lines=[],
                horizontal_lines=[], spine_points=[], corners={'a':[],'b':[]},
                processing_time=processing_time, confidence=0,
                warnings=[f"处理失败: {str(e)}"], debug_info={}
            )
    
    def process_and_split_image(self, image_path: str, output_dir: str) -> Tuple[Optional[Tuple[str, str]], ProcessingResult]:
        """处理并分割图像（保存分割结果）"""
        start_time = time.time()
        
        try:
            # 1. 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None, ProcessingResult(
                    success=False, page_type=BookType.STANDARD,
                    boundaries=PageBoundary(0,0,0,0), vertical_lines=[],
                    horizontal_lines=[], spine_points=[], corners={'a':[],'b':[]},
                    processing_time=0, confidence=0, warnings=["无法读取图像"],
                    debug_info={}
                )
            
            h, w = image.shape[:2]
            
            # 2. 灰度处理用于分析
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 3. 检测边界
            boundaries = self._detect_boundaries(gray)
            
            # 4. 检测线条
            vertical_lines = self._detect_lines(gray, boundaries, 'vertical')
            
            # 5. 计算分割位置
            if vertical_lines:
                # 使用检测到的垂直线确定分割位置
                a_lines = [l for l in vertical_lines if l.side == 'a']
                b_lines = [l for l in vertical_lines if l.side == 'b']
                
                if a_lines and b_lines:
                    # 取A面最左线和B面最右线
                    a_left = min([l.x1 for l in a_lines])
                    b_right = max([l.x2 for l in b_lines])
                    split_x = (a_left + b_right) // 2
                else:
                    split_x = w // 2
            else:
                split_x = w // 2
            
            # 6. 分割图像
            a_page = image[:, split_x:]
            b_page = image[:, :split_x]
            
            # 7. 保存结果
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            a_path = str(Path(output_dir) / f"{base_name}_A面.jpg")
            b_path = str(Path(output_dir) / f"{base_name}_B面.jpg")
            
            # 保持原图质量保存
            cv2.imwrite(a_path, a_page, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(b_path, b_page, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 8. 创建处理结果
            processing_time = time.time() - start_time
            confidence = 0.8 if vertical_lines else 0.5
            
            result = ProcessingResult(
                success=True,
                page_type=BookType.STANDARD,
                boundaries=boundaries,
                vertical_lines=vertical_lines,
                horizontal_lines=[],
                spine_points=[],
                corners={'a': [], 'b': []},
                processing_time=processing_time,
                confidence=confidence,
                warnings=[],
                debug_info={
                    'original_size': (w, h),
                    'split_position': split_x,
                    'output_paths': (a_path, b_path)
                }
            )
            
            return (a_path, b_path), result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return None, ProcessingResult(
                success=False, page_type=BookType.STANDARD,
                boundaries=PageBoundary(0,0,0,0), vertical_lines=[],
                horizontal_lines=[], spine_points=[], corners={'a':[],'b':[]},
                processing_time=processing_time, confidence=0,
                warnings=[f"分割失败: {str(e)}"], debug_info={}
            )
    
    def _detect_boundaries(self, gray: np.ndarray) -> PageBoundary:
        """检测页面边界"""
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
            
            return PageBoundary(
                left=x, right=x+w_rect,
                top=y, bottom=y+h_rect,
                confidence=min(confidence, 1.0)
            )
        
        # 默认使用整个图像
        return PageBoundary(0, w, 0, h, confidence=0.3)
    
    def _detect_lines(self, gray: np.ndarray, boundaries: PageBoundary, line_type: str) -> List[DetectedLine]:
        """检测线条"""
        lines = []
        
        # 提取ROI
        roi = gray[boundaries.top:boundaries.bottom, boundaries.left:boundaries.right]
        h_roi, w_roi = roi.shape
        
        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)
        
        # 霍夫变换检测直线
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min(h_roi, w_roi)//10,
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
                
                # 检查线型
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if line_type == 'vertical' and dx < 10 and dy > h_roi//4:
                    # 垂直线
                    line_center = (abs_x1 + abs_x2) // 2
                    page_center = (boundaries.left + boundaries.right) // 2
                    side = 'a' if line_center > page_center else 'b'
                    
                    lines.append(DetectedLine(
                        x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2,
                        confidence=min(dy / boundaries.height, 1.0),
                        line_type='vertical',
                        side=side
                    ))
                
                elif line_type == 'horizontal' and dy < 10 and dx > w_roi//4:
                    # 水平线
                    line_center = (abs_y1 + abs_y2) // 2
                    page_center = (boundaries.top + boundaries.bottom) // 2
                    side = 'a' if line_center > page_center else 'b'
                    
                    lines.append(DetectedLine(
                        x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2,
                        confidence=min(dx / boundaries.width, 1.0),
                        line_type='horizontal',
                        side=side
                    ))
        
        return lines
    
    def _detect_spine_points(self, gray: np.ndarray, boundaries: PageBoundary, 
                            vertical_lines: List[DetectedLine]) -> List[Tuple[int, int]]:
        """检测书口点"""
        points = []
        
        # 在边界区域内检测
        step = 50
        for y in range(boundaries.top + 50, boundaries.bottom - 50, step):
            line_region = gray[y-2:y+2, boundaries.left:boundaries.right]
            
            if line_region.size == 0:
                continue
            
            profile = np.mean(line_region, axis=0)
            gradient = np.gradient(profile)
            
            if len(gradient) > 0:
                max_idx = np.argmax(np.abs(gradient))
                if abs(gradient[max_idx]) > np.mean(np.abs(gradient)) * 1.5:
                    x = boundaries.left + max_idx
                    points.append((x, y))
        
        return points
    
    def _calculate_corners(self, boundaries: PageBoundary,
                         vertical_lines: List[DetectedLine],
                         horizontal_lines: List[DetectedLine]) -> Dict[str, List[Tuple[int, int]]]:
        """计算角点"""
        left, right, top, bottom = boundaries.left, boundaries.right, boundaries.top, boundaries.bottom
        middle_x = (left + right) // 2
        
        # 简化计算
        return {
            'a': [  # A面
                (middle_x, top),
                (right, top),
                (right, bottom),
                (middle_x, bottom)
            ],
            'b': [  # B面
                (left, top),
                (middle_x, top),
                (middle_x, bottom),
                (left, bottom)
            ]
        }
    
    def _draw_markup(self, image: np.ndarray, boundaries: PageBoundary,
                    vertical_lines: List[DetectedLine], horizontal_lines: List[DetectedLine],
                    spine_points: List[Tuple[int, int]], corners: Dict[str, List[Tuple[int, int]]]):
        """在图像上绘制标记"""
        h, w = image.shape[:2]
        
        # 1. 绘制边界
        cv2.rectangle(image, (boundaries.left, boundaries.top),
                     (boundaries.right, boundaries.bottom), self.colors['boundary'], 2)
        
        # 2. 绘制垂直线
        for line in vertical_lines:
            color = self.colors['a_vertical'] if line.side == 'a' else self.colors['b_vertical']
            cv2.line(image, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 3. 绘制水平线
        for line in horizontal_lines:
            color = self.colors['a_horizontal'] if line.side == 'a' else self.colors['b_horizontal']
            cv2.line(image, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 4. 绘制书口点
        for point in spine_points:
            cv2.circle(image, point, 6, self.colors['spine'], -1)
        
        # 5. 绘制中缝
        middle_x = w // 2
        cv2.line(image, (middle_x, 0), (middle_x, h), self.colors['middle'], 2)
        
        # 6. 添加信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        info = f"尺寸: {w}×{h} | 垂直线: {len(vertical_lines)} | 水平线: {len(horizontal_lines)}"
        cv2.putText(image, info, (10, 30), font, 0.7, (0, 0, 0), 3)
        cv2.putText(image, info, (10, 30), font, 0.7, (255, 255, 255), 1)
    
    def _calculate_confidence(self, boundaries: PageBoundary,
                            vertical_count: int, horizontal_count: int,
                            spine_count: int) -> float:
        """计算置信度"""
        conf1 = boundaries.confidence * 0.4
        conf2 = min((vertical_count + horizontal_count) / 10, 1.0) * 0.4
        conf3 = min(spine_count / 5, 1.0) * 0.2
        return min(conf1 + conf2 + conf3, 1.0)

# ============================================================================
# 图形界面类 - 支持文件夹批量处理
# ============================================================================

class AncientBookGUI:
    """古籍处理系统图形界面 - 支持文件夹批量处理"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("古籍书影智能处理系统 v4.0 - 文件夹批量处理版")
        self.root.geometry("1300x800")
        
        # 设置字体
        try:
            self.root.option_add("*Font", "微软雅黑 10")
        except:
            pass
        
        # 处理器实例
        self.processor = AncientBookProcessor()
        
        # 处理队列和状态
        self.task_queue = queue.Queue()
        self.processing = False
        self.current_folder = None
        self.image_files = []
        self.current_index = 0
        
        # 图像显示相关
        self.original_image_tk = None
        self.marked_image_tk = None
        self.result_images_tk = []  # 保存结果图像引用
        
        # 创建界面
        self._create_widgets()
        
        # 开始处理线程
        self._start_processing_thread()
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 绑定键盘事件
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 上部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 下部显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # ========== 控制面板内容 ==========
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(control_frame, text="文件选择", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件夹选择
        ttk.Label(file_frame, text="图片文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.folder_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.folder_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览文件夹...", command=self._browse_folder, width=15).grid(row=0, column=2, pady=5)
        
        # 输出目录
        ttk.Label(file_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self._browse_output, width=15).grid(row=1, column=2, pady=5)
        
        # 处理选项区域
        options_frame = ttk.LabelFrame(control_frame, text="处理选项", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 列1：基本选项
        col1 = ttk.Frame(options_frame)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1, text="自动检测边界", variable=self.auto_detect_var).pack(anchor=tk.W, pady=2)
        
        self.detect_lines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1, text="检测框线", variable=self.detect_lines_var).pack(anchor=tk.W, pady=2)
        
        self.detect_spine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1, text="检测书口", variable=self.detect_spine_var).pack(anchor=tk.W, pady=2)
        
        # 列2：分割选项
        col2 = ttk.Frame(options_frame)
        col2.pack(side=tk.LEFT, fill=tk.Y)
        
        self.auto_split_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col2, text="自动分割页面", variable=self.auto_split_var).pack(anchor=tk.W, pady=2)
        
        ttk.Label(col2, text="分割方式:").pack(anchor=tk.W, pady=2)
        self.split_method_var = tk.StringVar(value="auto")
        ttk.Radiobutton(col2, text="自动检测", variable=self.split_method_var, value="auto").pack(anchor=tk.W)
        ttk.Radiobutton(col2, text="固定中缝", variable=self.split_method_var, value="fixed").pack(anchor=tk.W)
        
        # 处理控制区域
        process_frame = ttk.LabelFrame(control_frame, text="处理控制", padding="10")
        process_frame.pack(fill=tk.X)
        
        # 按钮行1：导航控制
        nav_frame = ttk.Frame(process_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◀ 上一张", command=self._prev_image, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="▶ 下一张", command=self._next_image, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="定位中缝", command=self._align_spine, width=12).pack(side=tk.LEFT, padx=2)
        
        # 按钮行2：处理控制
        proc_frame = ttk.Frame(process_frame)
        proc_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(proc_frame, text="处理当前图片", command=self._process_current, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(proc_frame, text="处理全部图片", command=self._process_all, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(proc_frame, text="批量分割保存", command=self._batch_split, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(proc_frame, text="停止处理", command=self._stop_processing, width=15).pack(side=tk.LEFT, padx=2)
        
        # 进度和信息区域
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(info_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # 状态标签
        self.status_label = ttk.Label(info_frame, text="就绪 - 请选择图片文件夹")
        self.status_label.pack(pady=2)
        
        # 文件信息
        self.file_info_var = tk.StringVar(value="当前: 0/0")
        self.file_info_label = ttk.Label(info_frame, textvariable=self.file_info_var)
        self.file_info_label.pack(pady=2)
        
        # ========== 显示区域内容 ==========
        
        # 左中右三栏布局
        left_frame = ttk.LabelFrame(display_frame, text="原始图像", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        middle_frame = ttk.LabelFrame(display_frame, text="标记图像", padding="5")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.LabelFrame(display_frame, text="分割结果", padding="5")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 原始图像显示
        self.original_canvas = tk.Canvas(left_frame, bg='#f0f0f0', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 标记图像显示
        self.marked_canvas = tk.Canvas(middle_frame, bg='#f0f0f0', highlightthickness=0)
        self.marked_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 分割结果显示（上下排列）
        result_top_frame = ttk.Frame(right_frame)
        result_top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        result_bottom_frame = ttk.Frame(right_frame)
        result_bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # A面显示
        self.result_a_canvas = tk.Canvas(result_top_frame, bg='#f0f0f0', highlightthickness=0)
        self.result_a_canvas.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(result_top_frame, text="A面", font=("微软雅黑", 10, "bold")).place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        # B面显示
        self.result_b_canvas = tk.Canvas(result_bottom_frame, bg='#f0f0f0', highlightthickness=0)
        self.result_b_canvas.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(result_bottom_frame, text="B面", font=("微软雅黑", 10, "bold")).place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        # 日志区域（底部）
        log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="5")
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 日志控制按钮
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_btn_frame, text="清空日志", command=self._clear_log, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_btn_frame, text="保存日志", command=self._save_log, width=10).pack(side=tk.LEFT, padx=2)
    
    def _browse_folder(self):
        """浏览图片文件夹"""
        directory = filedialog.askdirectory(title="选择包含古籍图像的文件夹")
        if directory:
            self.folder_path_var.set(directory)
            self._load_folder(directory)
    
    def _browse_output(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir_var.set(directory)
    
    def _load_folder(self, folder_path):
        """加载文件夹中的图像文件"""
        try:
            self.current_folder = folder_path
            
            # 支持的图像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP'}
            
            # 查找所有图像文件
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(Path(folder_path).glob(f'*{ext}'))
            
            # 按文件名排序
            self.image_files.sort(key=lambda x: x.name.lower())
            self.current_index = 0
            
            # 更新状态
            total = len(self.image_files)
            self.file_info_var.set(f"当前: 0/{total}")
            self.status_label.config(text=f"找到 {total} 个图像文件")
            self._log(f"已加载文件夹: {folder_path}")
            self._log(f"找到 {total} 个图像文件")
            
            # 显示第一张图片
            if total > 0:
                self._display_current_image()
            else:
                self._clear_displays()
                messagebox.showinfo("提示", "文件夹中没有找到图像文件")
                
        except Exception as e:
            self._log(f"加载文件夹失败: {str(e)}")
    
    def _display_current_image(self):
        """显示当前图片"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        try:
            # 获取当前文件
            current_file = self.image_files[self.current_index]
            
            # 更新状态
            total = len(self.image_files)
            self.file_info_var.set(f"当前: {self.current_index + 1}/{total}")
            self.status_label.config(text=f"正在显示: {current_file.name}")
            
            # 加载并显示原始图像
            pil_image = Image.open(current_file)
            
            # 调整大小以适应画布
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 400, 500
            
            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图像大小
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.original_image_tk = ImageTk.PhotoImage(pil_image)
            
            # 显示在画布上
            self.original_canvas.delete("all")
            self.original_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.original_image_tk, anchor=tk.CENTER
            )
            
            # 清空其他显示区域
            self.marked_canvas.delete("all")
            self.result_a_canvas.delete("all")
            self.result_b_canvas.delete("all")
            
            self._log(f"显示图片: {current_file.name} ({img_width}×{img_height})")
            
        except Exception as e:
            self._log(f"显示图片失败: {str(e)}")
    
    def _prev_image(self):
        """显示上一张图片"""
        if len(self.image_files) > 1:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self._display_current_image()
    
    def _next_image(self):
        """显示下一张图片"""
        if len(self.image_files) > 1:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self._display_current_image()
    
    def _process_current(self):
        """处理当前图片"""
        if not self.image_files or self.current_index >= len(self.image_files):
            messagebox.showwarning("警告", "请先选择图片文件夹")
            return
        
        current_file = self.image_files[self.current_index]
        
        # 添加到处理队列
        self.task_queue.put(('process_current', str(current_file)))
        self._log(f"开始处理: {current_file.name}")
    
    def _process_all(self):
        """处理所有图片"""
        if not self.image_files:
            messagebox.showwarning("警告", "请先选择图片文件夹")
            return
        
        # 确认
        if not messagebox.askyesno("确认", f"开始处理全部 {len(self.image_files)} 张图片？"):
            return
        
        # 添加到处理队列
        for image_file in self.image_files:
            self.task_queue.put(('process_markup', str(image_file)))
        
        self._log(f"开始处理全部 {len(self.image_files)} 张图片")
    
    def _batch_split(self):
        """批量分割保存所有图片"""
        if not self.image_files:
            messagebox.showwarning("警告", "请先选择图片文件夹")
            return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showwarning("警告", "请先选择输出目录")
            return
        
        # 确认
        if not messagebox.askyesno("确认", f"批量分割保存全部 {len(self.image_files)} 张图片？"):
            return
        
        # 添加到处理队列
        for image_file in self.image_files:
            self.task_queue.put(('split_save', str(image_file), output_dir))
        
        self._log(f"开始批量分割保存 {len(self.image_files)} 张图片")
    
    def _align_spine(self):
        """定位中缝（手动调整）"""
        # 这里可以添加手动调整中缝位置的功能
        self._log("中缝定位功能开发中...")
    
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
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def _display_marked_image(self, image_array: np.ndarray, result: ProcessingResult):
        """显示标记图像"""
        try:
            # 转换为PIL图像
            if len(image_array.shape) == 3:
                # BGR转RGB
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image_array)
            
            # 调整大小以适应画布
            canvas_width = self.marked_canvas.winfo_width()
            canvas_height = self.marked_canvas.winfo_height()
            
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 400, 500
            
            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图像大小
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.marked_image_tk = ImageTk.PhotoImage(pil_image)
            
            # 显示在画布上
            self.marked_canvas.delete("all")
            self.marked_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.marked_image_tk, anchor=tk.CENTER
            )
            
            # 显示处理信息
            info = f"置信度: {result.confidence:.1%} | 时间: {result.processing_time:.2f}s"
            self.marked_canvas.create_text(
                10, 15, text=info, anchor=tk.W, 
                fill="white", font=("Arial", 10, "bold")
            )
            
            self._log(f"✓ 处理完成 - 置信度: {result.confidence:.1%}")
            
        except Exception as e:
            self._log(f"显示标记图像失败: {str(e)}")
    
    def _display_split_results(self, a_path: str, b_path: str):
        """显示分割结果"""
        try:
            # 显示A面
            if os.path.exists(a_path):
                pil_image_a = Image.open(a_path)
                
                canvas_width = self.result_a_canvas.winfo_width()
                canvas_height = self.result_a_canvas.winfo_height()
                
                if canvas_width > 10 and canvas_height > 10:
                    img_width, img_height = pil_image_a.size
                    scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    
                    pil_image_a = pil_image_a.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    image_tk_a = ImageTk.PhotoImage(pil_image_a)
                    self.result_images_tk.append(image_tk_a)  # 保存引用
                    
                    self.result_a_canvas.delete("all")
                    self.result_a_canvas.create_image(
                        canvas_width // 2, canvas_height // 2,
                        image=image_tk_a, anchor=tk.CENTER
                    )
            
            # 显示B面
            if os.path.exists(b_path):
                pil_image_b = Image.open(b_path)
                
                canvas_width = self.result_b_canvas.winfo_width()
                canvas_height = self.result_b_canvas.winfo_height()
                
                if canvas_width > 10 and canvas_height > 10:
                    img_width, img_height = pil_image_b.size
                    scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    
                    pil_image_b = pil_image_b.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    image_tk_b = ImageTk.PhotoImage(pil_image_b)
                    self.result_images_tk.append(image_tk_b)  # 保存引用
                    
                    self.result_b_canvas.delete("all")
                    self.result_b_canvas.create_image(
                        canvas_width // 2, canvas_height // 2,
                        image=image_tk_b, anchor=tk.CENTER
                    )
            
            self._log(f"✓ 分割完成 - A面: {os.path.basename(a_path)}, B面: {os.path.basename(b_path)}")
            
        except Exception as e:
            self._log(f"显示分割结果失败: {str(e)}")
    
    def _clear_displays(self):
        """清空所有显示"""
        self.original_canvas.delete("all")
        self.marked_canvas.delete("all")
        self.result_a_canvas.delete("all")
        self.result_b_canvas.delete("all")
        self.result_images_tk.clear()
    
    def _start_processing_thread(self):
        """启动处理线程"""
        def processing_worker():
            self.processing = True
            
            while self.processing:
                try:
                    # 从队列获取任务
                    task = self.task_queue.get(timeout=0.1)
                    
                    task_type = task[0]
                    
                    if task_type == 'process_current':
                        image_path = task[1]
                        self._process_single_markup(image_path)
                    
                    elif task_type == 'process_markup':
                        image_path = task[1]
                        self._process_markup_only(image_path)
                    
                    elif task_type == 'split_save':
                        image_path, output_dir = task[1], task[2]
                        self._process_split_save(image_path, output_dir)
                    
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self._log(f"处理线程错误: {str(e)}")
        
        # 启动线程
        thread = threading.Thread(target=processing_worker, daemon=True)
        thread.start()
    
    def _process_single_markup(self, image_path):
        """处理单张图片并显示标记"""
        try:
            self._update_progress(30, "正在处理...")
            
            # 处理图像（仅标记，不保存）
            marked_image, result = self.processor.process_image_with_markup(image_path)
            
            if result.success and marked_image is not None:
                # 在主线程中显示结果
                self.root.after(0, lambda: self._display_marked_image(marked_image, result))
                self._update_progress(100, "处理完成")
            else:
                self._log(f"✗ 处理失败: {image_path}")
                self._update_progress(0, "处理失败")
                
        except Exception as e:
            self._log(f"处理失败: {str(e)}")
            self._update_progress(0, "处理失败")
    
    def _process_markup_only(self, image_path):
        """仅处理标记（批量处理用）"""
        try:
            _, result = self.processor.process_image_with_markup(image_path)
            
            if result.success:
                self._log(f"✓ 已处理: {os.path.basename(image_path)} (置信度: {result.confidence:.1%})")
            else:
                self._log(f"✗ 处理失败: {os.path.basename(image_path)}")
                
        except Exception as e:
            self._log(f"处理失败 {os.path.basename(image_path)}: {str(e)}")
    
    def _process_split_save(self, image_path, output_dir):
        """处理并分割保存"""
        try:
            self._log(f"处理并分割: {os.path.basename(image_path)}")
            
            # 处理并分割图像
            output_paths, result = self.processor.process_and_split_image(image_path, output_dir)
            
            if result.success and output_paths:
                a_path, b_path = output_paths
                
                # 如果是当前显示的图片，更新显示
                current_file = self.image_files[self.current_index] if self.image_files else None
                if current_file and str(current_file) == image_path:
                    self.root.after(0, lambda: self._display_split_results(a_path, b_path))
                
                self._log(f"✓ 分割完成: {os.path.basename(a_path)}, {os.path.basename(b_path)}")
            else:
                self._log(f"✗ 分割失败: {os.path.basename(image_path)}")
                
        except Exception as e:
            self._log(f"分割失败 {os.path.basename(image_path)}: {str(e)}")
    
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
    print("古籍书影智能处理系统 v4.0 - 文件夹批量处理版")
    print("=" * 60)
    
    try:
        # 检查OpenCV
        print(f"OpenCV版本: {cv2.__version__}")
        
        # 创建并运行GUI
        app = AncientBookGUI()
        app.run()
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")


if __name__ == "__main__":
    main()
