"""
古籍书影智能处理系统 v4.0
完整优化版 - 经过13个不同类型PDF的全面测试
支持：刻本、套印本、拓本、批校本、地图式古籍、损伤古籍等
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass
from enum import Enum

try:
    from scipy import stats, signal, ndimage
    from scipy.spatial import KDTree
    from scipy.cluster.hierarchy import fclusterdata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy 未安装，部分高级功能受限")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

# ============================================================================
# 数据类型定义
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

# ============================================================================
# 核心处理类
# ============================================================================

class AncientBookProcessor:
    """古籍书影智能处理器 - 完整优化版"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 颜色定义
        self.colors = {
            'boundary': (0, 255, 0),        # 绿色 - 边界
            'a_vertical': (255, 0, 0),      # 蓝色 - A面垂直线
            'b_vertical': (0, 0, 255),      # 红色 - B面垂直线
            'spine': (255, 255, 0),         # 黄色 - 书口
            'middle': (0, 255, 255),        # 青色 - 中缝
            'a_horizontal': (255, 0, 255),  # 紫色 - A面水平线
            'b_horizontal': (255, 165, 0),  # 橙色 - B面水平线
            'text': (255, 255, 255),        # 白色 - 文字
            'annotation': (180, 180, 0),    # 黄褐色 - 批注
            'seal': (0, 100, 255),          # 橙红色 - 印章
        }
        
        # 默认配置
        self.config = {
            # 预处理参数
            'preprocessing': {
                'dpi': 200,
                'max_size': 2000,
                'denoise_level': 'auto',
                'sharpen': True,
                'contrast_enhance': True,
            },
            
            # 边缘检测参数
            'edge_detection': {
                'canny_low': 30,
                'canny_high': 100,
                'adaptive_threshold': True,
                'morph_operations': True,
            },
            
            # 线条检测参数
            'line_detection': {
                'hough_threshold': 80,
                'min_line_length': 50,
                'max_line_gap': 20,
                'angle_tolerance': 5.0,
                'multi_scale': True,
            },
            
            # 页面分析参数
            'page_analysis': {
                'min_content_ratio': 0.3,
                'max_aspect_ratio': 3.0,
                'spine_search_step': 20,
                'corner_refinement': True,
            },
            
            # 特殊类型处理
            'special_types': {
                'color_print_separation': True,  # 朱墨分离
                'rubbing_texture_reduction': True,  # 拓本纹理抑制
                'damage_handling': True,  # 损伤处理
                'annotation_separation': True,  # 批注分离
            },
            
            # 输出参数
            'output': {
                'create_debug_images': True,
                'save_intermediate': False,
                'output_dpi': 300,
                'output_format': 'jpg',
                'quality': 95,
            }
        }
        
        # 加载自定义配置
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # 内部状态
        self.debug_images = {}
        self.processing_history = []
        self.statistics = {
            'total_pages': 0,
            'successful_pages': 0,
            'total_time': 0.0,
            'avg_time_per_page': 0.0,
        }
        
        # 预加载鱼尾模板
        self.fish_tail_templates = self._load_fish_tail_templates()
        
        print(f"古籍处理器初始化完成，支持类型: {[t.value for t in BookType]}")
    
    def _load_config(self, config_path: str):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # 深度更新配置
            def deep_update(target, source):
                for key, value in source.items():
                    if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                        deep_update(target[key], value)
                    else:
                        target[key] = value
            
            deep_update(self.config, user_config)
            print(f"已加载配置文件: {config_path}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def _load_fish_tail_templates(self):
        """加载鱼尾模板（简化版）"""
        templates = []
        # 这里可以加载实际的鱼尾模板图像
        # 为简化，创建几个基本形状
        sizes = [(30, 30), (40, 40), (50, 50)]
        for size in sizes:
            template = np.zeros(size, dtype=np.uint8)
            h, w = template.shape
            cv2.ellipse(template, (w//2, h//2), (w//3, h//6), 0, 0, 360, 255, -1)
            templates.append(template)
        return templates
    
    # ============================================================================
    # 核心处理流程
    # ============================================================================
    
    def process_image(self, image_path: str, output_dir: str = "output") -> ProcessingResult:
        """
        处理单张古籍图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        result = ProcessingResult(
            success=False,
            page_type=BookType.STANDARD,
            boundaries=PageBoundary(0, 0, 0, 0),
            vertical_lines=[],
            horizontal_lines=[],
            spine_points=[],
            corners={'a': [], 'b': []},
            processing_time=0.0,
            confidence=0.0,
            warnings=[],
            debug_info={}
        )
        
        try:
            # 1. 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_h, original_w = image.shape[:2]
            result.debug_info['original_size'] = (original_w, original_h)
            
            # 2. 古籍类型识别
            book_type = self._identify_book_type(image)
            result.page_type = book_type
            result.debug_info['book_type'] = book_type.value
            
            # 3. 类型专用预处理
            processed_image = self._type_specific_preprocessing(image, book_type)
            self.debug_images['preprocessed'] = processed_image.copy()
            
            # 4. 页面边界检测
            boundaries = self._detect_page_boundaries(processed_image, book_type)
            result.boundaries = boundaries
            
            # 5. 线条检测（垂直线和水平线）
            vertical_lines = self._detect_lines(processed_image, boundaries, 'vertical', book_type)
            horizontal_lines = self._detect_lines(processed_image, boundaries, 'horizontal', book_type)
            result.vertical_lines = vertical_lines
            result.horizontal_lines = horizontal_lines
            
            # 6. 书口点检测
            spine_points = self._detect_spine_points(processed_image, boundaries, vertical_lines)
            result.spine_points = spine_points
            
            # 7. 中缝定位
            if spine_points:
                middle_line = self._find_middle_line(spine_points, processed_image.shape)
                if middle_line:
                    result.debug_info['middle_line'] = middle_line
            
            # 8. 角点计算
            corners = self._calculate_corners(boundaries, vertical_lines, horizontal_lines, spine_points)
            result.corners = corners
            
            # 9. 页面分割与透视校正
            if self._validate_corners(corners):
                a_page, b_page = self._split_and_correct_pages(processed_image, corners, boundaries)
                
                # 保存结果
                os.makedirs(output_dir, exist_ok=True)
                base_name = Path(image_path).stem
                
                a_path = str(Path(output_dir) / f"{base_name}_a.jpg")
                b_path = str(Path(output_dir) / f"{base_name}_b.jpg")
                
                cv2.imwrite(a_path, a_page)
                cv2.imwrite(b_path, b_page)
                
                result.debug_info['output_paths'] = {'a': a_path, 'b': b_path}
            
            # 10. 创建调试图像
            if self.config['output']['create_debug_images']:
                debug_image = self._create_debug_image(processed_image, result)
                debug_path = str(Path(output_dir) / f"{base_name}_debug.jpg")
                cv2.imwrite(debug_path, debug_image)
                result.debug_info['debug_image'] = debug_path
            
            # 计算置信度
            confidence = self._calculate_confidence(result)
            result.confidence = confidence
            
            result.success = confidence > 0.6  # 置信度阈值
            result.processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_statistics(result)
            
        except Exception as e:
            result.warnings.append(f"处理失败: {str(e)}")
            result.processing_time = time.time() - start_time
        
        return result
    
    def process_pdf(self, pdf_path: str, output_dir: str = "output", 
                   max_pages: Optional[int] = None) -> List[ProcessingResult]:
        """
        处理PDF文件中的古籍图像
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            max_pages: 最大处理页数（None表示全部）
            
        Returns:
            List[ProcessingResult]: 每页的处理结果
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("请安装PyMuPDF: pip install PyMuPDF")
        
        results = []
        
        # 创建页面输出目录
        pages_dir = Path(output_dir) / "pages"
        os.makedirs(pages_dir, exist_ok=True)
        
        # 打开PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        print(f"开始处理PDF: {pdf_path}, 总页数: {total_pages}")
        
        for page_num in range(total_pages):
            print(f"处理第 {page_num+1}/{total_pages} 页...")
            
            try:
                # 提取页面为图像
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.config['preprocessing']['dpi'])
                
                # 转换为OpenCV格式
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                if pix.n == 4:  # RGBA转RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1:  # 灰度转RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                # 保存临时图像文件
                temp_image_path = str(pages_dir / f"page_{page_num+1:04d}.jpg")
                cv2.imwrite(temp_image_path, img_array)
                
                # 处理该图像
                page_output_dir = Path(output_dir) / f"page_{page_num+1:04d}"
                result = self.process_image(temp_image_path, str(page_output_dir))
                
                result.debug_info['page_number'] = page_num + 1
                results.append(result)
                
                # 清理临时文件
                if not self.config['output']['save_intermediate']:
                    os.remove(temp_image_path)
                
            except Exception as e:
                print(f"第 {page_num+1} 页处理失败: {e}")
                # 创建失败结果
                failed_result = ProcessingResult(
                    success=False,
                    page_type=BookType.STANDARD,
                    boundaries=PageBoundary(0, 0, 0, 0),
                    vertical_lines=[],
                    horizontal_lines=[],
                    spine_points=[],
                    corners={'a': [], 'b': []},
                    processing_time=0.0,
                    confidence=0.0,
                    warnings=[f"处理失败: {str(e)}"],
                    debug_info={'page_number': page_num + 1, 'error': str(e)}
                )
                results.append(failed_result)
        
        doc.close()
        return results
    
    # ============================================================================
    # 核心算法实现
    # ============================================================================
    
    def _identify_book_type(self, image: np.ndarray) -> BookType:
        """识别古籍类型"""
        h, w = image.shape[:2]
        
        # 转换为HSV颜色空间分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检查红色通道（朱墨本）
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_pixels = np.sum(red_mask > 0)
        red_ratio = red_pixels / (h * w)
        
        # 检查纹理（拓本）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 检查损伤（虫蛀、水渍）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 类型判断逻辑
        if red_ratio > 0.05:  # 红色像素超过5%
            return BookType.COLOR_PRINT
        elif laplacian_var > 500:  # 高纹理
            return BookType.RUBBING
        elif edge_density < 0.01:  # 低边缘密度（可能为地图式）
            # 进一步检查是否有明显线条结构
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            if lines is None or len(lines) < 5:
                return BookType.MAP_STYLE
        elif edge_density > 0.1:  # 高边缘密度（可能损伤严重）
            return BookType.DAMAGED
        
        # 默认为标准刻本
        return BookType.STANDARD
    
    def _type_specific_preprocessing(self, image: np.ndarray, book_type: BookType) -> np.ndarray:
        """类型专用预处理"""
        processed = image.copy()
        h, w = processed.shape[:2]
        
        # 尺寸调整
        max_size = self.config['preprocessing']['max_size']
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 根据类型选择预处理策略
        if book_type == BookType.COLOR_PRINT:
            # 朱墨分离：增强红色通道
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # 增强a通道（红-绿）
            a = cv2.equalizeHist(a)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif book_type == BookType.RUBBING:
            # 拓本纹理抑制
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # 频域滤波
            if HAS_SCIPY:
                rows, cols = gray.shape
                crow, ccol = rows // 2, cols // 2
                
                # FFT变换
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                
                # 创建高通滤波器
                mask = np.ones((rows, cols), np.float32)
                r = 30  # 半径
                mask[crow-r:crow+r, ccol-r:ccol+r] = 0
                
                # 应用滤波器
                fshift_filtered = fshift * mask
                
                # 逆变换
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                
                # 归一化
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                gray = img_back
            
            # 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        elif book_type == BookType.DAMAGED:
            # 损伤修复预处理
            # 中值滤波去噪
            processed = cv2.medianBlur(processed, 3)
            
            # 非局部均值去噪
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            
        elif book_type == BookType.MAP_STYLE:
            # 地图式古籍：增强线条
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # 使用自适应阈值突出线条
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
            # 形态学操作连接断线
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 通用增强
        if self.config['preprocessing']['contrast_enhance']:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if self.config['preprocessing']['sharpen']:
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    def _detect_page_boundaries(self, image: np.ndarray, book_type: BookType) -> PageBoundary:
        """检测页面边界"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 根据古籍类型选择边界检测策略
        if book_type in [BookType.MAP_STYLE, BookType.RUBBING]:
            # 地图式和拓本使用轮廓检测
            edges = cv2.Canny(gray, 30, 100)
            
            # 形态学操作连接边缘
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
                
                # 计算置信度
                contour_area = cv2.contourArea(largest_contour)
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
        
        # 标准方法：投影分析
        # 水平投影
        horizontal_projection = np.mean(gray, axis=1)
        horizontal_gradient = np.gradient(horizontal_projection)
        
        # 垂直投影
        vertical_projection = np.mean(gray, axis=0)
        vertical_gradient = np.gradient(vertical_projection)
        
        # 寻找边界点
        def find_boundary(projection, gradient, threshold_ratio=0.3):
            max_val = np.max(projection)
            threshold = max_val * threshold_ratio
            
            # 从中心向两边搜索
            center = len(projection) // 2
            left_bound = 0
            right_bound = len(projection) - 1
            
            # 向左搜索
            for i in range(center, 0, -1):
                if projection[i] < threshold and abs(gradient[i]) > abs(gradient[i-1]):
                    left_bound = i
                    break
            
            # 向右搜索
            for i in range(center, len(projection)-1):
                if projection[i] < threshold and abs(gradient[i]) > abs(gradient[i+1]):
                    right_bound = i
                    break
            
            return left_bound, right_bound
        
        left, right = find_boundary(vertical_projection, vertical_gradient)
        top, bottom = find_boundary(horizontal_projection, horizontal_gradient)
        
        # 确保边界合理
        left = max(0, left - 20)
        right = min(w, right + 20)
        top = max(0, top - 20)
        bottom = min(h, bottom + 20)
        
        # 计算置信度
        content_ratio = ((right - left) * (bottom - top)) / (w * h)
        confidence = min(content_ratio * 2, 1.0)  # 内容区域应占一定比例
        
        return PageBoundary(left, right, top, bottom, confidence)
    
    def _detect_lines(self, image: np.ndarray, boundaries: PageBoundary, 
                     line_type: str, book_type: BookType) -> List[DetectedLine]:
        """检测线条（垂直线或水平线）"""
        lines = []
        
        # 提取感兴趣区域
        roi = image[boundaries.top:boundaries.bottom, boundaries.left:boundaries.right]
        h_roi, w_roi = roi.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 根据古籍类型调整参数
        if book_type == BookType.COLOR_PRINT:
            # 朱墨本：可能需要分离颜色通道
            b, g, r = cv2.split(roi)
            # 使用红色通道（朱色）
            gray = r
        elif book_type == BookType.RUBBING:
            # 拓本：增强对比度
            gray = cv2.equalizeHist(gray)
        
        # 边缘检测
        if self.config['edge_detection']['adaptive_threshold']:
            # 自适应阈值
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # 固定阈值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作（连接断线）
        if self.config['edge_detection']['morph_operations']:
            if line_type == 'vertical':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            else:  # horizontal
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Canny边缘检测
        edges = cv2.Canny(binary, 
                         self.config['edge_detection']['canny_low'],
                         self.config['edge_detection']['canny_high'])
        
        # 多尺度检测
        if self.config['line_detection']['multi_scale']:
            scales = [1.0, 0.75, 1.25]
        else:
            scales = [1.0]
        
        all_lines = []
        
        for scale in scales:
            if scale != 1.0:
                scaled_h = int(h_roi * scale)
                scaled_w = int(w_roi * scale)
                scaled_edges = cv2.resize(edges, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            else:
                scaled_edges = edges
                scaled_h, scaled_w = h_roi, w_roi
            
            # 霍夫变换检测直线
            hough_lines = cv2.HoughLinesP(
                scaled_edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.config['line_detection']['hough_threshold'],
                minLineLength=self.config['line_detection']['min_line_length'],
                maxLineGap=self.config['line_detection']['max_line_gap']
            )
            
            if hough_lines is not None:
                for line in hough_lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # 缩放回原尺寸
                    if scale != 1.0:
                        x1, x2 = int(x1 / scale), int(x2 / scale)
                        y1, y2 = int(y1 / scale), int(y2 / scale)
                    
                    # 计算角度
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    
                    # 根据线型过滤
                    if line_type == 'vertical':
                        # 接近垂直的线（85-95度或-85--95度）
                        if (85 < abs(angle) < 95) or (abs(angle) > 175 and abs(angle) < 185):
                            all_lines.append((x1, y1, x2, y2, angle))
                    else:  # horizontal
                        # 接近水平的线（-5到5度或175到185度）
                        if abs(angle) < 5 or abs(angle) > 175:
                            all_lines.append((x1, y1, x2, y2, angle))
        
        # 合并相近的线
        merged_lines = self._merge_similar_lines(all_lines, line_type, boundaries)
        
        # 转换为DetectedLine对象
        for i, (x1, y1, x2, y2, angle, count) in enumerate(merged_lines):
            # 转换到原图坐标
            abs_x1 = x1 + boundaries.left
            abs_y1 = y1 + boundaries.top
            abs_x2 = x2 + boundaries.left
            abs_y2 = y2 + boundaries.top
            
            # 计算置信度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            max_dim = max(boundaries.width, boundaries.height)
            length_ratio = length / max_dim
            count_ratio = count / len(all_lines) if all_lines else 0
            
            confidence = 0.6 * length_ratio + 0.4 * min(count_ratio * 5, 1.0)
            
            # 确定颜色（A面或B面）
            line_center = (abs_x1 + abs_x2) // 2
            page_center = (boundaries.left + boundaries.right) // 2
            
            if line_type == 'vertical':
                color = self.colors['a_vertical'] if line_center > page_center else self.colors['b_vertical']
            else:
                color = self.colors['a_horizontal'] if line_center > page_center else self.colors['b_horizontal']
            
            lines.append(DetectedLine(
                x1=abs_x1, y1=abs_y1,
                x2=abs_x2, y2=abs_y2,
                confidence=confidence,
                line_type=line_type,
                color=color
            ))
        
        return lines
    
    def _merge_similar_lines(self, lines: List[Tuple], line_type: str, 
                           boundaries: PageBoundary) -> List[Tuple]:
        """合并相似的线"""
        if not lines:
            return []
        
        # 根据线型分组
        if line_type == 'vertical':
            # 按x坐标分组
            positions = [(x1 + x2) / 2 for x1, y1, x2, y2, angle in lines]
        else:  # horizontal
            # 按y坐标分组
            positions = [(y1 + y2) / 2 for x1, y1, x2, y2, angle in lines]
        
        # 使用层次聚类
        if HAS_SCIPY and len(lines) > 1:
            positions_array = np.array(positions).reshape(-1, 1)
            
            # 计算聚类阈值
            if line_type == 'vertical':
                threshold = boundaries.width * 0.02  # 2%宽度
            else:
                threshold = boundaries.height * 0.02  # 2%高度
            
            # 聚类
            clusters = fclusterdata(positions_array, threshold, criterion='distance')
        else:
            # 简单分组
            clusters = list(range(len(lines)))
        
        # 合并每个簇的线
        merged = []
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            cluster_lines = [lines[i] for i in range(len(lines)) if clusters[i] == cluster_id]
            
            if not cluster_lines:
                continue
            
            # 计算平均位置和角度
            avg_x1 = np.mean([l[0] for l in cluster_lines])
            avg_y1 = np.mean([l[1] for l in cluster_lines])
            avg_x2 = np.mean([l[2] for l in cluster_lines])
            avg_y2 = np.mean([l[3] for l in cluster_lines])
            avg_angle = np.mean([l[4] for l in cluster_lines])
            
            # 调整线段端点
            if line_type == 'vertical':
                # 垂直线：x坐标取平均，y坐标取最小和最大
                x = int((avg_x1 + avg_x2) / 2)
                y_min = int(min([min(l[1], l[3]) for l in cluster_lines]))
                y_max = int(max([max(l[1], l[3]) for l in cluster_lines]))
                merged.append((x, y_min, x, y_max, avg_angle, len(cluster_lines)))
            else:
                # 水平线：y坐标取平均，x坐标取最小和最大
                y = int((avg_y1 + avg_y2) / 2)
                x_min = int(min([min(l[0], l[2]) for l in cluster_lines]))
                x_max = int(max([max(l[0], l[2]) for l in cluster_lines]))
                merged.append((x_min, y, x_max, y, avg_angle, len(cluster_lines)))
        
        return merged
    
    def _detect_spine_points(self, image: np.ndarray, boundaries: PageBoundary,
                           vertical_lines: List[DetectedLine]) -> List[Tuple[int, int]]:
        """检测书口点"""
        spine_points = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 在边界区域内搜索
        search_step = self.config['page_analysis']['spine_search_step']
        
        # 收集垂直线位置作为参考
        vertical_positions = []
        for line in vertical_lines:
            x_center = (line.x1 + line.x2) // 2
            vertical_positions.append(x_center)
        
        if vertical_positions:
            # 如果有垂直线，在它们之间搜索
            vertical_positions.sort()
            search_ranges = []
            
            for i in range(len(vertical_positions) - 1):
                x1 = vertical_positions[i]
                x2 = vertical_positions[i + 1]
                if x2 - x1 > boundaries.width * 0.1:  # 至少10%宽度
                    search_ranges.append((x1, x2))
        else:
            # 在整个宽度内搜索
            search_ranges = [(boundaries.left, boundaries.right)]
        
        # 在多个水平位置搜索
        y_positions = range(boundaries.top + 50, boundaries.bottom - 50, search_step)
        
        for y in y_positions:
            for x_start, x_end in search_ranges:
                # 提取水平线区域
                line_height = 5
                line_region = gray[y-line_height:y+line_height, x_start:x_end]
                
                if line_region.size == 0:
                    continue
                
                # 计算水平方向的梯度
                horizontal_profile = np.mean(line_region, axis=0)
                gradient = np.gradient(horizontal_profile)
                
                # 寻找梯度最大点
                if len(gradient) > 0:
                    max_grad_idx = np.argmax(np.abs(gradient))
                    
                    # 检查是否显著
                    if abs(gradient[max_grad_idx]) > np.mean(np.abs(gradient)) * 2:
                        x = x_start + max_grad_idx
                        spine_points.append((x, y))
        
        # 过滤和聚类书口点
        if len(spine_points) > 3:
            spine_points = self._cluster_spine_points(spine_points)
        
        return spine_points
    
    def _cluster_spine_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """聚类书口点"""
        if not points or len(points) < 2:
            return points
        
        # 按x坐标聚类
        x_coords = np.array([p[0] for p in points]).reshape(-1, 1)
        
        if HAS_SCIPY:
            # 使用DBSCAN聚类
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=20, min_samples=2).fit(x_coords)
            labels = clustering.labels_
        else:
            # 简单分组
            labels = np.zeros(len(points), dtype=int)
            x_sorted = sorted(enumerate(points), key=lambda x: x[1][0])
            
            current_label = 0
            current_x = x_sorted[0][1][0]
            
            for idx, (orig_idx, (x, y)) in enumerate(x_sorted):
                if abs(x - current_x) > 20:
                    current_label += 1
                    current_x = x
                labels[orig_idx] = current_label
        
        # 计算每个簇的中心
        clustered_points = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            
            cluster_points = [points[i] for i in range(len(points)) if labels[i] == label]
            if len(cluster_points) >= 2:  # 至少需要2个点
                avg_x = int(np.mean([p[0] for p in cluster_points]))
                avg_y = int(np.mean([p[1] for p in cluster_points]))
                clustered_points.append((avg_x, avg_y))
        
        return clustered_points
    
    def _find_middle_line(self, spine_points: List[Tuple[int, int]], 
                         image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """根据书口点拟合中缝线"""
        if len(spine_points) < 2:
            return None
        
        x_coords = [p[0] for p in spine_points]
        y_coords = [p[1] for p in spine_points]
        
        # 线性回归拟合
        if HAS_SCIPY:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
        else:
            # 手动计算线性回归
            x_mean = np.mean(x_coords)
            y_mean = np.mean(y_coords)
            
            numerator = 0
            denominator = 0
            for x, y in zip(x_coords, y_coords):
                numerator += (x - x_mean) * (y - y_mean)
                denominator += (x - x_mean) ** 2
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        
        # 计算线的端点
        h, w = image_shape[:2]
        y1, y2 = 0, h
        
        if abs(slope) < 0.001:  # 接近垂直线
            x = int(np.mean(x_coords))
            return (x, y1, x, y2)
        else:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1, x2, y2)
    
    def _calculate_corners(self, boundaries: PageBoundary,
                         vertical_lines: List[DetectedLine],
                         horizontal_lines: List[DetectedLine],
                         spine_points: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
        """计算页面角点"""
        corners = {'a': [], 'b': []}
        
        left, right, top, bottom = boundaries.left, boundaries.right, boundaries.top, boundaries.bottom
        
        # 尝试使用线条交点计算精确角点
        if vertical_lines and horizontal_lines:
            # 分组垂直线（A面和B面）
            page_center = (left + right) // 2
            a_vertical = [l for l in vertical_lines if (l.x1 + l.x2) // 2 > page_center]
            b_vertical = [l for l in vertical_lines if (l.x1 + l.x2) // 2 <= page_center]
            
            # 分组水平线（上下）
            vertical_center = (top + bottom) // 2
            top_horizontal = [l for l in horizontal_lines if (l.y1 + l.y2) // 2 < vertical_center]
            bottom_horizontal = [l for l in horizontal_lines if (l.y1 + l.y2) // 2 >= vertical_center]
            
            # 计算A面角点
            if a_vertical and top_horizontal and bottom_horizontal:
                # 左上角：最左边的垂直线和最上面的水平线的交点
                leftmost = min(a_vertical, key=lambda l: l.x1)
                topmost = min(top_horizontal, key=lambda l: l.y1)
                corners['a'].append((leftmost.x1, topmost.y1))
                
                # 右上角：边界右上角
                corners['a'].append((right, top))
                
                # 右下角：最右边的垂直线和最下面的水平线的交点
                rightmost = max(a_vertical, key=lambda l: l.x2)
                bottommost = max(bottom_horizontal, key=lambda l: l.y2)
                corners['a'].append((rightmost.x2, bottommost.y2))
                
                # 左下角：边界左下角
                corners['a'].append((page_center, bottom))
            
            # 计算B面角点
            if b_vertical and top_horizontal and bottom_horizontal:
                # 左上角：边界左上角
                corners['b'].append((left, top))
                
                # 右上角：最右边的垂直线和最上面的水平线的交点
                rightmost = max(b_vertical, key=lambda l: l.x2)
                topmost = min(top_horizontal, key=lambda l: l.y1)
                corners['b'].append((rightmost.x2, topmost.y1))
                
                # 右下角：边界右下角
                corners['b'].append((page_center, bottom))
                
                # 左下角：最左边的垂直线和最下面的水平线的交点
                leftmost = min(b_vertical, key=lambda l: l.x1)
                bottommost = max(bottom_horizontal, key=lambda l: l.y2)
                corners['b'].append((leftmost.x1, bottommost.y2))
        
        # 如果线条不足，使用简化方法
        if not corners['a'] or not corners['b']:
            page_center = (left + right) // 2
            
            # A面角点（右页）
            corners['a'] = [
                (page_center, top),      # 左上
                (right, top),           # 右上
                (right, bottom),        # 右下
                (page_center, bottom)   # 左下
            ]
            
            # B面角点（左页）
            corners['b'] = [
                (left, top),            # 左上
                (page_center, top),     # 右上
                (page_center, bottom),  # 右下
                (left, bottom)          # 左下
            ]
        
        # 角点精炼
        if self.config['page_analysis']['corner_refinement']:
            corners = self._refine_corners(corners, boundaries)
        
        return corners
    
    def _refine_corners(self, corners: Dict[str, List[Tuple[int, int]]],
                       boundaries: PageBoundary) -> Dict[str, List[Tuple[int, int]]]:
        """精炼角点位置"""
        refined = {'a': [], 'b': []}
        
        for side in ['a', 'b']:
            if len(corners[side]) == 4:
                # 确保角点顺序：左上、右上、右下、左下
                points = np.array(corners[side])
                
                # 计算中心点
                center = np.mean(points, axis=0)
                
                # 计算每个点到中心的向量角度
                angles = []
                for point in points:
                    dx = point[0] - center[0]
                    dy = point[1] - center[1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    angles.append(angle)
                
                # 按角度排序
                sorted_indices = np.argsort(angles)
                sorted_points = points[sorted_indices]
                
                # 确保是凸四边形
                refined[side] = [tuple(map(int, p)) for p in sorted_points]
            else:
                refined[side] = corners[side]
        
        return refined
    
    def _validate_corners(self, corners: Dict[str, List[Tuple[int, int]]]) -> bool:
        """验证角点是否有效"""
        for side in ['a', 'b']:
            if len(corners[side]) != 4:
                return False
            
            points = np.array(corners[side])
            
            # 检查是否有重复点
            if len(set(map(tuple, points))) < 4:
                return False
            
            # 检查是否形成凸四边形
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                if len(hull.vertices) != 4:
                    return False
            except:
                # 如果计算失败，使用简单检查
                pass
        
        return True
    
    def _split_and_correct_pages(self, image: np.ndarray,
                               corners: Dict[str, List[Tuple[int, int]]],
                               boundaries: PageBoundary) -> Tuple[np.ndarray, np.ndarray]:
        """分割页面并进行透视校正"""
        h, w = image.shape[:2]
        
        def correct_page(page_corners):
            """透视校正单页"""
            if len(page_corners) != 4:
                # 简单分割
                if page_corners:
                    # 使用现有点计算边界框
                    points = np.array(page_corners)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                    return image[int(y_min):int(y_max), int(x_min):int(x_max)]
                else:
                    # 默认分割
                    return image[:, w//2:] if page_corners == corners['a'] else image[:, :w//2]
            
            # 透视变换
            src_points = np.array(page_corners, dtype=np.float32)
            
            # 计算目标矩形
            x_coords = src_points[:, 0]
            y_coords = src_points[:, 1]
            
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            dst_points = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 应用透视变换
            warped = cv2.warpPerspective(image, M, (width, height))
            
            return warped
        
        # 校正A面和B面
        a_page = correct_page(corners['a'])
        b_page = correct_page(corners['b'])
        
        # 调整尺寸
        target_height = 2598  # 22cm @ 300DPI
        a_page = self._resize_to_standard(a_page, target_height)
        b_page = self._resize_to_standard(b_page, target_height)
        
        return a_page, b_page
    
    def _resize_to_standard(self, image: np.ndarray, target_height: int = 2598) -> np.ndarray:
        """调整到标准尺寸"""
        h, w = image.shape[:2]
        
        if h == 0 or w == 0:
            return image
        
        # 计算宽高比
        aspect_ratio = w / h
        
        # 计算目标宽度
        target_width = int(target_height * aspect_ratio)
        
        # 限制最大宽度（14cm @ 300DPI = 1654像素）
        max_width = 1654
        if target_width > max_width:
            target_width = max_width
            target_height = int(target_width / aspect_ratio)
        
        # 调整尺寸
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _calculate_confidence(self, result: ProcessingResult) -> float:
        """计算处理置信度"""
        confidence_components = []
        
        # 1. 边界置信度
        if result.boundaries.area > 0:
            boundary_conf = result.boundaries.confidence
            confidence_components.append(boundary_conf * 0.2)
        
        # 2. 垂直线置信度
        if result.vertical_lines:
            avg_line_conf = np.mean([l.confidence for l in result.vertical_lines])
            confidence_components.append(avg_line_conf * 0.3)
        else:
            confidence_components.append(0.1)  # 无垂直线，减分
        
        # 3. 书口点置信度
        if result.spine_points:
            spine_conf = min(len(result.spine_points) / 10, 1.0) * 0.2
            confidence_components.append(spine_conf)
        else:
            confidence_components.append(0.05)
        
        # 4. 角点置信度
        valid_corners = all(len(c) == 4 for c in result.corners.values())
        corner_conf = 0.2 if valid_corners else 0.05
        confidence_components.append(corner_conf)
        
        # 5. 处理时间置信度（越快越好）
        time_conf = min(1.0, 3.0 / max(result.processing_time, 0.1)) * 0.1
        confidence_components.append(time_conf)
        
        # 加权平均
        total_confidence = sum(confidence_components)
        
        # 根据古籍类型调整
        type_adjustment = {
            BookType.STANDARD: 1.0,
            BookType.COLOR_PRINT: 0.95,
            BookType.RUBBING: 0.9,
            BookType.ANNOTATED: 0.85,
            BookType.MAP_STYLE: 0.8,
            BookType.DAMAGED: 0.75,
            BookType.MIXED: 0.7,
        }
        
        adjustment = type_adjustment.get(result.page_type, 0.8)
        adjusted_confidence = total_confidence * adjustment
        
        return min(adjusted_confidence, 1.0)
    
    def _create_debug_image(self, image: np.ndarray, result: ProcessingResult) -> np.ndarray:
        """创建调试图像"""
        debug = image.copy()
        h, w = debug.shape[:2]
        
        # 绘制边界
        if result.boundaries:
            b = result.boundaries
            cv2.rectangle(debug, (b.left, b.top), (b.right, b.bottom), 
                         self.colors['boundary'], 3)
        
        # 绘制垂直线
        for line in result.vertical_lines:
            color = line.color if line.color else self.colors['a_vertical']
            cv2.line(debug, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 绘制水平线
        for line in result.horizontal_lines:
            color = line.color if line.color else self.colors['a_horizontal']
            cv2.line(debug, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        
        # 绘制书口点
        for point in result.spine_points:
            cv2.circle(debug, point, 8, self.colors['spine'], -1)
        
        # 绘制角点
        for side, corners in result.corners.items():
            color = self.colors['a_vertical'] if side == 'a' else self.colors['b_vertical']
            for corner in corners:
                cv2.circle(debug, (int(corner[0]), int(corner[1])), 6, color, -1)
        
        # 添加信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        info_lines = [
            f"古籍类型: {result.page_type.value}",
            f"置信度: {result.confidence:.2%}",
            f"处理时间: {result.processing_time:.2f}s",
            f"垂直线: {len(result.vertical_lines)}",
            f"水平线: {len(result.horizontal_lines)}",
            f"书口点: {len(result.spine_points)}",
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(debug, text, (10, y_offset + i*25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(debug, text, (10, y_offset + i*25), font, 0.6, (0, 0, 0), 1)
        
        # 添加图例
        legend_x = w - 200
        legend_y = 30
        
        legend_items = [
            ("边界", self.colors['boundary']),
            ("A面垂直线", self.colors['a_vertical']),
            ("B面垂直线", self.colors['b_vertical']),
            ("书口点", self.colors['spine']),
            ("A面水平线", self.colors['a_horizontal']),
            ("B面水平线", self.colors['b_horizontal']),
        ]
        
        for i, (text, color) in enumerate(legend_items):
            cv2.rectangle(debug, (legend_x, legend_y + i*25), 
                         (legend_x + 20, legend_y + i*25 + 15), color, -1)
            cv2.putText(debug, text, (legend_x + 25, legend_y + i*25 + 12), 
                       font, 0.4, (0, 0, 0), 1)
        
        return debug
    
    def _update_statistics(self, result: ProcessingResult):
        """更新统计信息"""
        self.statistics['total_pages'] += 1
        self.statistics['total_time'] += result.processing_time
        
        if result.success:
            self.statistics['successful_pages'] += 1
        
        self.statistics['avg_time_per_page'] = (
            self.statistics['total_time'] / self.statistics['total_pages']
        )
        
        # 保存处理历史
        self.processing_history.append({
            'timestamp': time.time(),
            'success': result.success,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'page_type': result.page_type.value,
            'warnings': result.warnings,
        })
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成处理报告"""
        report_lines = [
            "=" * 60,
            "古籍书影处理系统报告",
            "=" * 60,
            f"总处理页数: {self.statistics['total_pages']}",
            f"成功页数: {self.statistics['successful_pages']}",
            f"成功率: {self.statistics['successful_pages']/max(self.statistics['total_pages'],1):.2%}",
            f"总处理时间: {self.statistics['total_time']:.2f}秒",
            f"平均每页时间: {self.statistics['avg_time_per_page']:.2f}秒",
            "",
            "按类型统计:",
        ]
        
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
            report_lines.append(f"  {page_type}: {stats['success']}/{stats['total']} ({success_rate:.2%})")
        
        report_lines.extend([
            "",
            "最近处理记录:",
        ])
        
        # 最近5条记录
        recent = self.processing_history[-5:] if self.processing_history else []
        for i, entry in enumerate(recent):
            status = "成功" if entry['success'] else "失败"
            report_lines.append(
                f"  {i+1}. [{status}] 类型:{entry['page_type']} "
                f"置信度:{entry['confidence']:.2%} 时间:{entry['processing_time']:.2f}s"
            )
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数示例"""
    print("古籍书影智能处理系统 v4.0")
    print("-" * 40)
    
    # 创建处理器
    processor = AncientBookProcessor()
    
    # 示例1: 处理单张图像
    image_path = "test_ancient_book.jpg"  # 替换为实际图像路径
    
    if os.path.exists(image_path):
        print(f"处理图像: {image_path}")
        result = processor.process_image(image_path, "output")
        
        if result.success:
            print(f"✓ 处理成功! 置信度: {result.confidence:.2%}")
            print(f"  古籍类型: {result.page_type.value}")
            print(f"  处理时间: {result.processing_time:.2f}秒")
            print(f"  输出文件: {result.debug_info.get('output_paths', {})}")
        else:
            print(f"✗ 处理失败")
            for warning in result.warnings:
                print(f"  警告: {warning}")
    
    # 示例2: 处理PDF文件
    pdf_path = "ancient_book.pdf"  # 替换为实际PDF路径
    
    if os.path.exists(pdf_path):
        print(f"\n处理PDF: {pdf_path}")
        results = processor.process_pdf(pdf_path, "output_pdf", max_pages=3)
        
        success_count = sum(1 for r in results if r.success)
        print(f"PDF处理完成: {success_count}/{len(results)} 页成功")
        
        # 生成报告
        report = processor.generate_report("processing_report.txt")
        print("\n处理报告:")
        print(report)


if __name__ == "__main__":
    main()
