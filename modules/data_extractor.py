#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据提取模块 - 负责从各种格式的输入中提取数据
支持的格式包括：CSV, Excel, JSON, PDF, 图像中的表格和文字等
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import re
import io
import csv
import tabula
from pdf2image import convert_from_path
import camelot

class DataExtractor:
    """数据提取器类，用于从各种格式的输入中提取数据"""
    
    def __init__(self):
        """初始化数据提取器"""
        self.supported_formats = {
            'csv': self.extract_from_csv,
            'excel': self.extract_from_excel,
            'json': self.extract_from_json,
            'pdf': self.extract_from_pdf,
            'image': self.extract_from_image,
            'text': self.extract_from_text
        }
        
    def detect_format(self, file_path):
        """检测文件格式"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip('.')
        
        format_mapping = {
            'csv': 'csv',
            'xls': 'excel',
            'xlsx': 'excel',
            'json': 'json',
            'pdf': 'pdf',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'gif': 'image',
            'bmp': 'image',
            'txt': 'text',
            'md': 'text'
        }
        
        return format_mapping.get(ext, None)
    
    def extract(self, file_path, **kwargs):
        """根据文件格式调用相应的提取方法"""
        file_format = kwargs.get('format', self.detect_format(file_path))
        
        if file_format not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        return self.supported_formats[file_format](file_path, **kwargs)
    
    def extract_from_csv(self, file_path, **kwargs):
        """从CSV文件中提取数据"""
        encoding = kwargs.get('encoding', 'utf-8')
        delimiter = kwargs.get('delimiter', ',')
        
        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
            return {
                'data': df,
                'metadata': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'format': 'csv'
                },
                'text': None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_from_excel(self, file_path, **kwargs):
        """从Excel文件中提取数据"""
        sheet_name = kwargs.get('sheet_name', 0)
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 如果sheet_name=None，则返回所有工作表的字典
            if isinstance(df, dict):
                return {
                    'data': df,
                    'metadata': {
                        'sheets': list(df.keys()),
                        'format': 'excel'
                    },
                    'text': None
                }
            else:
                return {
                    'data': df,
                    'metadata': {
                        'rows': len(df),
                        'columns': list(df.columns),
                        'format': 'excel'
                    },
                    'text': None
                }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_from_json(self, file_path, **kwargs):
        """从JSON文件中提取数据"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # 尝试将JSON转换为DataFrame（如果是表格数据）
            try:
                df = pd.json_normalize(data)
                return {
                    'data': df,
                    'raw_data': data,
                    'metadata': {
                        'format': 'json',
                        'is_tabular': True
                    },
                    'text': json.dumps(data, ensure_ascii=False, indent=2)
                }
            except:
                return {
                    'data': None,
                    'raw_data': data,
                    'metadata': {
                        'format': 'json',
                        'is_tabular': False
                    },
                    'text': json.dumps(data, ensure_ascii=False, indent=2)
                }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_from_pdf(self, file_path, **kwargs):
        """从PDF文件中提取数据，包括文本和表格"""
        pages = kwargs.get('pages', 'all')
        
        try:
            # 提取文本
            text_result = self._extract_text_from_pdf(file_path, pages)
            
            # 提取表格
            tables_result = self._extract_tables_from_pdf(file_path, pages)
            
            return {
                'data': tables_result.get('data'),
                'text': text_result.get('text'),
                'metadata': {
                    'format': 'pdf',
                    'pages': text_result.get('metadata', {}).get('pages', 0),
                    'has_tables': tables_result.get('metadata', {}).get('has_tables', False)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_text_from_pdf(self, file_path, pages):
        """从PDF中提取文本"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                if pages == 'all':
                    pages = range(num_pages)
                elif isinstance(pages, int):
                    pages = [pages]
                elif isinstance(pages, str):
                    # 解析页码范围，如 "1-5"
                    if '-' in pages:
                        start, end = map(int, pages.split('-'))
                        pages = range(start-1, min(end, num_pages))
                    else:
                        pages = [int(pages) - 1]
                
                text = ""
                for page_num in pages:
                    if 0 <= page_num < num_pages:
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                
                return {
                    'text': text,
                    'metadata': {
                        'pages': num_pages
                    }
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_tables_from_pdf(self, file_path, pages):
        """从PDF中提取表格"""
        try:
            # 使用tabula-py提取表格
            if pages == 'all':
                tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            else:
                if isinstance(pages, int):
                    page_str = str(pages + 1)  # tabula使用1-based索引
                elif isinstance(pages, list):
                    page_str = ','.join(str(p + 1) for p in pages)  # tabula使用1-based索引
                else:
                    page_str = pages
                tables = tabula.read_pdf(file_path, pages=page_str, multiple_tables=True)
            
            # 尝试使用camelot提取更复杂的表格（如果tabula失败）
            if not tables:
                try:
                    if pages == 'all':
                        tables_camelot = camelot.read_pdf(file_path, pages='all')
                    else:
                        if isinstance(pages, int):
                            page_str = str(pages + 1)
                        elif isinstance(pages, list):
                            page_str = ','.join(str(p + 1) for p in pages)
                        else:
                            page_str = pages
                        tables_camelot = camelot.read_pdf(file_path, pages=page_str)
                    
                    tables = [table.df for table in tables_camelot]
                except:
                    pass
            
            return {
                'data': tables,
                'metadata': {
                    'has_tables': len(tables) > 0,
                    'table_count': len(tables)
                }
            }
        except Exception as e:
            return {'error': str(e), 'data': []}
    
    def extract_from_image(self, file_path, **kwargs):
        """从图像中提取数据，包括文本和表格"""
        try:
            # 打开图像
            image = Image.open(file_path)
            
            # 提取文本
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            # 提取表格（使用pytesseract的表格识别功能）
            tables_data = self._extract_tables_from_image(image)
            
            return {
                'data': tables_data.get('data'),
                'text': text,
                'metadata': {
                    'format': 'image',
                    'size': image.size,
                    'mode': image.mode,
                    'has_tables': tables_data.get('metadata', {}).get('has_tables', False)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_tables_from_image(self, image):
        """从图像中提取表格"""
        try:
            # 使用pytesseract的表格识别功能
            tables_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
            
            # 过滤出有意义的行
            tables_data = tables_data[tables_data['text'].str.strip() != '']
            
            # 检查是否有表格结构
            has_tables = False
            table_data = []
            
            # 简单的表格检测逻辑
            if not tables_data.empty:
                # 检查是否有表格边界线
                hocr_data = pytesseract.image_to_pdf_or_hocr(image, extension='hocr')
                if 'table' in str(hocr_data) or 'TR' in str(hocr_data) or 'TD' in str(hocr_data):
                    has_tables = True
                    
                    # 尝试重构表格
                    # 这里只是一个简化的实现，实际应用中可能需要更复杂的算法
                    table_data = self._reconstruct_table_from_ocr(tables_data)
            
            return {
                'data': table_data,
                'metadata': {
                    'has_tables': has_tables
                }
            }
        except Exception as e:
            return {'error': str(e), 'data': []}
    
    def _reconstruct_table_from_ocr(self, ocr_data):
        """从OCR数据中重构表格"""
        # 这是一个简化的实现，实际应用中可能需要更复杂的算法
        try:
            # 按块分组
            blocks = ocr_data.groupby('block_num')
            
            tables = []
            for block_num, block_data in blocks:
                # 按行分组
                lines = block_data.groupby('line_num')
                
                rows = []
                for line_num, line_data in lines:
                    # 按词分组并排序
                    words = line_data.sort_values('left')
                    row = ' '.join(words['text'].tolist())
                    rows.append(row)
                
                # 如果有多行，可能是表格
                if len(rows) > 1:
                    # 尝试将文本分割成列
                    table_rows = []
                    for row in rows:
                        # 使用空格分割（简化处理）
                        cols = row.split()
                        table_rows.append(cols)
                    
                    # 创建DataFrame
                    if table_rows:
                        max_cols = max(len(row) for row in table_rows)
                        # 填充缺失的列
                        padded_rows = [row + [''] * (max_cols - len(row)) for row in table_rows]
                        df = pd.DataFrame(padded_rows)
                        tables.append(df)
            
            return tables
        except Exception as e:
            return []
    
    def extract_from_text(self, file_path, **kwargs):
        """从文本文件中提取数据"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            # 尝试检测和提取文本中的表格（如Markdown表格或制表符分隔的表格）
            tables = self._extract_tables_from_text(text)
            
            return {
                'data': tables,
                'text': text,
                'metadata': {
                    'format': 'text',
                    'has_tables': len(tables) > 0,
                    'size': len(text)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_tables_from_text(self, text):
        """从文本中提取表格（如Markdown表格或制表符分隔的表格）"""
        tables = []
        
        # 检测Markdown表格
        md_tables = self._extract_markdown_tables(text)
        if md_tables:
            tables.extend(md_tables)
        
        # 检测制表符分隔的表格
        tab_tables = self._extract_tab_tables(text)
        if tab_tables:
            tables.extend(tab_tables)
        
        # 检测CSV格式的表格
        csv_tables = self._extract_csv_tables(text)
        if csv_tables:
            tables.extend(csv_tables)
        
        return tables
    
    def _extract_markdown_tables(self, text):
        """提取Markdown格式的表格"""
        # Markdown表格正则表达式
        pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
        
        tables = []
        for match in re.finditer(pattern, text):
            table_text = match.group(1)
            lines = table_text.strip().split('\n')
            
            # 提取表头
            header = lines[0]
            header_cols = [col.strip() for col in header.split('|')[1:-1]]
            
            # 跳过分隔行
            rows = []
            for line in lines[2:]:
                cols = [col.strip() for col in line.split('|')[1:-1]]
                rows.append(cols)
            
            # 创建DataFrame
            df = pd.DataFrame(rows, columns=header_cols)
            tables.append(df)
        
        return tables
    
    def _extract_tab_tables(self, text):
        """提取制表符分隔的表格"""
        # 按行分割
        lines = text.split('\n')
        
        # 查找可能的表格区域（连续的包含制表符的行）
        table_regions = []
        current_region = []
        
        for line in lines:
            if '\t' in line and line.strip():
                current_region.append(line)
            elif current_region:
                if len(current_region) > 1:  # 至少需要两行才能构成表格
                    table_regions.append(current_region)
                current_region = []
        
        # 处理最后一个区域
        if current_region and len(current_region) > 1:
            table_regions.append(current_region)
        
        # 将每个区域转换为DataFrame
        tables = []
        for region in table_regions:
            rows = []
            for line in region:
                cols = line.split('\t')
                rows.append(cols)
            
            # 确保所有行具有相同的列数
            max_cols = max(len(row) for row in rows)
            padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # 使用第一行作为表头
            header = padded_rows[0]
            data = padded_rows[1:]
            
            df = pd.DataFrame(data, columns=header)
            tables.append(df)
        
        return tables
    
    def _extract_csv_tables(self, text):
        """提取CSV格式的表格"""
        # 尝试使用CSV解析器
        try:
            f = io.StringIO(text)
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) > 1:  # 至少需要两行才能构成表格
                header = rows[0]
                data = rows[1:]
                
                df = pd.DataFrame(data, columns=header)
                return [df]
        except:
            pass
        
        return []

# 测试代码
if __name__ == "__main__":
    extractor = DataExtractor()
    # 测试CSV提取
    # result = extractor.extract('sample.csv')
    # print(result)
