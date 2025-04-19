#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据导出模块 - 负责将处理后的数据和可视化结果导出为各种格式
"""

import os
import pandas as pd
import json
import csv
import markdown
import base64
import io
import zipfile
import tempfile
import shutil
from datetime import datetime
import weasyprint
import jinja2
import pdfkit
import matplotlib.pyplot as plt
from PIL import Image

class DataExporter:
    """数据导出器类，用于将数据导出为各种格式"""
    
    def __init__(self):
        """初始化数据导出器"""
        self.data = None
        self.text = None
        self.metadata = {}
        self.visualizations = []
        self.export_history = []
        self.export_dir = 'exports'  # 默认导出目录
        self.template_dir = 'templates'  # 默认模板目录
        
    def load_data(self, data_result):
        """加载数据"""
        if 'error' in data_result:
            raise ValueError(f"数据加载错误: {data_result['error']}")
        
        self.data = data_result.get('data')
        self.metadata = data_result.get('metadata', {})
        self.text = data_result.get('text')
        
        self._add_to_history("数据加载完成")
        
        return self
    
    def add_visualization(self, file_path, description=None):
        """添加可视化结果"""
        if not os.path.exists(file_path):
            raise ValueError(f"可视化文件不存在: {file_path}")
        
        self.visualizations.append({
            'file_path': file_path,
            'description': description or os.path.basename(file_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        self._add_to_history("添加可视化结果", {'file_path': file_path, 'description': description})
        
        return self
    
    def _add_to_history(self, operation, details=None, file_path=None):
        """添加导出历史记录"""
        self.export_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'details': details,
            'file_path': file_path
        })
    
    def get_export_history(self):
        """获取导出历史记录"""
        return self.export_history
    
    def set_export_dir(self, export_dir):
        """设置导出目录"""
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        self.export_dir = export_dir
        self._add_to_history("设置导出目录", {'directory': export_dir})
        
        return self
    
    def set_template_dir(self, template_dir):
        """设置模板目录"""
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
        
        self.template_dir = template_dir
        self._add_to_history("设置模板目录", {'directory': template_dir})
        
        return self
    
    def _prepare_dataframe(self, table_index=None):
        """准备用于导出的DataFrame"""
        if table_index is not None:
            if isinstance(self.data, list) and 0 <= table_index < len(self.data):
                return self.data[table_index]
            else:
                raise ValueError(f"无效的表格索引: {table_index}")
        elif isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], pd.DataFrame):
            return self.data[0]  # 默认使用第一个表格
        else:
            raise ValueError("数据格式不支持导出操作")
    
    def export_to_csv(self, table_index=None, **kwargs):
        """导出数据为CSV格式"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        encoding = kwargs.get('encoding', 'utf-8')
        index = kwargs.get('index', False)
        sep = kwargs.get('sep', ',')
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.csv")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 导出为CSV
        df.to_csv(file_path, encoding=encoding, index=index, sep=sep)
        
        self._add_to_history("导出CSV", {'table_index': table_index, 'encoding': encoding}, file_path)
        
        return file_path
    
    def export_to_excel(self, table_indices=None, **kwargs):
        """导出数据为Excel格式"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        sheet_names = kwargs.get('sheet_names', None)
        index = kwargs.get('index', False)
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.xlsx")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 创建Excel写入器
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if isinstance(self.data, pd.DataFrame):
                # 单个DataFrame
                sheet_name = sheet_names[0] if sheet_names and len(sheet_names) > 0 else 'Sheet1'
                self.data.to_excel(writer, sheet_name=sheet_name, index=index)
                
            elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
                # DataFrame列表
                if table_indices is None:
                    # 导出所有表格
                    table_indices = list(range(len(self.data)))
                
                for i, idx in enumerate(table_indices):
                    if 0 <= idx < len(self.data):
                        sheet_name = sheet_names[i] if sheet_names and i < len(sheet_names) else f'Sheet{idx+1}'
                        self.data[idx].to_excel(writer, sheet_name=sheet_name, index=index)
            else:
                raise ValueError("数据格式不支持导出为Excel")
        
        self._add_to_history("导出Excel", {'table_indices': table_indices}, file_path)
        
        return file_path
    
    def export_to_json(self, table_index=None, **kwargs):
        """导出数据为JSON格式"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        orient = kwargs.get('orient', 'records')
        indent = kwargs.get('indent', 2)
        date_format = kwargs.get('date_format', 'iso')
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.json")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if table_index is not None:
            # 导出指定表格
            df = self._prepare_dataframe(table_index)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(df.to_json(orient=orient, date_format=date_format, indent=indent))
                
        elif isinstance(self.data, pd.DataFrame):
            # 导出单个DataFrame
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.data.to_json(orient=orient, date_format=date_format, indent=indent))
                
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            # 导出所有表格
            result = []
            for i, df in enumerate(self.data):
                result.append({
                    'table_index': i,
                    'data': json.loads(df.to_json(orient=orient, date_format=date_format))
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=indent)
                
        elif self.text:
            # 导出文本
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({'text': self.text}, f, indent=indent)
                
        else:
            raise ValueError("数据格式不支持导出为JSON")
        
        self._add_to_history("导出JSON", {'table_index': table_index, 'orient': orient}, file_path)
        
        return file_path
    
    def export_to_html(self, table_indices=None, **kwargs):
        """导出数据为HTML格式"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = kwargs.get('title', "数据分析报告")
        include_visualizations = kwargs.get('include_visualizations', True)
        include_text = kwargs.get('include_text', True)
        template = kwargs.get('template', None)
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.html")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备数据表格HTML
        tables_html = []
        
        if isinstance(self.data, pd.DataFrame):
            # 单个DataFrame
            tables_html.append(self.data.to_html(index=False, classes='dataframe'))
            
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            # DataFrame列表
            if table_indices is None:
                # 导出所有表格
                table_indices = list(range(len(self.data)))
            
            for idx in table_indices:
                if 0 <= idx < len(self.data):
                    tables_html.append(f"<h3>表格 {idx+1}</h3>")
                    tables_html.append(self.data[idx].to_html(index=False, classes='dataframe'))
        
        # 准备可视化图表HTML
        visualizations_html = []
        if include_visualizations and self.visualizations:
            for viz in self.visualizations:
                file_path = viz['file_path']
                description = viz['description']
                
                # 检查文件类型
                _, ext = os.path.splitext(file_path)
                if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                    # 图像文件
                    with open(file_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    img_tag = f'<img src="data:image/{ext[1:]};base64,{img_data}" alt="{description}" style="max-width:100%;">'
                    visualizations_html.append(f'<div class="visualization"><h4>{description}</h4>{img_tag}</div>')
                    
                elif ext.lower() == '.html':
                    # HTML文件（如交互式图表）
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # 提取<body>内容
                    import re
                    body_match = re.search(r'<body>(.*?)</body>', html_content, re.DOTALL)
                    if body_match:
                        body_content = body_match.group(1)
                        visualizations_html.append(f'<div class="visualization"><h4>{description}</h4>{body_content}</div>')
                    else:
                        visualizations_html.append(f'<div class="visualization"><h4>{description}</h4><iframe src="{file_path}" width="100%" height="500px"></iframe></div>')
        
        # 准备文本HTML
        text_html = ""
        if include_text and self.text:
            text_html = f'<div class="text-content"><h3>文本内容</h3><pre>{self.text}</pre></div>'
        
        # 使用模板或生成默认HTML
        if template and os.path.exists(os.path.join(self.template_dir, template)):
            # 使用Jinja2模板
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
            template = env.get_template(template)
            
            html_content = template.render(
                title=title,
                tables_html=tables_html,
                visualizations_html=visualizations_html,
                text_html=text_html,
                metadata=self.metadata,
                generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            # 生成默认HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3, h4 {{ color: #444; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .dataframe {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .dataframe th {{ background-color: #f2f2f2; }}
                    .dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .visualization {{ margin-bottom: 30px; }}
                    .text-content {{ margin-bottom: 30px; }}
                    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    .footer {{ margin-top: 50px; text-align: center; font-size: 0.8em; color: #777; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    
                    <!-- 数据表格 -->
                    <div class="tables-section">
                        <h2>数据表格</h2>
                        {''.join(tables_html) if tables_html else '<p>无表格数据</p>'}
                    </div>
                    
                    <!-- 可视化图表 -->
                    <div class="visualizations-section">
                        <h2>数据可视化</h2>
                        {''.join(visualizations_html) if visualizations_html else '<p>无可视化图表</p>'}
                    </div>
                    
                    <!-- 文本内容 -->
                    {text_html}
                    
                    <div class="footer">
                        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        # 写入HTML文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._add_to_history("导出HTML", {
            'table_indices': table_indices,
            'include_visualizations': include_visualizations,
            'include_text': include_text
        }, file_path)
        
        return file_path
    
    def export_to_pdf(self, table_indices=None, **kwargs):
        """导出数据为PDF格式"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = kwargs.get('title', "数据分析报告")
        include_visualizations = kwargs.get('include_visualizations', True)
        include_text = kwargs.get('include_text', True)
        template = kwargs.get('template', None)
        method = kwargs.get('method', 'weasyprint')  # 'weasyprint' 或 'wkhtmltopdf'
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.pdf")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 首先导出为HTML
        html_file = self.export_to_html(
            table_indices=table_indices,
            file_name=f"{file_name}_temp",
            title=title,
            include_visualizations=include_visualizations,
            include_text=include_text,
            template=template
        )
        
        # 将HTML转换为PDF
        if method == 'weasyprint':
            # 使用WeasyPrint
            try:
                weasyprint.HTML(html_file).write_pdf(file_path)
            except Exception as e:
                raise ValueError(f"PDF导出失败: {str(e)}")
                
        elif method == 'wkhtmltopdf':
            # 使用wkhtmltopdf
            try:
                pdfkit.from_file(html_file, file_path)
            except Exception as e:
                raise ValueError(f"PDF导出失败: {str(e)}")
        else:
            raise ValueError(f"不支持的PDF导出方法: {method}")
        
        # 删除临时HTML文件
        try:
            os.remove(html_file)
        except:
            pass
        
        self._add_to_history("导出PDF", {
            'table_indices': table_indices,
            'include_visualizations': include_visualizations,
            'include_text': include_text,
            'method': method
        }, file_path)
        
        return file_path
    
    def export_to_markdown(self, table_indices=None, **kwargs):
        """导出数据为Markdown格式"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = kwargs.get('title', "数据分析报告")
        include_visualizations = kwargs.get('include_visualizations', True)
        include_text = kwargs.get('include_text', True)
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.md")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备Markdown内容
        md_content = [f"# {title}\n\n"]
        
        # 添加数据表格
        md_content.append("## 数据表格\n\n")
        
        if isinstance(self.data, pd.DataFrame):
            # 单个DataFrame
            md_content.append(self._dataframe_to_markdown(self.data))
            
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            # DataFrame列表
            if table_indices is None:
                # 导出所有表格
                table_indices = list(range(len(self.data)))
            
            for idx in table_indices:
                if 0 <= idx < len(self.data):
                    md_content.append(f"### 表格 {idx+1}\n\n")
                    md_content.append(self._dataframe_to_markdown(self.data[idx]))
        
        # 添加可视化图表
        if include_visualizations and self.visualizations:
            md_content.append("\n## 数据可视化\n\n")
            
            # 创建可视化图片目录
            viz_dir = os.path.join(os.path.dirname(file_path), f"{file_name}_files")
            os.makedirs(viz_dir, exist_ok=True)
            
            for i, viz in enumerate(self.visualizations):
                file_path_src = viz['file_path']
                description = viz['description']
                
                # 检查文件类型
                _, ext = os.path.splitext(file_path_src)
                if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                    # 复制图像文件
                    viz_file_name = f"visualization_{i+1}{ext}"
                    viz_file_path = os.path.join(viz_dir, viz_file_name)
                    shutil.copy2(file_path_src, viz_file_path)
                    
                    # 添加到Markdown
                    rel_path = os.path.join(f"{file_name}_files", viz_file_name)
                    md_content.append(f"### {description}\n\n")
                    md_content.append(f"![{description}]({rel_path})\n\n")
        
        # 添加文本内容
        if include_text and self.text:
            md_content.append("\n## 文本内容\n\n")
            md_content.append("```\n")
            md_content.append(self.text)
            md_content.append("\n```\n\n")
        
        # 添加生成时间
        md_content.append(f"\n---\n\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # 写入Markdown文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md_content))
        
        self._add_to_history("导出Markdown", {
            'table_indices': table_indices,
            'include_visualizations': include_visualizations,
            'include_text': include_text
        }, file_path)
        
        return file_path
    
    def _dataframe_to_markdown(self, df, max_rows=100, max_cols=20):
        """将DataFrame转换为Markdown表格"""
        if df.empty:
            return "*空表格*\n\n"
        
        # 限制行数和列数
        if len(df) > max_rows:
            df = pd.concat([df.head(max_rows//2), df.tail(max_rows//2)])
        
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        # 转换为Markdown
        md_table = []
        
        # 添加表头
        header = "| " + " | ".join(str(col) for col in df.columns) + " |"
        md_table.append(header)
        
        # 添加分隔行
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        md_table.append(separator)
        
        # 添加数据行
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
            md_table.append(row_str)
        
        return "\n".join(md_table) + "\n\n"
    
    def export_to_image(self, table_index=None, **kwargs):
        """将数据表格导出为图像"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        file_name = kwargs.get('file_name', f"table_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        format = kwargs.get('format', 'png')
        dpi = kwargs.get('dpi', 100)
        max_rows = kwargs.get('max_rows', 50)
        max_cols = kwargs.get('max_cols', 15)
        
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.{format}")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 限制行数和列数
        if len(df) > max_rows:
            df = pd.concat([df.head(max_rows//2), df.tail(max_rows//2)])
        
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        # 创建表格图像
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center'
        )
        
        # 调整表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        # 保存图像
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)
        
        self._add_to_history("导出表格图像", {'table_index': table_index, 'format': format}, file_path)
        
        return file_path
    
    def export_all_to_zip(self, formats=None, **kwargs):
        """将数据导出为多种格式并打包为ZIP文件"""
        # 获取参数
        file_name = kwargs.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = kwargs.get('title', "数据分析报告")
        include_visualizations = kwargs.get('include_visualizations', True)
        include_text = kwargs.get('include_text', True)
        
        # 如果未指定格式，则使用所有支持的格式
        if formats is None:
            formats = ['csv', 'excel', 'json', 'html', 'pdf', 'markdown']
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 导出各种格式
            exported_files = []
            
            # 设置临时导出目录
            original_export_dir = self.export_dir
            self.export_dir = temp_dir
            
            try:
                # 导出各种格式
                for fmt in formats:
                    if fmt == 'csv':
                        if isinstance(self.data, pd.DataFrame):
                            exported_files.append(self.export_to_csv(file_name=f"{file_name}_csv"))
                        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
                            for i, df in enumerate(self.data):
                                temp_exporter = DataExporter()
                                temp_exporter.data = df
                                temp_exporter.export_dir = temp_dir
                                exported_files.append(temp_exporter.export_to_csv(file_name=f"{file_name}_table{i+1}_csv"))
                    
                    elif fmt == 'excel':
                        exported_files.append(self.export_to_excel(file_name=f"{file_name}_excel"))
                    
                    elif fmt == 'json':
                        exported_files.append(self.export_to_json(file_name=f"{file_name}_json"))
                    
                    elif fmt == 'html':
                        exported_files.append(self.export_to_html(
                            file_name=f"{file_name}_html",
                            title=title,
                            include_visualizations=include_visualizations,
                            include_text=include_text
                        ))
                    
                    elif fmt == 'pdf':
                        try:
                            exported_files.append(self.export_to_pdf(
                                file_name=f"{file_name}_pdf",
                                title=title,
                                include_visualizations=include_visualizations,
                                include_text=include_text
                            ))
                        except Exception as e:
                            print(f"PDF导出失败: {str(e)}")
                    
                    elif fmt == 'markdown':
                        exported_files.append(self.export_to_markdown(
                            file_name=f"{file_name}_markdown",
                            title=title,
                            include_visualizations=include_visualizations,
                            include_text=include_text
                        ))
                
                # 添加可视化文件
                if include_visualizations and self.visualizations:
                    for viz in self.visualizations:
                        viz_path = viz['file_path']
                        if os.path.exists(viz_path):
                            viz_name = os.path.basename(viz_path)
                            viz_dest = os.path.join(temp_dir, 'visualizations', viz_name)
                            os.makedirs(os.path.dirname(viz_dest), exist_ok=True)
                            shutil.copy2(viz_path, viz_dest)
                            exported_files.append(viz_dest)
                
                # 创建ZIP文件
                zip_path = os.path.join(original_export_dir, f"{file_name}.zip")
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in exported_files:
                        if os.path.exists(file):
                            # 获取相对路径
                            rel_path = os.path.relpath(file, temp_dir)
                            zipf.write(file, rel_path)
                
                # 恢复原始导出目录
                self.export_dir = original_export_dir
                
                self._add_to_history("导出ZIP", {'formats': formats}, zip_path)
                
                return zip_path
                
            finally:
                # 确保恢复原始导出目录
                self.export_dir = original_export_dir
    
    def create_report(self, **kwargs):
        """创建综合分析报告"""
        # 获取参数
        file_name = kwargs.get('file_name', f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        title = kwargs.get('title', "数据分析报告")
        author = kwargs.get('author', "数据分析系统")
        format = kwargs.get('format', 'html')  # 'html', 'pdf', 'markdown'
        include_toc = kwargs.get('include_toc', True)
        include_summary = kwargs.get('include_summary', True)
        include_visualizations = kwargs.get('include_visualizations', True)
        include_tables = kwargs.get('include_tables', True)
        include_text = kwargs.get('include_text', True)
        include_metadata = kwargs.get('include_metadata', True)
        template = kwargs.get('template', None)
        insights = kwargs.get('insights', None)
        
        # 准备报告内容
        report_sections = []
        
        # 添加标题
        report_sections.append({
            'type': 'title',
            'content': title,
            'level': 1
        })
        
        # 添加元数据
        if include_metadata:
            metadata_content = [
                f"- **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- **作者**: {author}"
            ]
            
            if self.metadata:
                for key, value in self.metadata.items():
                    if isinstance(value, dict):
                        continue  # 跳过嵌套字典
                    metadata_content.append(f"- **{key}**: {value}")
            
            report_sections.append({
                'type': 'metadata',
                'content': '\n'.join(metadata_content),
                'level': 2
            })
        
        # 添加摘要
        if include_summary and insights:
            summary_content = ["### 主要发现\n"]
            
            if 'summary' in insights and insights['summary']:
                for item in insights['summary']:
                    if isinstance(item, dict) and 'content' in item:
                        summary_content.append(f"- {item['content']}")
                    else:
                        summary_content.append(f"- {item}")
            
            summary_content.append("\n### 模式和趋势\n")
            
            if 'patterns' in insights and insights['patterns']:
                for item in insights['patterns']:
                    if isinstance(item, dict) and 'content' in item:
                        summary_content.append(f"- {item['content']}")
                    else:
                        summary_content.append(f"- {item}")
            
            summary_content.append("\n### 异常和问题\n")
            
            if 'anomalies' in insights and insights['anomalies']:
                for item in insights['anomalies']:
                    if isinstance(item, dict) and 'content' in item:
                        summary_content.append(f"- {item['content']}")
                    else:
                        summary_content.append(f"- {item}")
            
            summary_content.append("\n### 建议\n")
            
            if 'recommendations' in insights and insights['recommendations']:
                for item in insights['recommendations']:
                    if isinstance(item, dict) and 'content' in item:
                        summary_content.append(f"- {item['content']}")
                    else:
                        summary_content.append(f"- {item}")
            
            report_sections.append({
                'type': 'summary',
                'content': '\n'.join(summary_content),
                'level': 2
            })
        
        # 添加可视化
        if include_visualizations and self.visualizations:
            viz_content = []
            
            for viz in self.visualizations:
                file_path = viz['file_path']
                description = viz['description']
                
                viz_content.append({
                    'file_path': file_path,
                    'description': description
                })
            
            report_sections.append({
                'type': 'visualizations',
                'content': viz_content,
                'level': 2
            })
        
        # 添加数据表格
        if include_tables:
            tables_content = []
            
            if isinstance(self.data, pd.DataFrame):
                # 单个DataFrame
                tables_content.append({
                    'index': 0,
                    'data': self.data
                })
                
            elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
                # DataFrame列表
                for i, df in enumerate(self.data):
                    tables_content.append({
                        'index': i,
                        'data': df
                    })
            
            report_sections.append({
                'type': 'tables',
                'content': tables_content,
                'level': 2
            })
        
        # 添加文本内容
        if include_text and self.text:
            report_sections.append({
                'type': 'text',
                'content': self.text,
                'level': 2
            })
        
        # 根据格式生成报告
        if format == 'html':
            return self._create_html_report(report_sections, file_name, template, include_toc)
        elif format == 'pdf':
            return self._create_pdf_report(report_sections, file_name, template, include_toc)
        elif format == 'markdown':
            return self._create_markdown_report(report_sections, file_name, include_toc)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
    
    def _create_html_report(self, sections, file_name, template=None, include_toc=True):
        """创建HTML格式的报告"""
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.html")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 创建可视化图片目录
        viz_dir = os.path.join(os.path.dirname(file_path), f"{file_name}_files")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 准备HTML内容
        if template and os.path.exists(os.path.join(self.template_dir, template)):
            # 使用Jinja2模板
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_dir))
            template = env.get_template(template)
            
            # 处理各部分内容
            processed_sections = self._process_sections_for_html(sections, viz_dir, f"{file_name}_files")
            
            html_content = template.render(
                sections=processed_sections,
                include_toc=include_toc,
                generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            # 生成默认HTML
            html_parts = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '    <meta charset="UTF-8">',
                '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
                f'    <title>{sections[0]["content"] if sections and sections[0]["type"] == "title" else "数据分析报告"}</title>',
                '    <style>',
                '        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }',
                '        h1, h2, h3, h4 { color: #444; }',
                '        .container { max-width: 1200px; margin: 0 auto; }',
                '        .toc { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 30px; }',
                '        .toc ul { list-style-type: none; padding-left: 20px; }',
                '        .toc a { text-decoration: none; color: #0066cc; }',
                '        .toc a:hover { text-decoration: underline; }',
                '        .section { margin-bottom: 30px; }',
                '        .dataframe { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
                '        .dataframe th, .dataframe td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                '        .dataframe th { background-color: #f2f2f2; }',
                '        .dataframe tr:nth-child(even) { background-color: #f9f9f9; }',
                '        .visualization { margin-bottom: 30px; }',
                '        .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }',
                '        pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }',
                '        .footer { margin-top: 50px; text-align: center; font-size: 0.8em; color: #777; }',
                '    </style>',
                '</head>',
                '<body>',
                '    <div class="container">'
            ]
            
            # 添加目录
            if include_toc:
                html_parts.append('        <div class="toc">')
                html_parts.append('            <h2>目录</h2>')
                html_parts.append('            <ul>')
                
                for section in sections:
                    if section['type'] == 'title':
                        continue  # 跳过标题
                    
                    section_id = section['type']
                    section_title = {
                        'metadata': '元数据',
                        'summary': '摘要',
                        'visualizations': '数据可视化',
                        'tables': '数据表格',
                        'text': '文本内容'
                    }.get(section['type'], section['type'])
                    
                    html_parts.append(f'                <li><a href="#{section_id}">{section_title}</a></li>')
                
                html_parts.append('            </ul>')
                html_parts.append('        </div>')
            
            # 处理各部分内容
            for section in sections:
                if section['type'] == 'title':
                    html_parts.append(f'        <h1>{section["content"]}</h1>')
                    continue
                
                section_id = section['type']
                section_title = {
                    'metadata': '元数据',
                    'summary': '摘要',
                    'visualizations': '数据可视化',
                    'tables': '数据表格',
                    'text': '文本内容'
                }.get(section['type'], section['type'])
                
                html_parts.append(f'        <div id="{section_id}" class="section">')
                html_parts.append(f'            <h{section["level"]}>{section_title}</h{section["level"]}>')
                
                if section['type'] == 'metadata' or section['type'] == 'summary':
                    # Markdown内容转HTML
                    html_parts.append(f'            <div>{markdown.markdown(section["content"])}</div>')
                    
                elif section['type'] == 'visualizations':
                    for i, viz in enumerate(section['content']):
                        file_path = viz['file_path']
                        description = viz['description']
                        
                        # 检查文件类型
                        _, ext = os.path.splitext(file_path)
                        if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                            # 复制图像文件
                            viz_file_name = f"visualization_{i+1}{ext}"
                            viz_file_path = os.path.join(viz_dir, viz_file_name)
                            shutil.copy2(file_path, viz_file_path)
                            
                            # 添加到HTML
                            rel_path = os.path.join(f"{file_name}_files", viz_file_name)
                            html_parts.append(f'            <div class="visualization">')
                            html_parts.append(f'                <h3>{description}</h3>')
                            html_parts.append(f'                <img src="{rel_path}" alt="{description}">')
                            html_parts.append(f'            </div>')
                            
                        elif ext.lower() == '.html':
                            # HTML文件（如交互式图表）
                            with open(file_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            # 提取<body>内容
                            import re
                            body_match = re.search(r'<body>(.*?)</body>', html_content, re.DOTALL)
                            
                            html_parts.append(f'            <div class="visualization">')
                            html_parts.append(f'                <h3>{description}</h3>')
                            
                            if body_match:
                                body_content = body_match.group(1)
                                html_parts.append(f'                <div>{body_content}</div>')
                            else:
                                # 复制HTML文件
                                viz_file_name = f"visualization_{i+1}.html"
                                viz_file_path = os.path.join(viz_dir, viz_file_name)
                                shutil.copy2(file_path, viz_file_path)
                                
                                rel_path = os.path.join(f"{file_name}_files", viz_file_name)
                                html_parts.append(f'                <iframe src="{rel_path}" width="100%" height="500px"></iframe>')
                            
                            html_parts.append(f'            </div>')
                    
                elif section['type'] == 'tables':
                    for table in section['content']:
                        index = table['index']
                        df = table['data']
                        
                        html_parts.append(f'            <h3>表格 {index+1}</h3>')
                        html_parts.append(f'            {df.to_html(index=False, classes="dataframe")}')
                    
                elif section['type'] == 'text':
                    html_parts.append(f'            <pre>{section["content"]}</pre>')
                
                html_parts.append('        </div>')
            
            # 添加页脚
            html_parts.append('        <div class="footer">')
            html_parts.append(f'            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
            html_parts.append('        </div>')
            
            html_parts.append('    </div>')
            html_parts.append('</body>')
            html_parts.append('</html>')
            
            html_content = '\n'.join(html_parts)
        
        # 写入HTML文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._add_to_history("创建HTML报告", {'file_name': file_name, 'include_toc': include_toc}, file_path)
        
        return file_path
    
    def _process_sections_for_html(self, sections, viz_dir, viz_rel_dir):
        """处理报告各部分内容为HTML格式"""
        processed_sections = []
        
        for section in sections:
            processed_section = section.copy()
            
            if section['type'] == 'visualizations':
                processed_viz = []
                
                for i, viz in enumerate(section['content']):
                    file_path = viz['file_path']
                    description = viz['description']
                    
                    # 检查文件类型
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                        # 复制图像文件
                        viz_file_name = f"visualization_{i+1}{ext}"
                        viz_file_path = os.path.join(viz_dir, viz_file_name)
                        shutil.copy2(file_path, viz_file_path)
                        
                        # 添加到处理后的可视化
                        processed_viz.append({
                            'type': 'image',
                            'file_path': os.path.join(viz_rel_dir, viz_file_name),
                            'description': description
                        })
                        
                    elif ext.lower() == '.html':
                        # HTML文件（如交互式图表）
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # 提取<body>内容
                        import re
                        body_match = re.search(r'<body>(.*?)</body>', html_content, re.DOTALL)
                        
                        if body_match:
                            body_content = body_match.group(1)
                            processed_viz.append({
                                'type': 'html',
                                'content': body_content,
                                'description': description
                            })
                        else:
                            # 复制HTML文件
                            viz_file_name = f"visualization_{i+1}.html"
                            viz_file_path = os.path.join(viz_dir, viz_file_name)
                            shutil.copy2(file_path, viz_file_path)
                            
                            processed_viz.append({
                                'type': 'iframe',
                                'file_path': os.path.join(viz_rel_dir, viz_file_name),
                                'description': description
                            })
                
                processed_section['content'] = processed_viz
                
            elif section['type'] == 'tables':
                processed_tables = []
                
                for table in section['content']:
                    index = table['index']
                    df = table['data']
                    
                    processed_tables.append({
                        'index': index,
                        'html': df.to_html(index=False, classes="dataframe")
                    })
                
                processed_section['content'] = processed_tables
                
            elif section['type'] == 'metadata' or section['type'] == 'summary':
                # Markdown内容转HTML
                processed_section['html'] = markdown.markdown(section['content'])
            
            processed_sections.append(processed_section)
        
        return processed_sections
    
    def _create_pdf_report(self, sections, file_name, template=None, include_toc=True):
        """创建PDF格式的报告"""
        # 首先创建HTML报告
        html_file = self._create_html_report(sections, f"{file_name}_temp", template, include_toc)
        
        # 构建PDF文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.pdf")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 将HTML转换为PDF
        try:
            weasyprint.HTML(html_file).write_pdf(file_path)
        except Exception as e:
            try:
                # 尝试使用wkhtmltopdf
                pdfkit.from_file(html_file, file_path)
            except Exception as e2:
                raise ValueError(f"PDF导出失败: {str(e)}, 尝试wkhtmltopdf也失败: {str(e2)}")
        
        # 删除临时HTML文件
        try:
            os.remove(html_file)
            # 删除临时图片目录
            temp_viz_dir = os.path.join(os.path.dirname(html_file), f"{file_name}_temp_files")
            if os.path.exists(temp_viz_dir):
                shutil.rmtree(temp_viz_dir)
        except:
            pass
        
        self._add_to_history("创建PDF报告", {'file_name': file_name, 'include_toc': include_toc}, file_path)
        
        return file_path
    
    def _create_markdown_report(self, sections, file_name, include_toc=True):
        """创建Markdown格式的报告"""
        # 构建文件路径
        file_path = os.path.join(self.export_dir, f"{file_name}.md")
        
        # 确保导出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 创建可视化图片目录
        viz_dir = os.path.join(os.path.dirname(file_path), f"{file_name}_files")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 准备Markdown内容
        md_parts = []
        
        # 添加标题
        title_section = next((s for s in sections if s['type'] == 'title'), None)
        if title_section:
            md_parts.append(f"# {title_section['content']}\n\n")
        else:
            md_parts.append("# 数据分析报告\n\n")
        
        # 添加目录
        if include_toc:
            md_parts.append("## 目录\n\n")
            
            for section in sections:
                if section['type'] == 'title':
                    continue  # 跳过标题
                
                section_title = {
                    'metadata': '元数据',
                    'summary': '摘要',
                    'visualizations': '数据可视化',
                    'tables': '数据表格',
                    'text': '文本内容'
                }.get(section['type'], section['type'])
                
                md_parts.append(f"- [{section_title}](#{section['type']})\n")
            
            md_parts.append("\n")
        
        # 处理各部分内容
        for section in sections:
            if section['type'] == 'title':
                continue  # 已经处理过标题
            
            section_title = {
                'metadata': '元数据',
                'summary': '摘要',
                'visualizations': '数据可视化',
                'tables': '数据表格',
                'text': '文本内容'
            }.get(section['type'], section['type'])
            
            md_parts.append(f"<a id='{section['type']}'></a>\n\n")
            md_parts.append(f"{'#' * section['level']} {section_title}\n\n")
            
            if section['type'] == 'metadata' or section['type'] == 'summary':
                md_parts.append(f"{section['content']}\n\n")
                
            elif section['type'] == 'visualizations':
                for i, viz in enumerate(section['content']):
                    file_path_src = viz['file_path']
                    description = viz['description']
                    
                    # 检查文件类型
                    _, ext = os.path.splitext(file_path_src)
                    if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                        # 复制图像文件
                        viz_file_name = f"visualization_{i+1}{ext}"
                        viz_file_path = os.path.join(viz_dir, viz_file_name)
                        shutil.copy2(file_path_src, viz_file_path)
                        
                        # 添加到Markdown
                        rel_path = os.path.join(f"{file_name}_files", viz_file_name)
                        md_parts.append(f"### {description}\n\n")
                        md_parts.append(f"![{description}]({rel_path})\n\n")
                    
                    elif ext.lower() == '.html':
                        # HTML文件（如交互式图表）
                        # 复制HTML文件
                        viz_file_name = f"visualization_{i+1}.html"
                        viz_file_path = os.path.join(viz_dir, viz_file_name)
                        shutil.copy2(file_path_src, viz_file_path)
                        
                        # 添加到Markdown（作为链接）
                        rel_path = os.path.join(f"{file_name}_files", viz_file_name)
                        md_parts.append(f"### {description}\n\n")
                        md_parts.append(f"[查看交互式图表]({rel_path})\n\n")
                
            elif section['type'] == 'tables':
                for table in section['content']:
                    index = table['index']
                    df = table['data']
                    
                    md_parts.append(f"### 表格 {index+1}\n\n")
                    md_parts.append(self._dataframe_to_markdown(df))
                
            elif section['type'] == 'text':
                md_parts.append("```\n")
                md_parts.append(section['content'])
                md_parts.append("\n```\n\n")
        
        # 添加生成时间
        md_parts.append(f"\n---\n\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # 写入Markdown文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md_parts))
        
        self._add_to_history("创建Markdown报告", {'file_name': file_name, 'include_toc': include_toc}, file_path)
        
        return file_path

# 测试代码
if __name__ == "__main__":
    exporter = DataExporter()
    # 测试导出
    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # exporter.load_data({'data': df})
    # exporter.export_to_csv()
