#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据分析处理器 - Web应用主程序
"""

import os
import sys
import uuid
import json
import tempfile
import shutil
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.data_extractor import DataExtractor
from modules.data_processor import DataProcessor
from modules.data_visualizer import DataVisualizer
from modules.data_exporter import DataExporter

# 创建Flask应用
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 限制上传文件大小为50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
app.config['EXPORT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'json', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

# 会话数据存储
session_data = {}

def allowed_file(filename):
    """检查文件扩展名是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_session_folder(session_id):
    """获取会话的工作目录"""
    folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def get_export_folder(session_id):
    """获取会话的导出目录"""
    folder = os.path.join(app.config['EXPORT_FOLDER'], session_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def get_file_info(file_path):
    """获取文件信息"""
    file_info = {
        'name': os.path.basename(file_path),
        'path': file_path,
        'size': os.path.getsize(file_path),
        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 获取文件类型
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext in ['.csv', '.xlsx', '.xls']:
        file_info['type'] = 'table'
    elif ext in ['.json', '.txt']:
        file_info['type'] = 'text'
    elif ext == '.pdf':
        file_info['type'] = 'document'
    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
        file_info['type'] = 'image'
    else:
        file_info['type'] = 'unknown'
    
    return file_info

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

@app.route('/')
def index():
    """首页"""
    # 生成会话ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    # 初始化会话数据
    if session_id not in session_data:
        session_data[session_id] = {
            'files': [],
            'extracted_data': None,
            'processed_data': None,
            'visualizations': [],
            'exports': []
        }
    
    # 获取会话目录中的文件
    session_folder = get_session_folder(session_id)
    files = []
    
    if os.path.exists(session_folder):
        for filename in os.listdir(session_folder):
            file_path = os.path.join(session_folder, filename)
            if os.path.isfile(file_path):
                file_info = get_file_info(file_path)
                file_info['size_formatted'] = format_size(file_info['size'])
                files.append(file_info)
    
    # 获取导出文件
    export_folder = get_export_folder(session_id)
    exports = []
    
    if os.path.exists(export_folder):
        for filename in os.listdir(export_folder):
            file_path = os.path.join(export_folder, filename)
            if os.path.isfile(file_path):
                file_info = get_file_info(file_path)
                file_info['size_formatted'] = format_size(file_info['size'])
                exports.append(file_info)
    
    # 更新会话数据
    session_data[session_id]['files'] = files
    session_data[session_id]['exports'] = exports
    
    return render_template('index.html', 
                          session_id=session_id, 
                          files=files, 
                          exports=exports,
                          extracted_data=session_data[session_id]['extracted_data'] is not None,
                          processed_data=session_data[session_id]['processed_data'] is not None,
                          visualizations=session_data[session_id]['visualizations'])

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        flash('没有选择文件', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('没有选择文件', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        session_id = session.get('session_id', str(uuid.uuid4()))
        session_folder = get_session_folder(session_id)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_folder, filename)
        
        file.save(file_path)
        flash(f'文件 {filename} 上传成功', 'success')
    else:
        flash('不支持的文件类型', 'error')
    
    return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract_data():
    """提取数据"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期，请重新上传文件', 'error')
        return redirect(url_for('index'))
    
    file_path = request.form.get('file_path')
    if not file_path or not os.path.exists(file_path):
        flash('文件不存在', 'error')
        return redirect(url_for('index'))
    
    try:
        # 创建数据提取器
        extractor = DataExtractor()
        
        # 提取数据
        extraction_result = extractor.extract(file_path)
        
        # 保存提取结果到会话数据
        session_data[session_id]['extracted_data'] = extraction_result
        
        flash('数据提取成功', 'success')
    except Exception as e:
        flash(f'数据提取失败: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/process', methods=['GET', 'POST'])
def process_data():
    """处理数据"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期，请重新上传文件', 'error')
        return redirect(url_for('index'))
    
    # 检查是否有提取的数据
    if session_id not in session_data or not session_data[session_id]['extracted_data']:
        flash('请先提取数据', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # 获取处理参数
        table_index = request.form.get('table_index')
        if table_index:
            table_index = int(table_index)
        
        # 清洗参数
        clean_params = {
            'clean_column_names': 'clean_column_names' in request.form,
            'drop_duplicates': 'drop_duplicates' in request.form,
            'missing_strategy': request.form.get('missing_strategy', 'none')
        }
        
        # 转换参数
        transform_params = {}
        
        # 分析参数
        analysis_type = request.form.get('analysis_type', 'descriptive')
        
        try:
            # 创建数据处理器
            processor = DataProcessor()
            
            # 加载提取的数据
            processor.load_data(session_data[session_id]['extracted_data'])
            
            # 清洗数据
            processor.clean_data(table_index=table_index, **clean_params)
            
            # 如果有转换参数，执行转换
            if transform_params:
                processor.transform_data(table_index=table_index, **transform_params)
            
            # 分析数据
            analysis_results = processor.analyze_data(table_index=table_index, analysis_type=analysis_type)
            
            # 生成洞察
            insights = processor.generate_insights()
            
            # 保存处理结果到会话数据
            session_data[session_id]['processed_data'] = {
                'data': processor.data,
                'text': processor.text,
                'metadata': processor.metadata,
                'analysis_results': analysis_results,
                'insights': insights
            }
            
            flash('数据处理成功', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            flash(f'数据处理失败: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    # GET请求，显示处理表单
    extracted_data = session_data[session_id]['extracted_data']
    
    # 准备表格信息
    tables = []
    if isinstance(extracted_data.get('data'), pd.DataFrame):
        tables.append({
            'index': 0,
            'rows': len(extracted_data['data']),
            'columns': len(extracted_data['data'].columns),
            'preview': extracted_data['data'].head(5).to_html(index=False, classes='table table-striped')
        })
    elif isinstance(extracted_data.get('data'), list) and all(isinstance(item, pd.DataFrame) for item in extracted_data['data']):
        for i, df in enumerate(extracted_data['data']):
            tables.append({
                'index': i,
                'rows': len(df),
                'columns': len(df.columns),
                'preview': df.head(5).to_html(index=False, classes='table table-striped')
            })
    
    return render_template('process.html', 
                          session_id=session_id,
                          tables=tables,
                          has_text=extracted_data.get('text') is not None and len(extracted_data.get('text', '')) > 0,
                          text_preview=extracted_data.get('text', '')[:500] + '...' if extracted_data.get('text') and len(extracted_data.get('text')) > 500 else extracted_data.get('text', ''))

@app.route('/visualize', methods=['GET', 'POST'])
def visualize_data():
    """可视化数据"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期，请重新上传文件', 'error')
        return redirect(url_for('index'))
    
    # 检查是否有处理的数据
    if session_id not in session_data or not session_data[session_id]['processed_data']:
        flash('请先处理数据', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # 获取可视化参数
        chart_type = request.form.get('chart_type')
        table_index = request.form.get('table_index')
        if table_index:
            table_index = int(table_index)
        
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')
        color_column = request.form.get('color_column')
        title = request.form.get('title', '数据可视化')
        
        try:
            # 创建数据可视化器
            visualizer = DataVisualizer()
            
            # 设置输出目录
            export_folder = get_export_folder(session_id)
            visualizer.set_output_dir(export_folder)
            
            # 加载处理后的数据
            visualizer.load_data({
                'data': session_data[session_id]['processed_data']['data'],
                'text': session_data[session_id]['processed_data']['text'],
                'metadata': session_data[session_id]['processed_data']['metadata']
            })
            
            # 根据图表类型生成可视化
            viz_file = None
            
            if chart_type == 'bar':
                viz_file = visualizer.plot_bar(x_column, y_column, table_index=table_index, title=title)
            elif chart_type == 'line':
                viz_file = visualizer.plot_line(x_column, y_column, table_index=table_index, title=title)
            elif chart_type == 'scatter':
                viz_file = visualizer.plot_scatter(x_column, y_column, table_index=table_index, color=color_column, title=title)
            elif chart_type == 'histogram':
                viz_file = visualizer.plot_histogram(x_column, table_index=table_index, title=title)
            elif chart_type == 'boxplot':
                viz_file = visualizer.plot_boxplot(x_column, y_column, table_index=table_index, title=title)
            elif chart_type == 'heatmap':
                viz_file = visualizer.plot_heatmap(table_index=table_index, title=title)
            elif chart_type == 'pie':
                viz_file = visualizer.plot_pie(x_column, table_index=table_index, title=title)
            elif chart_type == 'wordcloud':
                if session_data[session_id]['processed_data']['text']:
                    viz_file = visualizer.plot_wordcloud(text=session_data[session_id]['processed_data']['text'], title=title)
                else:
                    viz_file = visualizer.plot_wordcloud(column=x_column, table_index=table_index, title=title)
            
            if viz_file:
                # 保存可视化结果到会话数据
                session_data[session_id]['visualizations'].append({
                    'file_path': viz_file,
                    'description': title or f"{chart_type.capitalize()} Chart",
                    'type': chart_type
                })
                
                flash('可视化生成成功', 'success')
            else:
                flash('可视化生成失败', 'error')
            
            return redirect(url_for('index'))
            
        except Exception as e:
            flash(f'可视化生成失败: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    # GET请求，显示可视化表单
    processed_data = session_data[session_id]['processed_data']
    
    # 准备表格信息
    tables = []
    columns = []
    
    if isinstance(processed_data['data'], pd.DataFrame):
        tables.append({
            'index': 0,
            'rows': len(processed_data['data']),
            'columns': len(processed_data['data'].columns)
        })
        columns = processed_data['data'].columns.tolist()
    elif isinstance(processed_data['data'], list) and all(isinstance(item, pd.DataFrame) for item in processed_data['data']):
        for i, df in enumerate(processed_data['data']):
            tables.append({
                'index': i,
                'rows': len(df),
                'columns': len(df.columns)
            })
        if tables:
            # 默认使用第一个表格的列
            columns = processed_data['data'][0].columns.tolist()
    
    return render_template('visualize.html', 
                          session_id=session_id,
                          tables=tables,
                          columns=columns,
                          has_text=processed_data['text'] is not None and len(processed_data['text']) > 0)

@app.route('/export', methods=['GET', 'POST'])
def export_data():
    """导出数据"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期，请重新上传文件', 'error')
        return redirect(url_for('index'))
    
    # 检查是否有处理的数据
    if session_id not in session_data or not session_data[session_id]['processed_data']:
        flash('请先处理数据', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # 获取导出参数
        export_format = request.form.get('export_format')
        table_index = request.form.get('table_index')
        if table_index:
            table_index = int(table_index)
        
        include_visualizations = 'include_visualizations' in request.form
        include_text = 'include_text' in request.form
        file_name = request.form.get('file_name', f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            # 创建数据导出器
            exporter = DataExporter()
            
            # 设置导出目录
            export_folder = get_export_folder(session_id)
            exporter.set_export_dir(export_folder)
            
            # 加载处理后的数据
            exporter.load_data({
                'data': session_data[session_id]['processed_data']['data'],
                'text': session_data[session_id]['processed_data']['text'],
                'metadata': session_data[session_id]['processed_data']['metadata']
            })
            
            # 添加可视化结果
            if include_visualizations and session_data[session_id]['visualizations']:
                for viz in session_data[session_id]['visualizations']:
                    exporter.add_visualization(viz['file_path'], viz['description'])
            
            # 根据导出格式导出数据
            export_file = None
            
            if export_format == 'csv':
                export_file = exporter.export_to_csv(table_index=table_index, file_name=file_name)
            elif export_format == 'excel':
                export_file = exporter.export_to_excel(file_name=file_name)
            elif export_format == 'json':
                export_file = exporter.export_to_json(table_index=table_index, file_name=file_name)
            elif export_format == 'html':
                export_file = exporter.export_to_html(file_name=file_name, include_visualizations=include_visualizations, include_text=include_text)
            elif export_format == 'pdf':
                export_file = exporter.export_to_pdf(file_name=file_name, include_visualizations=include_visualizations, include_text=include_text)
            elif export_format == 'markdown':
                export_file = exporter.export_to_markdown(file_name=file_name, include_visualizations=include_visualizations, include_text=include_text)
            elif export_format == 'image':
                export_file = exporter.export_to_image(table_index=table_index, file_name=file_name)
            elif export_format == 'report':
                # 创建综合报告
                export_file = exporter.create_report(
                    file_name=file_name,
                    title="数据分析报告",
                    format='html',
                    include_visualizations=include_visualizations,
                    include_text=include_text,
                    insights=session_data[session_id]['processed_data'].get('insights')
                )
            elif export_format == 'zip':
                # 导出所有格式并打包为ZIP
                formats = ['csv', 'excel', 'json', 'html', 'markdown']
                if include_visualizations:
                    formats.append('pdf')
                
                export_file = exporter.export_all_to_zip(
                    formats=formats,
                    file_name=file_name,
                    include_visualizations=include_visualizations,
                    include_text=include_text
                )
            
            if export_file:
                flash('数据导出成功', 'success')
            else:
                flash('数据导出失败', 'error')
            
            return redirect(url_for('index'))
            
        except Exception as e:
            flash(f'数据导出失败: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    # GET请求，显示导出表单
    processed_data = session_data[session_id]['processed_data']
    
    # 准备表格信息
    tables = []
    
    if isinstance(processed_data['data'], pd.DataFrame):
        tables.append({
            'index': 0,
            'rows': len(processed_data['data']),
            'columns': len(processed_data['data'].columns)
        })
    elif isinstance(processed_data['data'], list) and all(isinstance(item, pd.DataFrame) for item in processed_data['data']):
        for i, df in enumerate(processed_data['data']):
            tables.append({
                'index': i,
                'rows': len(df),
                'columns': len(df.columns)
            })
    
    return render_template('export.html', 
                          session_id=session_id,
                          tables=tables,
                          has_text=processed_data['text'] is not None and len(processed_data['text']) > 0,
                          has_visualizations=len(session_data[session_id]['visualizations']) > 0,
                          visualizations=session_data[session_id]['visualizations'])

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载文件"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期', 'error')
        return redirect(url_for('index'))
    
    # 检查文件是否在导出目录中
    export_folder = get_export_folder(session_id)
    file_path = os.path.join(export_folder, filename)
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(export_folder, filename, as_attachment=True)
    
    flash('文件不存在', 'error')
    return redirect(url_for('index'))

@app.route('/view/<path:filename>')
def view_file(filename):
    """查看文件"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期', 'error')
        return redirect(url_for('index'))
    
    # 检查文件是否在导出目录中
    export_folder = get_export_folder(session_id)
    file_path = os.path.join(export_folder, filename)
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        # 根据文件类型返回不同的响应
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.gif']:
            return send_from_directory(export_folder, filename)
        elif ext in ['.html']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        elif ext in ['.csv', '.json', '.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return render_template('view_text.html', filename=filename, content=content, ext=ext[1:])
        else:
            return send_from_directory(export_folder, filename)
    
    flash('文件不存在', 'error')
    return redirect(url_for('index'))

@app.route('/delete/<path:filename>')
def delete_file(filename):
    """删除文件"""
    session_id = session.get('session_id')
    if not session_id:
        flash('会话已过期', 'error')
        return redirect(url_for('index'))
    
    # 检查文件是否在会话目录或导出目录中
    session_folder = get_session_folder(session_id)
    export_folder = get_export_folder(session_id)
    
    file_path = os.path.join(session_folder, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        flash(f'文件 {filename} 已删除', 'success')
        return redirect(url_for('index'))
    
    file_path = os.path.join(export_folder, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        
        # 如果是可视化文件，也从会话数据中删除
        for i, viz in enumerate(session_data[session_id]['visualizations']):
            if os.path.basename(viz['file_path']) == filename:
                session_data[session_id]['visualizations'].pop(i)
                break
        
        flash(f'文件 {filename} 已删除', 'success')
        return redirect(url_for('index'))
    
    flash('文件不存在', 'error')
    return redirect(url_for('index'))

@app.route('/clear_session')
def clear_session():
    """清除会话数据"""
    session_id = session.get('session_id')
    if session_id:
        # 清除会话目录
        session_folder = get_session_folder(session_id)
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)
            os.makedirs(session_folder, exist_ok=True)
        
        # 清除导出目录
        export_folder = get_export_folder(session_id)
        if os.path.exists(export_folder):
            shutil.rmtree(export_folder)
            os.makedirs(export_folder, exist_ok=True)
        
        # 清除会话数据
        if session_id in session_data:
            session_data[session_id] = {
                'files': [],
                'extracted_data': None,
                'processed_data': None,
                'visualizations': [],
                'exports': []
            }
        
        flash('会话数据已清除', 'success')
    
    return redirect(url_for('index'))

@app.route('/api/columns', methods=['POST'])
def get_columns():
    """获取表格列名（AJAX API）"""
    session_id = session.get('session_id')
    if not session_id or session_id not in session_data:
        return jsonify({'error': '会话已过期'}), 400
    
    data = request.json
    table_index = data.get('table_index')
    
    if table_index is not None:
        table_index = int(table_index)
    
    # 获取处理后的数据（如果有）
    if session_data[session_id]['processed_data']:
        data_source = session_data[session_id]['processed_data']['data']
    # 否则获取提取的数据
    elif session_data[session_id]['extracted_data']:
        data_source = session_data[session_id]['extracted_data']['data']
    else:
        return jsonify({'error': '没有可用的数据'}), 400
    
    # 获取列名
    columns = []
    
    if isinstance(data_source, pd.DataFrame):
        if table_index is None or table_index == 0:
            columns = data_source.columns.tolist()
    elif isinstance(data_source, list) and all(isinstance(item, pd.DataFrame) for item in data_source):
        if table_index is not None and 0 <= table_index < len(data_source):
            columns = data_source[table_index].columns.tolist()
        elif data_source:
            # 默认使用第一个表格
            columns = data_source[0].columns.tolist()
    
    # 分类列和数值列
    categorical_columns = []
    numeric_columns = []
    
    if isinstance(data_source, pd.DataFrame):
        df = data_source
        if table_index is None or table_index == 0:
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(data_source, list) and all(isinstance(item, pd.DataFrame) for item in data_source):
        if table_index is not None and 0 <= table_index < len(data_source):
            df = data_source[table_index]
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        elif data_source:
            # 默认使用第一个表格
            df = data_source[0]
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return jsonify({
        'columns': columns,
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns
    })

@app.route('/api/preview', methods=['POST'])
def get_preview():
    """获取数据预览（AJAX API）"""
    session_id = session.get('session_id')
    if not session_id or session_id not in session_data:
        return jsonify({'error': '会话已过期'}), 400
    
    data = request.json
    table_index = data.get('table_index')
    
    if table_index is not None:
        table_index = int(table_index)
    
    # 获取处理后的数据（如果有）
    if session_data[session_id]['processed_data']:
        data_source = session_data[session_id]['processed_data']['data']
    # 否则获取提取的数据
    elif session_data[session_id]['extracted_data']:
        data_source = session_data[session_id]['extracted_data']['data']
    else:
        return jsonify({'error': '没有可用的数据'}), 400
    
    # 获取预览
    preview_html = ""
    
    if isinstance(data_source, pd.DataFrame):
        if table_index is None or table_index == 0:
            preview_html = data_source.head(10).to_html(index=False, classes='table table-striped')
    elif isinstance(data_source, list) and all(isinstance(item, pd.DataFrame) for item in data_source):
        if table_index is not None and 0 <= table_index < len(data_source):
            preview_html = data_source[table_index].head(10).to_html(index=False, classes='table table-striped')
        elif data_source:
            # 默认使用第一个表格
            preview_html = data_source[0].head(10).to_html(index=False, classes='table table-striped')
    
    return jsonify({
        'preview_html': preview_html
    })

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """获取数据洞察（AJAX API）"""
    session_id = session.get('session_id')
    if not session_id or session_id not in session_data:
        return jsonify({'error': '会话已过期'}), 400
    
    # 检查是否有处理的数据
    if not session_data[session_id]['processed_data'] or 'insights' not in session_data[session_id]['processed_data']:
        return jsonify({'error': '没有可用的洞察'}), 400
    
    insights = session_data[session_id]['processed_data']['insights']
    
    return jsonify(insights)

@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大错误"""
    flash('上传的文件过大，请压缩后再上传', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_server_error(error):
    """处理服务器错误"""
    flash('服务器错误，请稍后再试', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
