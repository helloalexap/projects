#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据可视化模块 - 负责将处理后的数据转换为直观的图表和可视化效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import os
import json
import re

# 设置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 设置Seaborn样式
sns.set(style="whitegrid")

class DataVisualizer:
    """数据可视化器类，用于将数据转换为各种可视化图表"""
    
    def __init__(self):
        """初始化数据可视化器"""
        self.data = None
        self.text = None
        self.metadata = {}
        self.visualization_history = []
        self.default_width = 800
        self.default_height = 500
        self.default_dpi = 100
        self.color_palette = 'viridis'  # 默认颜色方案
        self.theme = 'light'  # 默认主题
        self.output_format = 'png'  # 默认输出格式
        self.output_dir = 'exports'  # 默认输出目录
        
    def load_data(self, data_result):
        """加载数据"""
        if 'error' in data_result:
            raise ValueError(f"数据加载错误: {data_result['error']}")
        
        self.data = data_result.get('data')
        self.metadata = data_result.get('metadata', {})
        self.text = data_result.get('text')
        
        self._add_to_history("数据加载完成")
        
        return self
    
    def _add_to_history(self, operation, details=None, file_path=None):
        """添加可视化历史记录"""
        self.visualization_history.append({
            'operation': operation,
            'details': details,
            'file_path': file_path
        })
    
    def get_visualization_history(self):
        """获取可视化历史记录"""
        return self.visualization_history
    
    def set_theme(self, theme):
        """设置可视化主题"""
        available_themes = ['light', 'dark', 'colorblind', 'pastel', 'bright']
        if theme not in available_themes:
            raise ValueError(f"不支持的主题: {theme}。可用主题: {available_themes}")
        
        self.theme = theme
        
        # 应用主题设置
        if theme == 'light':
            sns.set_style("whitegrid")
            self.color_palette = 'viridis'
        elif theme == 'dark':
            sns.set_style("darkgrid")
            self.color_palette = 'plasma'
        elif theme == 'colorblind':
            sns.set_style("whitegrid")
            self.color_palette = 'colorblind'
        elif theme == 'pastel':
            sns.set_style("whitegrid")
            self.color_palette = 'pastel'
        elif theme == 'bright':
            sns.set_style("whitegrid")
            self.color_palette = 'bright'
        
        self._add_to_history("设置主题", {'theme': theme})
        
        return self
    
    def set_output_format(self, output_format):
        """设置输出格式"""
        available_formats = ['png', 'jpg', 'svg', 'pdf', 'html', 'json']
        if output_format not in available_formats:
            raise ValueError(f"不支持的输出格式: {output_format}。可用格式: {available_formats}")
        
        self.output_format = output_format
        self._add_to_history("设置输出格式", {'format': output_format})
        
        return self
    
    def set_output_dir(self, output_dir):
        """设置输出目录"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.output_dir = output_dir
        self._add_to_history("设置输出目录", {'directory': output_dir})
        
        return self
    
    def set_figure_size(self, width=None, height=None, dpi=None):
        """设置图表尺寸"""
        if width is not None:
            self.default_width = width
        if height is not None:
            self.default_height = height
        if dpi is not None:
            self.default_dpi = dpi
        
        self._add_to_history("设置图表尺寸", {
            'width': self.default_width,
            'height': self.default_height,
            'dpi': self.default_dpi
        })
        
        return self
    
    def _prepare_dataframe(self, table_index=None):
        """准备用于可视化的DataFrame"""
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
            raise ValueError("数据格式不支持可视化操作")
    
    def _save_figure(self, fig, file_name, plot_type, backend='matplotlib'):
        """保存图表到文件"""
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 构建文件路径
        file_path = os.path.join(self.output_dir, f"{file_name}.{self.output_format}")
        
        # 根据不同的后端保存图表
        if backend == 'matplotlib':
            if self.output_format in ['png', 'jpg', 'svg', 'pdf']:
                fig.savefig(file_path, dpi=self.default_dpi, bbox_inches='tight')
            else:
                # 对于不支持的格式，默认保存为PNG
                alt_file_path = os.path.join(self.output_dir, f"{file_name}.png")
                fig.savefig(alt_file_path, dpi=self.default_dpi, bbox_inches='tight')
                file_path = alt_file_path
            plt.close(fig)
        
        elif backend == 'plotly':
            if self.output_format == 'html':
                pio.write_html(fig, file=file_path)
            elif self.output_format == 'json':
                with open(file_path, 'w') as f:
                    f.write(json.dumps(fig.to_dict()))
            else:
                # 对于其他格式，使用plotly的写图功能
                pio.write_image(fig, file_path)
        
        self._add_to_history(f"保存{plot_type}图表", {'file_name': file_name}, file_path)
        
        return file_path
    
    def _get_color_palette(self, n_colors=10):
        """获取颜色方案"""
        return sns.color_palette(self.color_palette, n_colors=n_colors)
    
    def plot_bar(self, x, y, table_index=None, **kwargs):
        """绘制条形图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        if y not in df.columns:
            raise ValueError(f"列 '{y}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"{y} by {x}")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', y)
        orientation = kwargs.get('orientation', 'vertical')
        color = kwargs.get('color', self._get_color_palette()[0])
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"bar_{x}_{y}")
        backend = kwargs.get('backend', 'matplotlib')
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            if orientation == 'vertical':
                sns.barplot(x=x, y=y, data=df, ax=ax, color=color)
            else:
                sns.barplot(x=y, y=x, data=df, ax=ax, color=color)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 旋转x轴标签（如果数量多）
            if len(df[x].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            if orientation == 'vertical':
                fig = px.bar(df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel})
            else:
                fig = px.bar(df, x=y, y=x, title=title, orientation='h', labels={x: xlabel, y: ylabel})
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "条形图", backend)
        
        return file_path
    
    def plot_line(self, x, y, table_index=None, **kwargs):
        """绘制折线图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        if y not in df.columns:
            raise ValueError(f"列 '{y}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"{y} over {x}")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', y)
        color = kwargs.get('color', self._get_color_palette()[0])
        marker = kwargs.get('marker', 'o')
        linestyle = kwargs.get('linestyle', '-')
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"line_{x}_{y}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 确保数据按x排序
        df = df.sort_values(by=x)
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            ax.plot(df[x], df[y], marker=marker, linestyle=linestyle, color=color)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 旋转x轴标签（如果数量多）
            if len(df[x].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = px.line(df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel}, markers=True)
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "折线图", backend)
        
        return file_path
    
    def plot_scatter(self, x, y, table_index=None, **kwargs):
        """绘制散点图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        if y not in df.columns:
            raise ValueError(f"列 '{y}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"{y} vs {x}")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', y)
        color_col = kwargs.get('color', None)  # 用于着色的列
        size_col = kwargs.get('size', None)    # 用于调整点大小的列
        alpha = kwargs.get('alpha', 0.7)
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"scatter_{x}_{y}")
        backend = kwargs.get('backend', 'matplotlib')
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            if color_col and color_col in df.columns:
                scatter = ax.scatter(df[x], df[y], c=df[color_col], alpha=alpha, cmap=self.color_palette)
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                ax.scatter(df[x], df[y], alpha=alpha, color=self._get_color_palette()[0])
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            if color_col and color_col in df.columns:
                fig = px.scatter(df, x=x, y=y, color=color_col, size=size_col,
                                title=title, labels={x: xlabel, y: ylabel})
            else:
                fig = px.scatter(df, x=x, y=y, size=size_col,
                                title=title, labels={x: xlabel, y: ylabel})
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "散点图", backend)
        
        return file_path
    
    def plot_histogram(self, column, table_index=None, **kwargs):
        """绘制直方图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"Distribution of {column}")
        xlabel = kwargs.get('xlabel', column)
        ylabel = kwargs.get('ylabel', 'Frequency')
        bins = kwargs.get('bins', 'auto')
        kde = kwargs.get('kde', True)
        color = kwargs.get('color', self._get_color_palette()[0])
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"histogram_{column}")
        backend = kwargs.get('backend', 'matplotlib')
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            sns.histplot(df[column], bins=bins, kde=kde, ax=ax, color=color)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = px.histogram(df, x=column, title=title, labels={column: xlabel})
            
            if kde:
                # 添加KDE曲线
                hist_data = [df[column].dropna()]
                group_labels = [column]
                
                fig_kde = ff.create_distplot(hist_data, group_labels, show_hist=False)
                for trace in fig_kde.data:
                    fig.add_trace(trace)
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height,
                yaxis_title=ylabel
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "直方图", backend)
        
        return file_path
    
    def plot_boxplot(self, x=None, y=None, table_index=None, **kwargs):
        """绘制箱线图"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        title = kwargs.get('title', "Boxplot")
        orient = kwargs.get('orient', 'v')  # 'v' for vertical, 'h' for horizontal
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', "boxplot")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 检查列是否存在
        if x is not None and x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        if y is not None and y not in df.columns:
            raise ValueError(f"列 '{y}' 不存在")
        
        # 如果只提供了一个列，则绘制该列的箱线图
        if x is not None and y is None:
            column = x
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"列 '{column}' 不是数值类型")
            
            title = kwargs.get('title', f"Distribution of {column}")
            file_name = kwargs.get('file_name', f"boxplot_{column}")
            
            if backend == 'matplotlib':
                fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
                sns.boxplot(y=df[column], ax=ax)
                ax.set_title(title)
                plt.tight_layout()
            
            elif backend == 'plotly':
                fig = px.box(df, y=column, title=title)
                fig.update_layout(width=self.default_width, height=self.default_height)
        
        # 如果提供了两个列，则绘制分组箱线图
        elif x is not None and y is not None:
            if orient == 'v':
                xlabel, ylabel = x, y
            else:
                xlabel, ylabel = y, x
            
            title = kwargs.get('title', f"Distribution of {ylabel} by {xlabel}")
            file_name = kwargs.get('file_name', f"boxplot_{xlabel}_{ylabel}")
            
            if backend == 'matplotlib':
                fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
                
                if orient == 'v':
                    sns.boxplot(x=x, y=y, data=df, ax=ax)
                else:
                    sns.boxplot(x=y, y=x, data=df, ax=ax)
                
                ax.set_title(title)
                
                # 旋转x轴标签（如果数量多）
                if orient == 'v' and len(df[x].unique()) > 5:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
            
            elif backend == 'plotly':
                if orient == 'v':
                    fig = px.box(df, x=x, y=y, title=title)
                else:
                    fig = px.box(df, x=y, y=x, title=title)
                
                fig.update_layout(width=self.default_width, height=self.default_height)
        
        # 如果没有提供列，则绘制所有数值列的箱线图
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("没有数值列可以绘制箱线图")
            
            title = kwargs.get('title', "Distribution of Numeric Variables")
            file_name = kwargs.get('file_name', "boxplot_all_numeric")
            
            if backend == 'matplotlib':
                fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
                
                # 将数据转换为长格式
                df_melt = df[numeric_cols].melt()
                
                sns.boxplot(x='variable', y='value', data=df_melt, ax=ax)
                ax.set_title(title)
                ax.set_xlabel('')
                ax.set_ylabel('Value')
                
                # 旋转x轴标签
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
            
            elif backend == 'plotly':
                fig = make_subplots(rows=1, cols=1)
                
                for i, col in enumerate(numeric_cols):
                    fig.add_trace(go.Box(y=df[col], name=col))
                
                fig.update_layout(
                    title=title,
                    width=self.default_width,
                    height=self.default_height
                )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "箱线图", backend)
        
        return file_path
    
    def plot_heatmap(self, table_index=None, **kwargs):
        """绘制热力图（相关性矩阵）"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        title = kwargs.get('title', "Correlation Matrix")
        method = kwargs.get('method', 'pearson')  # 'pearson', 'kendall', 'spearman'
        cmap = kwargs.get('cmap', self.color_palette)
        annot = kwargs.get('annot', True)
        fmt = kwargs.get('fmt', '.2f')
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"heatmap_corr_{method}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("没有数值列可以计算相关性")
        
        # 计算相关性矩阵
        corr_matrix = numeric_df.corr(method=method)
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, fmt=fmt, 
                        linewidths=.5, ax=ax, vmin=-1, vmax=1)
            
            ax.set_title(title)
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = px.imshow(corr_matrix, 
                           labels=dict(x="Features", y="Features", color="Correlation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           title=title,
                           color_continuous_scale=px.colors.diverging.RdBu_r,
                           zmin=-1, zmax=1)
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "热力图", backend)
        
        return file_path
    
    def plot_pie(self, column, table_index=None, **kwargs):
        """绘制饼图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"Distribution of {column}")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"pie_{column}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 计算值计数
        value_counts = df[column].value_counts()
        
        # 如果类别太多，只保留前N个
        max_categories = kwargs.get('max_categories', 10)
        if len(value_counts) > max_categories:
            others = value_counts[max_categories:].sum()
            value_counts = value_counts[:max_categories]
            value_counts['Others'] = others
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            colors = self._get_color_palette(len(value_counts))
            wedges, texts, autotexts = ax.pie(
                value_counts, 
                labels=value_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            # 设置标签文本属性
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            ax.set_title(title)
            ax.axis('equal')  # 确保饼图是圆形的
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title
            )
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "饼图", backend)
        
        return file_path
    
    def plot_wordcloud(self, text=None, column=None, table_index=None, **kwargs):
        """绘制词云图"""
        # 获取参数
        title = kwargs.get('title', "Word Cloud")
        width = kwargs.get('width', self.default_width)
        height = kwargs.get('height', self.default_height)
        background_color = kwargs.get('background_color', 'white')
        max_words = kwargs.get('max_words', 200)
        file_name = kwargs.get('file_name', "wordcloud")
        
        # 获取文本数据
        if text is None:
            if self.text is not None:
                text = self.text
            elif column is not None:
                df = self._prepare_dataframe(table_index)
                if column not in df.columns:
                    raise ValueError(f"列 '{column}' 不存在")
                
                # 合并列中的所有文本
                text = ' '.join(df[column].astype(str).tolist())
                file_name = f"wordcloud_{column}"
            else:
                raise ValueError("必须提供文本数据或指定文本列")
        
        # 创建词云
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            collocations=False,
            regexp=r'\w+'
        ).generate(text)
        
        # 绘制词云
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=self.default_dpi)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "词云图", 'matplotlib')
        
        return file_path
    
    def plot_pca(self, table_index=None, **kwargs):
        """绘制PCA可视化"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        n_components = kwargs.get('n_components', 2)
        title = kwargs.get('title', f"PCA Visualization ({n_components} Components)")
        color_col = kwargs.get('color', None)  # 用于着色的列
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"pca_{n_components}d")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("没有数值列可以进行PCA")
        
        if len(numeric_df.columns) < n_components:
            raise ValueError(f"数值列数量({len(numeric_df.columns)})少于请求的主成分数量({n_components})")
        
        # 处理缺失值
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(numeric_df)
        
        # 标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 创建PCA结果DataFrame
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        
        # 如果提供了颜色列，添加到PCA结果中
        if color_col and color_col in df.columns:
            pca_df[color_col] = df[color_col].values
        
        # 计算解释方差比例
        explained_variance = pca.explained_variance_ratio_
        
        if n_components == 2:
            # 2D PCA可视化
            if backend == 'matplotlib':
                fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
                
                if color_col and color_col in df.columns:
                    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df[color_col], cmap=self.color_palette, alpha=0.7)
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        plt.colorbar(scatter, ax=ax, label=color_col)
                    else:
                        # 为分类变量创建图例
                        categories = df[color_col].unique()
                        for category in categories:
                            mask = pca_df[color_col] == category
                            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], label=category, alpha=0.7)
                        ax.legend()
                else:
                    ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, color=self._get_color_palette()[0])
                
                ax.set_title(title)
                ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                
                plt.tight_layout()
                
            elif backend == 'plotly':
                if color_col and color_col in df.columns:
                    fig = px.scatter(
                        pca_df, x='PC1', y='PC2', color=color_col,
                        title=title,
                        labels={
                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                            'PC2': f'PC2 ({explained_variance[1]:.2%})'
                        }
                    )
                else:
                    fig = px.scatter(
                        pca_df, x='PC1', y='PC2',
                        title=title,
                        labels={
                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                            'PC2': f'PC2 ({explained_variance[1]:.2%})'
                        }
                    )
                
                fig.update_layout(
                    width=self.default_width,
                    height=self.default_height
                )
        
        elif n_components == 3:
            # 3D PCA可视化
            if backend == 'matplotlib':
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=figsize, dpi=self.default_dpi)
                ax = fig.add_subplot(111, projection='3d')
                
                if color_col and color_col in df.columns:
                    scatter = ax.scatter(
                        pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
                        c=pca_df[color_col], cmap=self.color_palette, alpha=0.7
                    )
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        plt.colorbar(scatter, ax=ax, label=color_col)
                    else:
                        # 为分类变量创建图例
                        categories = df[color_col].unique()
                        for category in categories:
                            mask = pca_df[color_col] == category
                            ax.scatter(
                                pca_df.loc[mask, 'PC1'],
                                pca_df.loc[mask, 'PC2'],
                                pca_df.loc[mask, 'PC3'],
                                label=category, alpha=0.7
                            )
                        ax.legend()
                else:
                    ax.scatter(
                        pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
                        alpha=0.7, color=self._get_color_palette()[0]
                    )
                
                ax.set_title(title)
                ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
                
                plt.tight_layout()
                
            elif backend == 'plotly':
                if color_col and color_col in df.columns:
                    fig = px.scatter_3d(
                        pca_df, x='PC1', y='PC2', z='PC3', color=color_col,
                        title=title,
                        labels={
                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                            'PC2': f'PC2 ({explained_variance[1]:.2%})',
                            'PC3': f'PC3 ({explained_variance[2]:.2%})'
                        }
                    )
                else:
                    fig = px.scatter_3d(
                        pca_df, x='PC1', y='PC2', z='PC3',
                        title=title,
                        labels={
                            'PC1': f'PC1 ({explained_variance[0]:.2%})',
                            'PC2': f'PC2 ({explained_variance[1]:.2%})',
                            'PC3': f'PC3 ({explained_variance[2]:.2%})'
                        }
                    )
                
                fig.update_layout(
                    width=self.default_width,
                    height=self.default_height
                )
        
        else:
            # 对于更高维度的PCA，绘制解释方差比例图
            if backend == 'matplotlib':
                fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
                
                # 绘制解释方差比例
                ax.bar(range(1, n_components + 1), explained_variance, alpha=0.7, color=self._get_color_palette()[0])
                ax.plot(range(1, n_components + 1), np.cumsum(explained_variance), 'r-o', alpha=0.7)
                
                ax.set_title("PCA Explained Variance")
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Explained Variance Ratio")
                ax.set_xticks(range(1, n_components + 1))
                
                # 添加累积解释方差的第二个y轴
                ax2 = ax.twinx()
                ax2.set_ylabel('Cumulative Explained Variance')
                ax2.plot(range(1, n_components + 1), np.cumsum(explained_variance), 'r-o', alpha=0.7)
                ax2.set_ylim([0, 1.1])
                
                plt.tight_layout()
                
            elif backend == 'plotly':
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 添加条形图（解释方差比例）
                fig.add_trace(
                    go.Bar(
                        x=list(range(1, n_components + 1)),
                        y=explained_variance,
                        name="Explained Variance"
                    ),
                    secondary_y=False
                )
                
                # 添加折线图（累积解释方差）
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, n_components + 1)),
                        y=np.cumsum(explained_variance),
                        name="Cumulative Explained Variance",
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="PCA Explained Variance",
                    xaxis_title="Principal Component",
                    width=self.default_width,
                    height=self.default_height
                )
                
                fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
                fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "PCA可视化", backend)
        
        return file_path
    
    def plot_cluster(self, table_index=None, **kwargs):
        """绘制聚类可视化"""
        df = self._prepare_dataframe(table_index)
        
        # 获取参数
        n_clusters = kwargs.get('n_clusters', 3)
        method = kwargs.get('method', 'kmeans')  # 'kmeans', 'hierarchical'
        features = kwargs.get('features', None)  # 用于聚类的特征列表
        title = kwargs.get('title', f"Cluster Visualization ({method}, {n_clusters} clusters)")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"cluster_{method}_{n_clusters}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 选择数值列
        if features:
            # 确保所有特征都存在且为数值类型
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"特征列不存在: {missing_features}")
            
            non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(df[f])]
            if non_numeric:
                raise ValueError(f"非数值特征列: {non_numeric}")
            
            numeric_df = df[features]
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("没有数值列可以进行聚类")
        
        if len(numeric_df.columns) < 2:
            raise ValueError("至少需要两个数值列进行聚类可视化")
        
        # 处理缺失值
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(numeric_df)
        
        # 标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行聚类
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        clusters = cluster_model.fit_predict(X_scaled)
        
        # 执行降维以便可视化（如果特征数量大于2）
        if X_scaled.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            
            # 创建可视化数据
            viz_df = pd.DataFrame({
                'x': X_2d[:, 0],
                'y': X_2d[:, 1],
                'cluster': clusters
            })
            
            x_label = f'PC1 ({explained_variance[0]:.2%})'
            y_label = f'PC2 ({explained_variance[1]:.2%})'
            
        else:
            # 如果只有两个特征，直接使用
            viz_df = pd.DataFrame({
                'x': X_scaled[:, 0],
                'y': X_scaled[:, 1],
                'cluster': clusters
            })
            
            x_label = numeric_df.columns[0]
            y_label = numeric_df.columns[1]
        
        # 绘制聚类结果
        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 为每个聚类绘制散点图
            for cluster_id in range(n_clusters):
                cluster_points = viz_df[viz_df['cluster'] == cluster_id]
                ax.scatter(
                    cluster_points['x'], cluster_points['y'],
                    label=f'Cluster {cluster_id}',
                    alpha=0.7
                )
            
            # 如果是K-means，绘制聚类中心
            if method == 'kmeans' and X_scaled.shape[1] > 2:
                # 将聚类中心转换到PCA空间
                centers = cluster_model.cluster_centers_
                centers_2d = pca.transform(centers)
                
                ax.scatter(
                    centers_2d[:, 0], centers_2d[:, 1],
                    s=100, c='black', marker='X', label='Centroids'
                )
            
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            viz_df['cluster'] = viz_df['cluster'].astype(str)  # 将聚类ID转换为字符串
            
            fig = px.scatter(
                viz_df, x='x', y='y', color='cluster',
                title=title,
                labels={'x': x_label, 'y': y_label, 'cluster': 'Cluster'}
            )
            
            # 如果是K-means，添加聚类中心
            if method == 'kmeans' and X_scaled.shape[1] > 2:
                # 将聚类中心转换到PCA空间
                centers = cluster_model.cluster_centers_
                centers_2d = pca.transform(centers)
                
                for i, (x, y) in enumerate(centers_2d):
                    fig.add_trace(
                        go.Scatter(
                            x=[x], y=[y],
                            mode='markers',
                            marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                            name=f'Centroid {i}'
                        )
                    )
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "聚类可视化", backend)
        
        return file_path
    
    def plot_time_series(self, date_col, value_col, table_index=None, **kwargs):
        """绘制时间序列图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if date_col not in df.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在")
        if value_col not in df.columns:
            raise ValueError(f"值列 '{value_col}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', f"{value_col} over Time")
        xlabel = kwargs.get('xlabel', date_col)
        ylabel = kwargs.get('ylabel', value_col)
        color = kwargs.get('color', self._get_color_palette()[0])
        marker = kwargs.get('marker', 'o')
        linestyle = kwargs.get('linestyle', '-')
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"time_series_{value_col}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 确保日期列是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                raise ValueError(f"无法将列 '{date_col}' 转换为日期时间类型")
        
        # 按日期排序
        df = df.sort_values(by=date_col)
        
        # 计算移动平均（如果需要）
        window = kwargs.get('window', None)
        if window:
            df[f'{value_col}_MA{window}'] = df[value_col].rolling(window=window).mean()
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 绘制原始数据
            ax.plot(df[date_col], df[value_col], marker=marker, linestyle=linestyle, color=color, label=value_col)
            
            # 绘制移动平均（如果有）
            if window:
                ax.plot(df[date_col], df[f'{value_col}_MA{window}'], linestyle='-', color='red', 
                       label=f'{window}-period Moving Average')
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 设置x轴日期格式
            fig.autofmt_xdate()
            
            if window:
                ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            # 添加原始数据
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[value_col],
                mode='lines+markers' if marker else 'lines',
                name=value_col
            ))
            
            # 添加移动平均（如果有）
            if window:
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[f'{value_col}_MA{window}'],
                    mode='lines',
                    name=f'{window}-period Moving Average',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "时间序列图", backend)
        
        return file_path
    
    def plot_multiple_lines(self, x, y_columns, table_index=None, **kwargs):
        """绘制多条折线图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        for col in y_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', "Multiple Lines")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', "Value")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"multi_line_{x}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 确保数据按x排序
        df = df.sort_values(by=x)
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 获取颜色方案
            colors = self._get_color_palette(len(y_columns))
            
            # 绘制每一列
            for i, col in enumerate(y_columns):
                ax.plot(df[x], df[col], label=col, color=colors[i])
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 旋转x轴标签（如果数量多）
            if len(df[x].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            # 添加每一列
            for col in y_columns:
                fig.add_trace(go.Scatter(
                    x=df[x], y=df[col],
                    mode='lines+markers',
                    name=col
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "多折线图", backend)
        
        return file_path
    
    def plot_stacked_bar(self, x, y_columns, table_index=None, **kwargs):
        """绘制堆叠条形图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        for col in y_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', "Stacked Bar Chart")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', "Value")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"stacked_bar_{x}")
        backend = kwargs.get('backend', 'matplotlib')
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 获取颜色方案
            colors = self._get_color_palette(len(y_columns))
            
            # 绘制堆叠条形图
            bottom = np.zeros(len(df))
            for i, col in enumerate(y_columns):
                ax.bar(df[x], df[col], bottom=bottom, label=col, color=colors[i])
                bottom += df[col].values
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 旋转x轴标签（如果数量多）
            if len(df[x].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            # 添加每一列
            for col in y_columns:
                fig.add_trace(go.Bar(
                    x=df[x], y=df[col],
                    name=col
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                barmode='stack',
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "堆叠条形图", backend)
        
        return file_path
    
    def plot_grouped_bar(self, x, y_columns, table_index=None, **kwargs):
        """绘制分组条形图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        for col in y_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', "Grouped Bar Chart")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', "Value")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"grouped_bar_{x}")
        backend = kwargs.get('backend', 'matplotlib')
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 获取x值和设置条形宽度
            x_values = df[x].unique()
            n_groups = len(x_values)
            n_bars = len(y_columns)
            bar_width = 0.8 / n_bars
            
            # 获取颜色方案
            colors = self._get_color_palette(len(y_columns))
            
            # 绘制分组条形图
            for i, col in enumerate(y_columns):
                x_pos = np.arange(n_groups) - 0.4 + (i + 0.5) * bar_width
                ax.bar(x_pos, df.groupby(x)[col].mean(), width=bar_width, label=col, color=colors[i])
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 设置x轴刻度
            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels(x_values)
            
            # 旋转x轴标签（如果数量多）
            if len(x_values) > 5:
                plt.xticks(rotation=45, ha='right')
            
            ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            # 添加每一列
            for col in y_columns:
                fig.add_trace(go.Bar(
                    x=df[x], y=df[col],
                    name=col
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                barmode='group',
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "分组条形图", backend)
        
        return file_path
    
    def plot_area(self, x, y_columns, table_index=None, **kwargs):
        """绘制面积图"""
        df = self._prepare_dataframe(table_index)
        
        # 检查列是否存在
        if x not in df.columns:
            raise ValueError(f"列 '{x}' 不存在")
        
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        for col in y_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在")
        
        # 获取参数
        title = kwargs.get('title', "Area Chart")
        xlabel = kwargs.get('xlabel', x)
        ylabel = kwargs.get('ylabel', "Value")
        stacked = kwargs.get('stacked', True)
        alpha = kwargs.get('alpha', 0.7)
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', f"area_{x}")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 确保数据按x排序
        df = df.sort_values(by=x)
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
            
            # 获取颜色方案
            colors = self._get_color_palette(len(y_columns))
            
            # 绘制面积图
            if stacked:
                ax.stackplot(df[x], [df[col] for col in y_columns], labels=y_columns, colors=colors, alpha=alpha)
            else:
                for i, col in enumerate(y_columns):
                    ax.fill_between(df[x], df[col], alpha=alpha, color=colors[i], label=col)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # 旋转x轴标签（如果数量多）
            if len(df[x].unique()) > 5:
                plt.xticks(rotation=45, ha='right')
            
            ax.legend()
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            # 添加每一列
            for i, col in enumerate(y_columns):
                fig.add_trace(go.Scatter(
                    x=df[x], y=df[col],
                    mode='lines',
                    name=col,
                    fill='tonexty' if stacked else 'tozeroy',
                    stackgroup='one' if stacked else None
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "面积图", backend)
        
        return file_path
    
    def plot_radar(self, categories, values, **kwargs):
        """绘制雷达图"""
        # 获取参数
        title = kwargs.get('title', "Radar Chart")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', "radar_chart")
        backend = kwargs.get('backend', 'matplotlib')
        
        # 确保categories和values是列表
        if not isinstance(categories, list):
            categories = [categories]
        if not isinstance(values, list):
            values = [values]
        
        # 确保categories和values长度相同
        if len(categories) != len(values):
            raise ValueError("categories和values的长度必须相同")
        
        # 添加首尾相连
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        if backend == 'matplotlib':
            # 使用Matplotlib绘制
            fig = plt.figure(figsize=figsize, dpi=self.default_dpi)
            ax = fig.add_subplot(111, polar=True)
            
            # 计算角度
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # 设置刻度标签
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            
            ax.set_title(title)
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=title
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.1]
                    )
                ),
                width=self.default_width,
                height=self.default_height
            )
        
        # 保存图表
        file_path = self._save_figure(fig, file_name, "雷达图", backend)
        
        return file_path
    
    def plot_treemap(self, labels, values, **kwargs):
        """绘制树形图"""
        # 获取参数
        title = kwargs.get('title', "Treemap")
        figsize = kwargs.get('figsize', (self.default_width/100, self.default_height/100))
        file_name = kwargs.get('file_name', "treemap")
        backend = kwargs.get('backend', 'plotly')  # 树形图主要使用Plotly
        
        # 确保labels和values是列表
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(values, list):
            values = [values]
        
        # 确保labels和values长度相同
        if len(labels) != len(values):
            raise ValueError("labels和values的长度必须相同")
        
        if backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure(go.Treemap(
                labels=labels,
                values=values,
                parents=[""] * len(labels),
                textinfo="label+value+percent"
            ))
            
            fig.update_layout(
                title=title,
                width=self.default_width,
                height=self.default_height
            )
            
            # 保存图表
            file_path = self._save_figure(fig, file_name, "树形图", backend)
            
            return file_path
        else:
            raise ValueError("树形图仅支持Plotly后端")
    
    def plot_sunburst(self, labels, parents, values, **kwargs):
        """绘制旭日图"""
        # 获取参数
        title = kwargs.get('title', "Sunburst Chart")
        file_name = kwargs.get('file_name', "sunburst")
        backend = kwargs.get('backend', 'plotly')  # 旭日图主要使用Plotly
        
        # 确保labels、parents和values是列表
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(parents, list):
            parents = [parents]
        if not isinstance(values, list):
            values = [values]
        
        # 确保labels、parents和values长度相同
        if len(labels) != len(parents) or len(labels) != len(values):
            raise ValueError("labels、parents和values的长度必须相同")
        
        if backend == 'plotly':
            # 使用Plotly绘制
            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total"
            ))
            
            fig.update_layout(
                title=title,
                width=self.default_width,
                height=self.default_height
            )
            
            # 保存图表
            file_path = self._save_figure(fig, file_name, "旭日图", backend)
            
            return file_path
        else:
            raise ValueError("旭日图仅支持Plotly后端")
    
    def plot_sankey(self, sources, targets, values, **kwargs):
        """绘制桑基图"""
        # 获取参数
        title = kwargs.get('title', "Sankey Diagram")
        node_labels = kwargs.get('node_labels', None)
        file_name = kwargs.get('file_name', "sankey")
        backend = kwargs.get('backend', 'plotly')  # 桑基图主要使用Plotly
        
        # 确保sources、targets和values是列表
        if not isinstance(sources, list):
            sources = [sources]
        if not isinstance(targets, list):
            targets = [targets]
        if not isinstance(values, list):
            values = [values]
        
        # 确保sources、targets和values长度相同
        if len(sources) != len(targets) or len(sources) != len(values):
            raise ValueError("sources、targets和values的长度必须相同")
        
        if backend == 'plotly':
            # 使用Plotly绘制
            
            # 如果没有提供节点标签，则从sources和targets中提取
            if node_labels is None:
                all_nodes = list(set(sources + targets))
                node_labels = {node: node for node in all_nodes}
            
            # 创建节点索引映射
            node_indices = {node: i for i, node in enumerate(node_labels.keys())}
            
            # 转换sources和targets为索引
            source_indices = [node_indices[source] for source in sources]
            target_indices = [node_indices[target] for target in targets]
            
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(node_labels.values())
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values
                )
            ))
            
            fig.update_layout(
                title=title,
                width=self.default_width,
                height=self.default_height
            )
            
            # 保存图表
            file_path = self._save_figure(fig, file_name, "桑基图", backend)
            
            return file_path
        else:
            raise ValueError("桑基图仅支持Plotly后端")
    
    def create_dashboard(self, plots, layout=None, **kwargs):
        """创建仪表板（多图表组合）"""
        # 获取参数
        title = kwargs.get('title', "Dashboard")
        file_name = kwargs.get('file_name', "dashboard")
        
        # 确保plots是列表
        if not isinstance(plots, list):
            plots = [plots]
        
        # 检查所有图表文件是否存在
        for plot in plots:
            if not os.path.exists(plot):
                raise ValueError(f"图表文件不存在: {plot}")
        
        # 如果未指定布局，则自动计算
        if layout is None:
            n_plots = len(plots)
            if n_plots <= 2:
                layout = (1, n_plots)
            elif n_plots <= 4:
                layout = (2, 2)
            elif n_plots <= 6:
                layout = (2, 3)
            elif n_plots <= 9:
                layout = (3, 3)
            else:
                layout = (4, 3)
        
        rows, cols = layout
        
        # 创建仪表板
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5), dpi=self.default_dpi)
        
        # 将axes转换为一维数组
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 将图表添加到仪表板
        for i, plot in enumerate(plots):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            # 读取图表图像
            img = plt.imread(plot)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            
            # 添加图表标题（使用文件名）
            plot_title = os.path.splitext(os.path.basename(plot))[0]
            axes[row, col].set_title(plot_title)
        
        # 隐藏未使用的子图
        for i in range(len(plots), rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])
        
        # 设置仪表板标题
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为标题留出空间
        
        # 保存仪表板
        output_path = os.path.join(self.output_dir, f"{file_name}.{self.output_format}")
        fig.savefig(output_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close(fig)
        
        self._add_to_history("创建仪表板", {'plots': plots, 'layout': layout}, output_path)
        
        return output_path
    
    def create_interactive_dashboard(self, plots, layout=None, **kwargs):
        """创建交互式仪表板（使用Plotly）"""
        # 获取参数
        title = kwargs.get('title', "Interactive Dashboard")
        file_name = kwargs.get('file_name', "interactive_dashboard")
        
        # 确保plots是列表
        if not isinstance(plots, list):
            plots = [plots]
        
        # 检查所有图表文件是否存在
        for plot in plots:
            if not os.path.exists(plot):
                raise ValueError(f"图表文件不存在: {plot}")
        
        # 如果未指定布局，则自动计算
        if layout is None:
            n_plots = len(plots)
            if n_plots <= 2:
                layout = (1, n_plots)
            elif n_plots <= 4:
                layout = (2, 2)
            elif n_plots <= 6:
                layout = (2, 3)
            elif n_plots <= 9:
                layout = (3, 3)
            else:
                layout = (4, 3)
        
        rows, cols = layout
        
        # 创建子图
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[os.path.splitext(os.path.basename(plot))[0] for plot in plots]
        )
        
        # 将图表添加到仪表板
        for i, plot in enumerate(plots):
            if i >= rows * cols:
                break
                
            row = i // cols + 1  # Plotly使用1-based索引
            col = i % cols + 1   # Plotly使用1-based索引
            
            # 读取Plotly图表
            try:
                with open(plot, 'r') as f:
                    plot_data = json.load(f)
                
                # 添加图表的所有轨迹
                for trace in plot_data['data']:
                    fig.add_trace(trace, row=row, col=col)
                
                # 更新轴标签
                if 'layout' in plot_data:
                    if 'xaxis' in plot_data['layout'] and 'title' in plot_data['layout']['xaxis']:
                        fig.update_xaxes(title_text=plot_data['layout']['xaxis']['title']['text'], row=row, col=col)
                    if 'yaxis' in plot_data['layout'] and 'title' in plot_data['layout']['yaxis']:
                        fig.update_yaxes(title_text=plot_data['layout']['yaxis']['title']['text'], row=row, col=col)
            except:
                # 如果不是Plotly JSON文件，则尝试作为图像添加
                try:
                    # 读取图像
                    with open(plot, 'rb') as f:
                        img_bytes = f.read()
                    
                    # 转换为base64
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # 添加图像
                    fig.add_trace(
                        go.Image(source=f"data:image/png;base64,{img_base64}"),
                        row=row, col=col
                    )
                except:
                    print(f"无法添加图表: {plot}")
        
        # 更新布局
        fig.update_layout(
            title=title,
            width=self.default_width * cols,
            height=self.default_height * rows,
            showlegend=False
        )
        
        # 保存交互式仪表板
        output_path = os.path.join(self.output_dir, f"{file_name}.html")
        pio.write_html(fig, file=output_path)
        
        self._add_to_history("创建交互式仪表板", {'plots': plots, 'layout': layout}, output_path)
        
        return output_path

# 测试代码
if __name__ == "__main__":
    visualizer = DataVisualizer()
    # 测试可视化
    # df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    # visualizer.load_data({'data': df})
    # visualizer.plot_bar('A', 'B')
