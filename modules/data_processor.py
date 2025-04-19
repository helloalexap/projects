#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理模块 - 负责对提取的数据进行清洗、转换和分析
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats

class DataProcessor:
    """数据处理器类，用于对提取的数据进行清洗、转换和分析"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.data = None
        self.original_data = None
        self.metadata = {}
        self.text = None
        self.processing_history = []
        
    def load_data(self, extraction_result):
        """加载提取的数据"""
        if 'error' in extraction_result:
            raise ValueError(f"数据加载错误: {extraction_result['error']}")
        
        self.data = extraction_result.get('data')
        self.metadata = extraction_result.get('metadata', {})
        self.text = extraction_result.get('text')
        
        # 保存原始数据的副本
        if isinstance(self.data, pd.DataFrame):
            self.original_data = self.data.copy()
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            self.original_data = [df.copy() for df in self.data]
        else:
            self.original_data = self.data
            
        self._add_to_history("数据加载完成")
        
        return self
    
    def _add_to_history(self, operation, details=None):
        """添加处理历史记录"""
        self.processing_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation,
            'details': details
        })
    
    def get_processing_history(self):
        """获取处理历史记录"""
        return self.processing_history
    
    def get_data_info(self):
        """获取数据信息"""
        info = {
            'metadata': self.metadata,
            'has_text': self.text is not None and len(self.text) > 0
        }
        
        if isinstance(self.data, pd.DataFrame):
            info['data_type'] = 'dataframe'
            info['shape'] = self.data.shape
            info['columns'] = list(self.data.columns)
            info['dtypes'] = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            info['missing_values'] = self.data.isnull().sum().to_dict()
            info['sample'] = self.data.head(5).to_dict('records')
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            info['data_type'] = 'dataframe_list'
            info['table_count'] = len(self.data)
            info['tables'] = []
            for i, df in enumerate(self.data):
                table_info = {
                    'index': i,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'missing_values': df.isnull().sum().to_dict(),
                    'sample': df.head(5).to_dict('records')
                }
                info['tables'].append(table_info)
        elif isinstance(self.data, dict):
            info['data_type'] = 'dict'
            info['keys'] = list(self.data.keys())
        elif self.data is None and self.text:
            info['data_type'] = 'text_only'
            info['text_length'] = len(self.text)
            info['text_sample'] = self.text[:200] + ('...' if len(self.text) > 200 else '')
        else:
            info['data_type'] = str(type(self.data))
            
        return info
    
    def clean_data(self, table_index=None, **kwargs):
        """清洗数据"""
        if table_index is not None:
            if isinstance(self.data, list) and 0 <= table_index < len(self.data):
                df = self.data[table_index]
                self.data[table_index] = self._clean_dataframe(df, **kwargs)
                self._add_to_history(f"清洗表格 #{table_index}", kwargs)
            else:
                raise ValueError(f"无效的表格索引: {table_index}")
        elif isinstance(self.data, pd.DataFrame):
            self.data = self._clean_dataframe(self.data, **kwargs)
            self._add_to_history("清洗数据", kwargs)
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            self.data = [self._clean_dataframe(df, **kwargs) for df in self.data]
            self._add_to_history("清洗所有表格", kwargs)
        else:
            raise ValueError("数据格式不支持清洗操作")
            
        return self
    
    def _clean_dataframe(self, df, **kwargs):
        """清洗DataFrame"""
        # 创建DataFrame的副本
        df = df.copy()
        
        # 处理列名
        if kwargs.get('clean_column_names', False):
            df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # 删除重复行
        if kwargs.get('drop_duplicates', False):
            df = df.drop_duplicates()
        
        # 处理缺失值
        missing_strategy = kwargs.get('missing_strategy', 'none')
        if missing_strategy != 'none':
            df = self._handle_missing_values(df, strategy=missing_strategy)
        
        # 删除指定列
        columns_to_drop = kwargs.get('drop_columns', [])
        if columns_to_drop:
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # 数据类型转换
        type_conversions = kwargs.get('type_conversions', {})
        if type_conversions:
            df = self._convert_data_types(df, type_conversions)
        
        # 过滤行
        filter_conditions = kwargs.get('filter_conditions', [])
        if filter_conditions:
            df = self._filter_rows(df, filter_conditions)
        
        # 处理异常值
        outlier_strategy = kwargs.get('outlier_strategy', 'none')
        if outlier_strategy != 'none':
            outlier_columns = kwargs.get('outlier_columns', df.select_dtypes(include=[np.number]).columns.tolist())
            df = self._handle_outliers(df, columns=outlier_columns, strategy=outlier_strategy)
        
        return df
    
    def _clean_column_name(self, column_name):
        """清洗列名"""
        # 转换为小写
        column_name = str(column_name).lower()
        # 替换空格和特殊字符为下划线
        column_name = re.sub(r'[^\w\s]', '', column_name)
        column_name = re.sub(r'\s+', '_', column_name)
        # 确保列名不以数字开头
        if column_name and column_name[0].isdigit():
            column_name = 'col_' + column_name
        return column_name
    
    def _handle_missing_values(self, df, strategy='drop'):
        """处理缺失值"""
        if strategy == 'drop':
            # 删除包含缺失值的行
            return df.dropna()
        elif strategy == 'drop_columns':
            # 删除包含缺失值的列
            return df.dropna(axis=1)
        elif strategy == 'fill_mean':
            # 使用均值填充数值列的缺失值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                imputer = SimpleImputer(strategy='mean')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        elif strategy == 'fill_median':
            # 使用中位数填充数值列的缺失值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        elif strategy == 'fill_mode':
            # 使用众数填充所有列的缺失值
            for col in df.columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
            return df
        elif strategy == 'fill_zero':
            # 使用0填充缺失值
            return df.fillna(0)
        elif strategy == 'fill_empty':
            # 使用空字符串填充缺失值
            return df.fillna('')
        else:
            # 不处理缺失值
            return df
    
    def _convert_data_types(self, df, type_conversions):
        """转换数据类型"""
        for col, dtype in type_conversions.items():
            if col in df.columns:
                try:
                    if dtype == 'int':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'str':
                        df[col] = df[col].astype(str)
                    elif dtype == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif dtype == 'bool':
                        df[col] = df[col].astype(bool)
                except Exception as e:
                    print(f"转换列 '{col}' 到类型 '{dtype}' 时出错: {str(e)}")
        return df
    
    def _filter_rows(self, df, filter_conditions):
        """根据条件过滤行"""
        for condition in filter_conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if column not in df.columns:
                continue
                
            if operator == 'eq':
                df = df[df[column] == value]
            elif operator == 'ne':
                df = df[df[column] != value]
            elif operator == 'gt':
                df = df[df[column] > value]
            elif operator == 'lt':
                df = df[df[column] < value]
            elif operator == 'ge':
                df = df[df[column] >= value]
            elif operator == 'le':
                df = df[df[column] <= value]
            elif operator == 'contains':
                df = df[df[column].astype(str).str.contains(str(value), na=False)]
            elif operator == 'not_contains':
                df = df[~df[column].astype(str).str.contains(str(value), na=False)]
            elif operator == 'starts_with':
                df = df[df[column].astype(str).str.startswith(str(value), na=False)]
            elif operator == 'ends_with':
                df = df[df[column].astype(str).str.endswith(str(value), na=False)]
            elif operator == 'is_null':
                df = df[df[column].isnull()]
            elif operator == 'is_not_null':
                df = df[~df[column].isnull()]
                
        return df
    
    def _handle_outliers(self, df, columns, strategy='zscore'):
        """处理异常值"""
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'zscore':
                    # 使用Z-score方法识别异常值（|z| > 3）
                    z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                    df = df[(z_scores < 3) | np.isnan(z_scores)]
                elif strategy == 'iqr':
                    # 使用IQR方法识别异常值
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif strategy == 'cap':
                    # 将异常值限制在一定范围内
                    Q1 = df[col].quantile(0.05)
                    Q3 = df[col].quantile(0.95)
                    df[col] = df[col].clip(lower=Q1, upper=Q3)
                    
        return df
    
    def transform_data(self, table_index=None, **kwargs):
        """转换数据"""
        if table_index is not None:
            if isinstance(self.data, list) and 0 <= table_index < len(self.data):
                df = self.data[table_index]
                self.data[table_index] = self._transform_dataframe(df, **kwargs)
                self._add_to_history(f"转换表格 #{table_index}", kwargs)
            else:
                raise ValueError(f"无效的表格索引: {table_index}")
        elif isinstance(self.data, pd.DataFrame):
            self.data = self._transform_dataframe(self.data, **kwargs)
            self._add_to_history("转换数据", kwargs)
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            self.data = [self._transform_dataframe(df, **kwargs) for df in self.data]
            self._add_to_history("转换所有表格", kwargs)
        else:
            raise ValueError("数据格式不支持转换操作")
            
        return self
    
    def _transform_dataframe(self, df, **kwargs):
        """转换DataFrame"""
        # 创建DataFrame的副本
        df = df.copy()
        
        # 选择列
        select_columns = kwargs.get('select_columns')
        if select_columns:
            df = df[[col for col in select_columns if col in df.columns]]
        
        # 重命名列
        rename_columns = kwargs.get('rename_columns', {})
        if rename_columns:
            df = df.rename(columns={old: new for old, new in rename_columns.items() if old in df.columns})
        
        # 排序
        sort_by = kwargs.get('sort_by')
        if sort_by:
            ascending = kwargs.get('ascending', True)
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # 分组聚合
        group_by = kwargs.get('group_by')
        if group_by:
            agg_functions = kwargs.get('agg_functions', {})
            if agg_functions:
                df = df.groupby(group_by).agg(agg_functions).reset_index()
        
        # 数据规范化
        normalize = kwargs.get('normalize')
        if normalize:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if normalize == 'standard':
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            elif normalize == 'minmax':
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 数据透视表
        pivot = kwargs.get('pivot')
        if pivot:
            index = pivot.get('index')
            columns = pivot.get('columns')
            values = pivot.get('values')
            aggfunc = pivot.get('aggfunc', 'mean')
            
            if index and columns and values:
                try:
                    df = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
                    df = df.reset_index()
                except Exception as e:
                    print(f"创建数据透视表时出错: {str(e)}")
        
        # 合并列
        merge_columns = kwargs.get('merge_columns')
        if merge_columns:
            source_columns = merge_columns.get('source_columns', [])
            target_column = merge_columns.get('target_column')
            separator = merge_columns.get('separator', ' ')
            
            if source_columns and target_column:
                # 确保所有源列都存在
                existing_source_columns = [col for col in source_columns if col in df.columns]
                if existing_source_columns:
                    df[target_column] = df[existing_source_columns].astype(str).agg(separator.join, axis=1)
        
        # 拆分列
        split_column = kwargs.get('split_column')
        if split_column:
            source_column = split_column.get('source_column')
            target_columns = split_column.get('target_columns', [])
            separator = split_column.get('separator', ',')
            
            if source_column in df.columns and target_columns:
                try:
                    split_df = df[source_column].str.split(separator, expand=True)
                    # 限制拆分列数不超过目标列数
                    split_df = split_df.iloc[:, :len(target_columns)]
                    # 重命名拆分后的列
                    split_df.columns = target_columns[:split_df.shape[1]]
                    # 合并回原始DataFrame
                    df = pd.concat([df, split_df], axis=1)
                except Exception as e:
                    print(f"拆分列 '{source_column}' 时出错: {str(e)}")
        
        # 应用函数
        apply_function = kwargs.get('apply_function')
        if apply_function:
            column = apply_function.get('column')
            function = apply_function.get('function')
            new_column = apply_function.get('new_column')
            
            if column in df.columns and function and new_column:
                try:
                    if function == 'upper':
                        df[new_column] = df[column].astype(str).str.upper()
                    elif function == 'lower':
                        df[new_column] = df[column].astype(str).str.lower()
                    elif function == 'length':
                        df[new_column] = df[column].astype(str).str.len()
                    elif function == 'abs':
                        df[new_column] = df[column].abs()
                    elif function == 'log':
                        df[new_column] = np.log(df[column])
                    elif function == 'sqrt':
                        df[new_column] = np.sqrt(df[column])
                    elif function == 'square':
                        df[new_column] = df[column] ** 2
                except Exception as e:
                    print(f"应用函数 '{function}' 到列 '{column}' 时出错: {str(e)}")
        
        # 二值化
        binarize = kwargs.get('binarize')
        if binarize:
            column = binarize.get('column')
            threshold = binarize.get('threshold', 0)
            new_column = binarize.get('new_column')
            
            if column in df.columns and new_column:
                try:
                    df[new_column] = (df[column] > threshold).astype(int)
                except Exception as e:
                    print(f"二值化列 '{column}' 时出错: {str(e)}")
        
        # 独热编码
        one_hot = kwargs.get('one_hot')
        if one_hot:
            columns = one_hot if isinstance(one_hot, list) else [one_hot]
            existing_columns = [col for col in columns if col in df.columns]
            
            if existing_columns:
                try:
                    # 对每个指定的列进行独热编码
                    for col in existing_columns:
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                except Exception as e:
                    print(f"对列 {existing_columns} 进行独热编码时出错: {str(e)}")
        
        return df
    
    def analyze_data(self, table_index=None, **kwargs):
        """分析数据"""
        analysis_results = {}
        
        if table_index is not None:
            if isinstance(self.data, list) and 0 <= table_index < len(self.data):
                df = self.data[table_index]
                analysis_results = self._analyze_dataframe(df, **kwargs)
                self._add_to_history(f"分析表格 #{table_index}", kwargs)
            else:
                raise ValueError(f"无效的表格索引: {table_index}")
        elif isinstance(self.data, pd.DataFrame):
            analysis_results = self._analyze_dataframe(self.data, **kwargs)
            self._add_to_history("分析数据", kwargs)
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            analysis_results = [self._analyze_dataframe(df, **kwargs) for df in self.data]
            self._add_to_history("分析所有表格", kwargs)
        else:
            if self.text:
                analysis_results = self._analyze_text(self.text, **kwargs)
                self._add_to_history("分析文本", kwargs)
            else:
                raise ValueError("数据格式不支持分析操作")
            
        return analysis_results
    
    def _analyze_dataframe(self, df, **kwargs):
        """分析DataFrame"""
        analysis_type = kwargs.get('analysis_type', 'descriptive')
        results = {}
        
        if analysis_type == 'descriptive':
            # 描述性统计分析
            results['summary'] = self._descriptive_analysis(df)
        elif analysis_type == 'correlation':
            # 相关性分析
            results['correlation'] = self._correlation_analysis(df, **kwargs)
        elif analysis_type == 'pca':
            # 主成分分析
            results['pca'] = self._pca_analysis(df, **kwargs)
        elif analysis_type == 'clustering':
            # 聚类分析
            results['clustering'] = self._clustering_analysis(df, **kwargs)
        elif analysis_type == 'regression':
            # 回归分析
            results['regression'] = self._regression_analysis(df, **kwargs)
        elif analysis_type == 'time_series':
            # 时间序列分析
            results['time_series'] = self._time_series_analysis(df, **kwargs)
        elif analysis_type == 'comprehensive':
            # 综合分析（包含多种分析）
            results['summary'] = self._descriptive_analysis(df)
            results['correlation'] = self._correlation_analysis(df, **kwargs)
            
            # 检查是否有足够的数值列进行PCA
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                results['pca'] = self._pca_analysis(df, **kwargs)
            
            # 检查是否有足够的数据进行聚类
            if len(df) > 10 and len(numeric_cols) >= 2:
                results['clustering'] = self._clustering_analysis(df, **kwargs)
        
        return results
    
    def _descriptive_analysis(self, df):
        """描述性统计分析"""
        results = {}
        
        # 基本信息
        results['shape'] = df.shape
        results['columns'] = list(df.columns)
        
        # 数值列的描述性统计
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            results['numeric_stats'] = numeric_df.describe().to_dict()
            
            # 计算偏度和峰度
            skewness = numeric_df.skew().to_dict()
            kurtosis = numeric_df.kurtosis().to_dict()
            results['skewness'] = skewness
            results['kurtosis'] = kurtosis
        
        # 分类列的描述性统计
        categorical_df = df.select_dtypes(exclude=[np.number])
        if not categorical_df.empty:
            results['categorical_stats'] = {}
            for col in categorical_df.columns:
                value_counts = df[col].value_counts().to_dict()
                unique_count = len(value_counts)
                most_common = df[col].value_counts().index[0] if not df[col].value_counts().empty else None
                results['categorical_stats'][col] = {
                    'unique_count': unique_count,
                    'most_common': most_common,
                    'value_counts': value_counts if unique_count <= 10 else {k: value_counts[k] for k in list(value_counts.keys())[:10]}
                }
        
        # 缺失值统计
        missing_values = df.isnull().sum().to_dict()
        results['missing_values'] = missing_values
        results['missing_percentage'] = {col: count / len(df) * 100 for col, count in missing_values.items()}
        
        return results
    
    def _correlation_analysis(self, df, **kwargs):
        """相关性分析"""
        results = {}
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "没有数值列可以进行相关性分析"}
        
        # 计算相关系数
        method = kwargs.get('correlation_method', 'pearson')
        correlation_matrix = numeric_df.corr(method=method).round(4).to_dict()
        results['correlation_matrix'] = correlation_matrix
        
        # 找出高相关性的特征对
        threshold = kwargs.get('correlation_threshold', 0.7)
        high_correlations = []
        
        for i, col1 in enumerate(numeric_df.columns):
            for j, col2 in enumerate(numeric_df.columns):
                if i < j:  # 避免重复和自相关
                    corr = abs(numeric_df[col1].corr(numeric_df[col2], method=method))
                    if corr >= threshold:
                        high_correlations.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': round(corr, 4)
                        })
        
        results['high_correlations'] = sorted(high_correlations, key=lambda x: x['correlation'], reverse=True)
        
        return results
    
    def _pca_analysis(self, df, **kwargs):
        """主成分分析"""
        results = {}
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "没有数值列可以进行PCA分析"}
        
        # 处理缺失值
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(numeric_df)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 确定组件数量
        n_components = kwargs.get('n_components', min(X_scaled.shape[1], 5))
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 保存结果
        results['n_components'] = n_components
        results['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
        results['cumulative_explained_variance'] = np.cumsum(pca.explained_variance_ratio_).tolist()
        
        # 计算特征贡献
        loadings = pca.components_
        feature_names = numeric_df.columns
        
        components_dict = {}
        for i, component in enumerate(loadings):
            components_dict[f'PC{i+1}'] = {feature: round(loading, 4) for feature, loading in zip(feature_names, component)}
        
        results['components'] = components_dict
        
        # 计算最佳组件数量
        optimal_n_components = np.argmax(results['cumulative_explained_variance'] >= 0.8) + 1
        if optimal_n_components == 0:  # 如果所有组件的累积方差解释率都小于0.8
            optimal_n_components = len(results['cumulative_explained_variance'])
        
        results['optimal_n_components'] = optimal_n_components
        
        return results
    
    def _clustering_analysis(self, df, **kwargs):
        """聚类分析"""
        results = {}
        
        # 选择数值列
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "没有数值列可以进行聚类分析"}
        
        # 处理缺失值
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(numeric_df)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 确定聚类数量
        n_clusters = kwargs.get('n_clusters', min(5, len(df) // 10 + 1))
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 保存结果
        results['n_clusters'] = n_clusters
        results['cluster_centers'] = kmeans.cluster_centers_.tolist()
        results['inertia'] = kmeans.inertia_
        
        # 计算每个聚类的样本数量
        cluster_counts = np.bincount(clusters)
        results['cluster_counts'] = cluster_counts.tolist()
        
        # 计算每个聚类的特征均值
        cluster_means = {}
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        for i in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            cluster_means[f'cluster_{i}'] = cluster_data.select_dtypes(include=[np.number]).mean().to_dict()
        
        results['cluster_means'] = cluster_means
        
        return results
    
    def _regression_analysis(self, df, **kwargs):
        """回归分析"""
        results = {}
        
        # 获取目标变量和特征
        target = kwargs.get('target')
        features = kwargs.get('features', [])
        
        if not target or target not in df.columns:
            return {"error": "未指定有效的目标变量"}
        
        # 如果未指定特征，则使用所有数值列（除目标变量外）
        if not features:
            features = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]
        
        # 确保所有特征都存在
        features = [col for col in features if col in df.columns]
        
        if not features:
            return {"error": "没有有效的特征变量"}
        
        # 准备数据
        X = df[features]
        y = df[target]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # 执行OLS回归
        try:
            model = sm.OLS(y, X).fit()
            
            # 保存结果
            results['model_summary'] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic
            }
            
            # 系数和p值
            coefficients = []
            for i, var in enumerate(model.params.index):
                coefficients.append({
                    'variable': var,
                    'coefficient': model.params[i],
                    'std_err': model.bse[i],
                    't_value': model.tvalues[i],
                    'p_value': model.pvalues[i],
                    'significance': '***' if model.pvalues[i] < 0.001 else
                                   '**' if model.pvalues[i] < 0.01 else
                                   '*' if model.pvalues[i] < 0.05 else
                                   '.' if model.pvalues[i] < 0.1 else ''
                })
            
            results['coefficients'] = coefficients
            
            # 预测值和残差
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            results['residuals_summary'] = {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max()
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _time_series_analysis(self, df, **kwargs):
        """时间序列分析"""
        results = {}
        
        # 获取时间列和值列
        time_col = kwargs.get('time_column')
        value_col = kwargs.get('value_column')
        
        if not time_col or time_col not in df.columns:
            return {"error": "未指定有效的时间列"}
        
        if not value_col or value_col not in df.columns:
            return {"error": "未指定有效的值列"}
        
        # 确保时间列是日期时间类型
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return {"error": "无法将时间列转换为日期时间类型"}
        
        # 按时间排序
        df = df.sort_values(by=time_col)
        
        # 设置时间索引
        ts_df = df[[time_col, value_col]].set_index(time_col)
        
        # 基本统计信息
        results['basic_stats'] = {
            'mean': ts_df[value_col].mean(),
            'std': ts_df[value_col].std(),
            'min': ts_df[value_col].min(),
            'max': ts_df[value_col].max(),
            'start_date': ts_df.index.min().strftime('%Y-%m-%d'),
            'end_date': ts_df.index.max().strftime('%Y-%m-%d'),
            'duration_days': (ts_df.index.max() - ts_df.index.min()).days
        }
        
        # 检测时间序列的频率
        try:
            freq = pd.infer_freq(ts_df.index)
            results['frequency'] = freq if freq else "不规则"
        except:
            results['frequency'] = "不规则"
        
        # 计算移动平均
        window = kwargs.get('window', 7)
        try:
            ts_df['rolling_mean'] = ts_df[value_col].rolling(window=window).mean()
            results['rolling_mean'] = {
                'window': window,
                'last_values': ts_df['rolling_mean'].tail(5).to_dict()
            }
        except:
            pass
        
        # 计算同比/环比增长率
        try:
            # 计算环比增长率（与前一期相比）
            ts_df['pct_change'] = ts_df[value_col].pct_change() * 100
            results['pct_change'] = {
                'mean': ts_df['pct_change'].mean(),
                'std': ts_df['pct_change'].std(),
                'last_values': ts_df['pct_change'].tail(5).to_dict()
            }
            
            # 尝试计算同比增长率（与去年同期相比）
            if results['frequency'] in ['M', 'MS', 'D', 'B']:
                try:
                    ts_df['yoy_pct_change'] = ts_df[value_col].pct_change(periods=12 if results['frequency'] in ['M', 'MS'] else 365) * 100
                    results['yoy_pct_change'] = {
                        'mean': ts_df['yoy_pct_change'].mean(),
                        'std': ts_df['yoy_pct_change'].std(),
                        'last_values': ts_df['yoy_pct_change'].tail(5).to_dict()
                    }
                except:
                    pass
        except:
            pass
        
        # 检测季节性和趋势
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # 确保没有缺失值
            ts_clean = ts_df[value_col].fillna(method='ffill').fillna(method='bfill')
            
            # 如果数据点足够多，尝试进行季节性分解
            if len(ts_clean) >= max(2 * window, 12):
                decomposition = seasonal_decompose(ts_clean, model='additive', period=min(window, len(ts_clean) // 2))
                
                results['decomposition'] = {
                    'trend': decomposition.trend.tail(5).to_dict(),
                    'seasonal': decomposition.seasonal.tail(5).to_dict(),
                    'residual': decomposition.resid.tail(5).to_dict()
                }
        except:
            pass
        
        return results
    
    def _analyze_text(self, text, **kwargs):
        """分析文本"""
        results = {}
        
        # 基本统计
        results['basic_stats'] = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1
        }
        
        # 词频分析
        if kwargs.get('word_frequency', False):
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 1:  # 忽略单个字符
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 按频率排序
            sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            results['word_frequency'] = dict(sorted_word_freq[:50])  # 只返回前50个
        
        # 情感分析
        if kwargs.get('sentiment_analysis', False):
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                results['sentiment'] = {
                    'polarity': blob.sentiment.polarity,  # -1到1之间，负面到正面
                    'subjectivity': blob.sentiment.subjectivity  # 0到1之间，客观到主观
                }
            except:
                pass
        
        # 关键词提取
        if kwargs.get('keyword_extraction', False):
            try:
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(text, topK=20)
                results['keywords'] = keywords
            except:
                try:
                    from rake_nltk import Rake
                    rake = Rake()
                    rake.extract_keywords_from_text(text)
                    results['keywords'] = rake.get_ranked_phrases()[:20]
                except:
                    pass
        
        return results
    
    def merge_tables(self, table_indices=None, **kwargs):
        """合并多个表格"""
        if not isinstance(self.data, list) or not all(isinstance(item, pd.DataFrame) for item in self.data):
            raise ValueError("数据不是DataFrame列表，无法执行合并操作")
        
        # 如果未指定表格索引，则合并所有表格
        if table_indices is None:
            table_indices = list(range(len(self.data)))
        
        # 确保所有索引都有效
        valid_indices = [i for i in table_indices if 0 <= i < len(self.data)]
        if not valid_indices:
            raise ValueError("没有有效的表格索引")
        
        # 获取要合并的表格
        tables_to_merge = [self.data[i] for i in valid_indices]
        
        # 获取合并方法
        merge_method = kwargs.get('merge_method', 'concat')
        
        if merge_method == 'concat':
            # 纵向合并（堆叠）
            axis = kwargs.get('axis', 0)  # 默认纵向堆叠
            ignore_index = kwargs.get('ignore_index', True)
            
            merged_df = pd.concat(tables_to_merge, axis=axis, ignore_index=ignore_index)
            
        elif merge_method == 'merge':
            # 横向合并（基于键）
            on = kwargs.get('on')  # 合并键
            how = kwargs.get('how', 'inner')  # 合并方式
            
            if not on:
                raise ValueError("使用merge方法时必须指定'on'参数")
            
            # 从第一个表格开始，依次与其他表格合并
            merged_df = tables_to_merge[0]
            for df in tables_to_merge[1:]:
                merged_df = pd.merge(merged_df, df, on=on, how=how)
        
        else:
            raise ValueError(f"不支持的合并方法: {merge_method}")
        
        # 更新数据
        self.data = merged_df
        self._add_to_history("合并表格", {
            'table_indices': valid_indices,
            'merge_method': merge_method,
            **{k: v for k, v in kwargs.items() if k not in ['merge_method']}
        })
        
        return self
    
    def generate_insights(self, **kwargs):
        """生成数据洞察"""
        insights = {
            'summary': [],
            'patterns': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # 处理DataFrame
        if isinstance(self.data, pd.DataFrame):
            df = self.data
            self._generate_dataframe_insights(df, insights, **kwargs)
        
        # 处理DataFrame列表
        elif isinstance(self.data, list) and all(isinstance(item, pd.DataFrame) for item in self.data):
            for i, df in enumerate(self.data):
                table_insights = {
                    'summary': [],
                    'patterns': [],
                    'anomalies': [],
                    'recommendations': []
                }
                self._generate_dataframe_insights(df, table_insights, **kwargs)
                
                # 添加表格标识
                for key in table_insights:
                    for item in table_insights[key]:
                        insights[key].append({
                            'table_index': i,
                            'content': item
                        })
        
        # 处理文本
        elif self.text:
            self._generate_text_insights(self.text, insights, **kwargs)
        
        else:
            insights['summary'].append("无法为当前数据格式生成洞察")
        
        return insights
    
    def _generate_dataframe_insights(self, df, insights, **kwargs):
        """为DataFrame生成洞察"""
        # 基本摘要
        row_count = len(df)
        col_count = len(df.columns)
        insights['summary'].append(f"数据集包含{row_count}行和{col_count}列")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            missing_cols = missing_values[missing_values > 0]
            insights['summary'].append(f"发现{len(missing_cols)}列存在缺失值，总计{missing_values.sum()}个缺失值")
            
            # 缺失值比例高的列
            high_missing_cols = missing_cols[missing_cols / row_count > 0.2]
            if not high_missing_cols.empty:
                for col, count in high_missing_cols.items():
                    insights['anomalies'].append(f"列'{col}'的缺失值比例高达{count/row_count:.1%}")
                    insights['recommendations'].append(f"考虑处理'{col}'列的缺失值，或者评估是否可以删除该列")
        
        # 数值列分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            # 检查异常值
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if len(outliers) > 0 and len(outliers) / row_count > 0.01:
                    insights['anomalies'].append(f"列'{col}'中发现{len(outliers)}个异常值（{len(outliers)/row_count:.1%}）")
                    if len(outliers) / row_count > 0.05:
                        insights['recommendations'].append(f"建议检查'{col}'列的异常值，可能需要进行数据清洗")
            
            # 检查高相关性
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    for col1, col2, corr in high_corr_pairs:
                        insights['patterns'].append(f"'{col1}'和'{col2}'之间存在{corr:.2f}的高相关性")
                    
                    if len(high_corr_pairs) > 2:
                        insights['recommendations'].append("存在多对高相关特征，考虑使用降维技术如PCA来减少特征冗余")
        
        # 分类列分析
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                unique_count = len(value_counts)
                
                # 检查基数过高的分类列
                if unique_count > 100 and unique_count / row_count > 0.5:
                    insights['anomalies'].append(f"列'{col}'的唯一值数量（{unique_count}）过高，可能不适合作为分类变量")
                
                # 检查分布不均的分类列
                if unique_count > 1 and unique_count <= 10:
                    top_value = value_counts.index[0]
                    top_percentage = value_counts.iloc[0] / row_count
                    
                    if top_percentage > 0.8:
                        insights['patterns'].append(f"列'{col}'中，值'{top_value}'占比高达{top_percentage:.1%}，分布极不均衡")
                        insights['recommendations'].append(f"列'{col}'分布不均衡，在建模时可能需要特殊处理")
        
        # 时间相关分析
        date_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)
                else:
                    # 尝试转换为日期时间类型
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
            except:
                continue
        
        if date_cols:
            for col in date_cols:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    insights['recommendations'].append(f"列'{col}'可能包含日期时间信息，建议转换为日期时间类型以便进行时间序列分析")
                else:
                    date_range = df[col].max() - df[col].min()
                    insights['summary'].append(f"列'{col}'的时间跨度为{date_range.days}天")
                    
                    # 检查时间间隔
                    try:
                        freq = pd.infer_freq(df[col].sort_values())
                        if freq:
                            insights['patterns'].append(f"列'{col}'的时间频率为{freq}")
                        else:
                            insights['anomalies'].append(f"列'{col}'的时间间隔不规则")
                    except:
                        pass
        
        # 数据完整性检查
        if row_count > 0:
            # 检查重复行
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                insights['anomalies'].append(f"发现{duplicate_count}行重复数据（{duplicate_count/row_count:.1%}）")
                insights['recommendations'].append("建议处理重复行，可以考虑删除或标记")
        
        # 生成综合建议
        if not insights['recommendations']:
            if row_count > 0 and col_count > 0:
                insights['recommendations'].append("数据质量良好，可以直接进行进一步分析")
                
                # 根据数据特点推荐分析方法
                if len(numeric_cols) >= 2:
                    insights['recommendations'].append("可以考虑进行相关性分析和特征重要性分析")
                
                if len(date_cols) > 0 and len(numeric_cols) > 0:
                    insights['recommendations'].append("数据包含时间信息，可以考虑进行时间序列分析")
                
                if row_count > 100 and len(numeric_cols) >= 3:
                    insights['recommendations'].append("数据量充足且有多个数值特征，可以考虑使用聚类或分类算法进行建模")
    
    def _generate_text_insights(self, text, insights, **kwargs):
        """为文本生成洞察"""
        # 基本摘要
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1
        
        insights['summary'].append(f"文本包含{char_count}个字符，约{word_count}个词，{line_count}行")
        
        # 检查文本长度
        if char_count < 100:
            insights['anomalies'].append("文本内容较短，可能不足以进行深入分析")
        
        # 检查是否包含表格结构
        if '|' in text and '-+-' in text or '\t' in text:
            insights['patterns'].append("文本中可能包含表格结构")
            insights['recommendations'].append("建议提取文本中的表格数据进行单独分析")
        
        # 检查是否包含JSON或XML结构
        if ('{' in text and '}' in text and ':' in text) or ('<' in text and '>' in text and '</' in text):
            insights['patterns'].append("文本中可能包含结构化数据（JSON或XML）")
            insights['recommendations'].append("建议提取文本中的结构化数据进行解析")
        
        # 检查是否包含URL
        url_pattern = re.compile(r'https?://\S+')
        urls = url_pattern.findall(text)
        if urls:
            insights['patterns'].append(f"文本中包含{len(urls)}个URL链接")
        
        # 检查是否包含数字数据
        number_pattern = re.compile(r'\b\d+\.?\d*\b')
        numbers = number_pattern.findall(text)
        if len(numbers) > word_count * 0.1:
            insights['patterns'].append(f"文本中包含大量数字（{len(numbers)}个），可能包含数据报告或统计信息")
            insights['recommendations'].append("建议提取文本中的数字数据进行量化分析")
        
        # 检查是否包含日期
        date_pattern = re.compile(r'\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}\b|\b\d{1,2}[-/月]\d{1,2}[-/日]?\b')
        dates = date_pattern.findall(text)
        if dates:
            insights['patterns'].append(f"文本中包含{len(dates)}个日期")
            if len(dates) > 3:
                insights['recommendations'].append("文本包含多个日期，可能适合进行时间序列相关分析")
        
        # 生成综合建议
        if not insights['recommendations']:
            insights['recommendations'].append("可以考虑使用文本挖掘技术进行关键词提取和主题建模")
            insights['recommendations'].append("对于长文本，建议进行文本摘要以提取核心信息")

# 测试代码
if __name__ == "__main__":
    processor = DataProcessor()
    # 测试数据处理
    # processor.load_data({'data': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})})
    # processor.clean_data(drop_duplicates=True)
    # print(processor.data)
