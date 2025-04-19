#!/bin/bash

# 数据分析处理器启动脚本

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
pip install flask pandas numpy matplotlib seaborn plotly wordcloud scikit-learn jinja2 openpyxl

# 尝试安装PDF相关依赖
echo "尝试安装PDF导出依赖..."
pip install weasyprint pdfkit

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data exports

# 启动应用
echo "启动数据分析处理器..."
python app.py
