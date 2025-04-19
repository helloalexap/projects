# 数据分析处理器 - 部署和使用指南

## 项目概述

数据分析处理器是一个功能强大的数据分析工具，可以帮助用户处理各种格式的数据，生成可视化图表，并导出多种格式的分析报告。该工具具有以下特点：

- 支持多种数据格式的提取和处理（CSV、Excel、JSON、文本、PDF等）
- 提供丰富的数据清洗和转换功能
- 支持多种可视化图表（柱状图、折线图、散点图、热力图、词云图等）
- 支持多种导出格式（CSV、Excel、JSON、HTML、PDF、Markdown等）
- 提供综合数据分析报告，包含数据洞察和建议
- 用户友好的Web界面，操作简单直观

## 系统要求

- Python 3.8+
- 现代Web浏览器（Chrome、Firefox、Edge等）
- 至少2GB可用内存
- 至少500MB可用磁盘空间

## 安装指南

### 1. 安装依赖

首先，确保您的系统已安装Python 3.8或更高版本。然后，安装所需的Python包：

```bash
# 创建并激活虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖包
pip install flask pandas numpy matplotlib seaborn plotly wordcloud scikit-learn weasyprint pdfkit jinja2 openpyxl
```

对于PDF导出功能，您可能还需要安装额外的系统依赖：

- 对于WeasyPrint（默认PDF生成器）：
  - 在Ubuntu/Debian上：`sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info`
  - 在Windows上：请参考[WeasyPrint安装指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)

- 对于wkhtmltopdf（备选PDF生成器）：
  - 在Ubuntu/Debian上：`sudo apt-get install wkhtmltopdf`
  - 在Windows上：从[wkhtmltopdf官网](https://wkhtmltopdf.org/downloads.html)下载并安装

### 2. 下载项目

将项目文件下载到您的计算机上，或者使用git克隆仓库：

```bash
git clone https://github.com/yourusername/data-analyzer.git
cd data-analyzer
```

### 3. 配置项目

项目默认配置应该可以直接使用，但您可以根据需要修改`app.py`文件中的配置参数：

```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 限制上传文件大小为50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
app.config['EXPORT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
```

### 4. 运行应用

```bash
python app.py
```

应用将在本地启动，默认地址为：http://127.0.0.1:5000/

## 使用指南

### 1. 上传数据

1. 在首页点击"选择文件"按钮，选择要上传的数据文件
2. 支持的文件格式包括：CSV、Excel、JSON、TXT、PDF和图像文件
3. 点击"上传"按钮开始上传
4. 上传完成后，文件将显示在"已上传文件"列表中
5. 点击文件旁边的"提取数据"按钮开始数据提取

### 2. 处理数据

1. 数据提取成功后，点击右侧"处理数据"按钮
2. 在数据处理页面，您可以：
   - 查看数据预览
   - 设置数据清洗选项（清理列名、删除重复行等）
   - 选择缺失值处理策略
   - 选择分析类型
3. 点击"处理数据"按钮开始数据处理

### 3. 可视化数据

1. 数据处理成功后，点击右侧"可视化数据"按钮
2. 在数据可视化页面，您可以：
   - 选择图表类型（柱状图、折线图、散点图等）
   - 选择要可视化的表格（如果有多个）
   - 设置图表标题
   - 选择X轴、Y轴和颜色分组列
3. 点击"生成图表"按钮创建可视化
4. 生成的图表将显示在首页的"可视化结果"部分

### 4. 导出数据

1. 数据处理成功后，点击右侧"导出数据"按钮
2. 在导出数据页面，您可以：
   - 选择导出格式（CSV、Excel、JSON、HTML、PDF、Markdown等）
   - 设置文件名
   - 选择要导出的表格（如果有多个）
   - 选择是否包含可视化图表和文本内容
3. 点击"导出数据"按钮开始导出
4. 导出的文件将显示在首页的"导出文件"部分，您可以查看或下载

## 功能详解

### 数据提取

数据提取模块支持从多种格式的文件中提取数据：

- **CSV/Excel**：自动识别分隔符和表头，支持多个工作表
- **JSON**：解析JSON结构，支持嵌套数据
- **文本**：提取纯文本内容，支持基本文本分析
- **PDF**：提取PDF中的文本和表格
- **图像**：提取图像中的文本（OCR）和表格

### 数据处理

数据处理模块提供多种数据清洗和转换功能：

- **基本清洗**：清理列名、删除重复行
- **缺失值处理**：删除含缺失值的行、使用均值/中位数/众数填充、使用0填充
- **数据分析**：描述性统计、相关性分析、主成分分析、聚类分析、综合分析

### 数据可视化

数据可视化模块支持多种图表类型：

- **柱状图**：比较不同类别之间的数值大小
- **折线图**：展示数据随时间或顺序变化的趋势
- **散点图**：展示两个变量之间的关系
- **直方图**：展示数值分布情况
- **箱线图**：展示数据的分布情况，包括中位数、四分位数和异常值
- **热力图**：展示矩阵数据，通过颜色深浅表示数值大小
- **饼图**：展示各部分占整体的比例
- **词云图**：直观展示文本中词语出现的频率

### 数据导出

数据导出模块支持多种导出格式：

- **CSV**：逗号分隔值文件，适合表格数据
- **Excel**：Microsoft Excel格式，支持多个工作表
- **JSON**：结构化数据格式，适合程序处理
- **HTML**：网页格式，可在浏览器中查看
- **PDF**：便携式文档格式，适合打印和分享
- **Markdown**：轻量级标记语言，适合文档编写
- **图像**：将表格导出为图像格式
- **分析报告**：综合分析报告，包含数据、图表和洞察
- **打包下载**：将所有格式打包为ZIP文件下载

## 高级部署

### 使用Gunicorn和Nginx部署（Linux）

对于生产环境，建议使用Gunicorn作为WSGI服务器，并使用Nginx作为反向代理：

1. 安装Gunicorn：

```bash
pip install gunicorn
```

2. 创建Gunicorn启动脚本（`start.sh`）：

```bash
#!/bin/bash
cd /path/to/data-analyzer
source venv/bin/activate
exec gunicorn -w 4 -b 127.0.0.1:8000 app:app
```

3. 使脚本可执行：

```bash
chmod +x start.sh
```

4. 安装Nginx：

```bash
sudo apt-get install nginx
```

5. 创建Nginx配置文件（`/etc/nginx/sites-available/data-analyzer`）：

```
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /path/to/data-analyzer/static;
    }
}
```

6. 启用站点并重启Nginx：

```bash
sudo ln -s /etc/nginx/sites-available/data-analyzer /etc/nginx/sites-enabled
sudo systemctl restart nginx
```

### 使用Docker部署

您也可以使用Docker容器化应用：

1. 创建Dockerfile：

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    wkhtmltopdf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

2. 创建requirements.txt文件：

```
flask
pandas
numpy
matplotlib
seaborn
plotly
wordcloud
scikit-learn
weasyprint
pdfkit
jinja2
openpyxl
```

3. 构建并运行Docker镜像：

```bash
docker build -t data-analyzer .
docker run -p 5000:5000 data-analyzer
```

## 故障排除

### 常见问题

1. **上传文件失败**
   - 检查文件大小是否超过限制（默认50MB）
   - 确保文件格式受支持

2. **PDF导出失败**
   - 检查是否已安装WeasyPrint或wkhtmltopdf的系统依赖
   - 尝试在app.py中将PDF导出方法从'weasyprint'改为'wkhtmltopdf'

3. **图表生成失败**
   - 确保选择了正确的列类型（数值列/分类列）
   - 检查数据中是否存在异常值或格式问题

4. **应用无法启动**
   - 检查是否已安装所有依赖包
   - 检查端口5000是否被其他应用占用

### 获取帮助

如果您遇到其他问题，请：

1. 检查控制台日志以获取错误信息
2. 查阅项目文档和常见问题解答
3. 提交问题到项目的GitHub仓库

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 贡献

欢迎贡献代码、报告问题或提出改进建议。请通过GitHub仓库提交问题或拉取请求。

---

感谢使用数据分析处理器！希望它能帮助您更高效地分析和可视化数据。
