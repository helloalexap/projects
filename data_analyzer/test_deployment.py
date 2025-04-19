#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据分析处理器 - 部署测试脚本
用于测试部署后的应用程序功能
"""

import os
import sys
import requests
import time
import json
import argparse
from urllib.parse import urljoin

def test_homepage(base_url):
    """测试首页是否正常加载"""
    print("测试首页...")
    response = requests.get(base_url)
    if response.status_code == 200:
        print("✅ 首页加载成功")
        return True
    else:
        print(f"❌ 首页加载失败: 状态码 {response.status_code}")
        return False

def test_static_resources(base_url):
    """测试静态资源是否正常加载"""
    print("测试静态资源...")
    css_url = urljoin(base_url, "static/css/style.css")
    response = requests.get(css_url)
    if response.status_code == 200:
        print("✅ 静态资源加载成功")
        return True
    else:
        print(f"❌ 静态资源加载失败: 状态码 {response.status_code}")
        return False

def test_process_page(base_url):
    """测试数据处理页面是否正常加载"""
    print("测试数据处理页面...")
    process_url = urljoin(base_url, "process")
    response = requests.get(process_url)
    if response.status_code == 200:
        print("✅ 数据处理页面加载成功")
        return True
    else:
        print(f"❌ 数据处理页面加载失败: 状态码 {response.status_code}")
        return False

def test_visualize_page(base_url):
    """测试数据可视化页面是否正常加载"""
    print("测试数据可视化页面...")
    visualize_url = urljoin(base_url, "visualize")
    response = requests.get(visualize_url)
    if response.status_code == 200:
        print("✅ 数据可视化页面加载成功")
        return True
    else:
        print(f"❌ 数据可视化页面加载失败: 状态码 {response.status_code}")
        return False

def test_export_page(base_url):
    """测试数据导出页面是否正常加载"""
    print("测试数据导出页面...")
    export_url = urljoin(base_url, "export")
    response = requests.get(export_url)
    if response.status_code == 200:
        print("✅ 数据导出页面加载成功")
        return True
    else:
        print(f"❌ 数据导出页面加载失败: 状态码 {response.status_code}")
        return False

def test_file_upload(base_url, test_file_path):
    """测试文件上传功能"""
    if not os.path.exists(test_file_path):
        print(f"❌ 测试文件不存在: {test_file_path}")
        return False
    
    print(f"测试文件上传... ({test_file_path})")
    upload_url = urljoin(base_url, "upload")
    
    with open(test_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(upload_url, files=files, allow_redirects=False)
    
    if response.status_code in [200, 302]:
        print("✅ 文件上传请求成功")
        return True
    else:
        print(f"❌ 文件上传失败: 状态码 {response.status_code}")
        return False

def run_all_tests(base_url, test_file_path=None):
    """运行所有测试"""
    print(f"开始测试部署的应用程序: {base_url}\n")
    
    results = []
    results.append(test_homepage(base_url))
    results.append(test_static_resources(base_url))
    results.append(test_process_page(base_url))
    results.append(test_visualize_page(base_url))
    results.append(test_export_page(base_url))
    
    if test_file_path:
        results.append(test_file_upload(base_url, test_file_path))
    
    success_count = results.count(True)
    total_count = len(results)
    
    print("\n测试摘要:")
    print(f"总测试数: {total_count}")
    print(f"成功测试数: {success_count}")
    print(f"失败测试数: {total_count - success_count}")
    
    if success_count == total_count:
        print("\n✅ 所有测试通过！应用程序已成功部署。")
        return True
    else:
        print("\n⚠️ 部分测试失败。请检查应用程序配置和日志。")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试部署的数据分析处理器应用程序")
    parser.add_argument("url", help="部署的应用程序基础URL")
    parser.add_argument("--file", help="用于测试上传功能的测试文件路径")
    
    args = parser.parse_args()
    run_all_tests(args.url, args.file)
