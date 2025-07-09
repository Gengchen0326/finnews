#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

def load_config():
    """加载配置文件"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return None

def check_single_file(file_path, columns):
    """检查单个数据文件"""
    try:
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 检查数据列是否存在
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"警告: 文件 {file_path} 中缺少列: {missing_cols}")
            return None
        
        # 提取所需列
        df = df[columns].copy()
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        
        # 检查无穷值
        inf_values = {col: np.isinf(df[col]).sum() for col in df.columns if np.issubdtype(df[col].dtype, np.number)}
        
        # 基本统计信息
        stats = df.describe()
        
        # 检查异常值 (使用IQR方法)
        outliers = {}
        for col in columns:
            if np.issubdtype(df[col].dtype, np.number):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        return {
            'file_name': os.path.basename(file_path),
            'shape': df.shape,
            'missing_values': missing_values.to_dict(),
            'inf_values': inf_values,
            'outliers': outliers,
            'stats': stats
        }
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def run_checks():
    """运行所有数据检查"""
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 获取数据路径和列名
    nonsentiment_config = config.get('nonsentiment_model', {})
    sentiment_config = config.get('sentiment_model', {})
    
    # 创建结果目录
    if not os.path.exists('data_check_results'):
        os.makedirs('data_check_results')
    
    # 检查非情感模型数据
    if nonsentiment_config:
        print("\n检查非情感模型数据:")
        data_path = nonsentiment_config.get('data_path', 'data')
        columns = nonsentiment_config.get('columns', [])
        
        # 获取所有CSV文件
        if os.path.isdir(data_path):
            csv_files = glob(os.path.join(data_path, '*.csv'))
        else:
            csv_files = [data_path] if data_path.endswith('.csv') else []
        
        if not csv_files:
            print(f"警告: 在 {data_path} 中未找到CSV文件")
        else:
            print(f"找到 {len(csv_files)} 个CSV文件")
            
            # 存储所有文件的统计信息
            all_stats = []
            
            # 检查每个文件
            for file_path in tqdm(csv_files[:20]):  # 只检查前20个文件以节省时间
                result = check_single_file(file_path, columns)
                if result:
                    all_stats.append(result)
            
            # 分析统计信息
            if all_stats:
                # 统计每列的缺失值总数
                missing_total = {}
                for stat in all_stats:
                    for col, count in stat['missing_values'].items():
                        missing_total[col] = missing_total.get(col, 0) + count
                
                # 统计每列的异常值总数
                outliers_total = {}
                for stat in all_stats:
                    for col, count in stat['outliers'].items():
                        outliers_total[col] = outliers_total.get(col, 0) + count
                
                print("\n数据统计摘要:")
                print(f"检查的文件数: {len(all_stats)}")
                print(f"每列缺失值总数: {missing_total}")
                print(f"每列异常值总数: {outliers_total}")
                
                # 创建可视化
                visualize_data_stats(all_stats, 'nonsentiment')
    
    # 检查情感模型数据
    if sentiment_config:
        print("\n检查情感模型数据:")
        data_path = sentiment_config.get('data_path', 'data')
        columns = sentiment_config.get('columns', [])
        
        # 获取所有CSV文件
        if os.path.isdir(data_path):
            csv_files = glob(os.path.join(data_path, '*.csv'))
        else:
            csv_files = [data_path] if data_path.endswith('.csv') else []
        
        if not csv_files:
            print(f"警告: 在 {data_path} 中未找到CSV文件")
        else:
            print(f"找到 {len(csv_files)} 个CSV文件")
            
            # 存储所有文件的统计信息
            all_stats = []
            
            # 检查每个文件
            for file_path in tqdm(csv_files[:20]):  # 只检查前20个文件以节省时间
                result = check_single_file(file_path, columns)
                if result:
                    all_stats.append(result)
            
            # 分析统计信息
            if all_stats:
                # 统计每列的缺失值总数
                missing_total = {}
                for stat in all_stats:
                    for col, count in stat['missing_values'].items():
                        missing_total[col] = missing_total.get(col, 0) + count
                
                # 统计每列的异常值总数
                outliers_total = {}
                for stat in all_stats:
                    for col, count in stat['outliers'].items():
                        outliers_total[col] = outliers_total.get(col, 0) + count
                
                print("\n数据统计摘要:")
                print(f"检查的文件数: {len(all_stats)}")
                print(f"每列缺失值总数: {missing_total}")
                print(f"每列异常值总数: {outliers_total}")
                
                # 创建可视化
                visualize_data_stats(all_stats, 'sentiment')

def visualize_data_stats(all_stats, model_type):
    """为数据统计创建可视化"""
    # 创建结果目录
    save_dir = f'data_check_results/{model_type}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 绘制每个特征的箱线图
    plt.figure(figsize=(15, 10))
    
    # 收集所有文件的所有特征值
    all_features = {}
    for stat in all_stats:
        for col in stat['stats'].columns:
            # 跳过非数值列
            if col in ['count', '25%', '50%', '75%']:
                continue
            if col not in all_features:
                all_features[col] = []
            all_features[col].append(stat['stats'][col])
    
    # 绘制箱线图
    data_to_plot = []
    labels = []
    for col, values in all_features.items():
        # 将每列的统计信息转换为一个数组
        data = np.array(values).flatten()
        if len(data) > 0:
            data_to_plot.append(data)
            labels.append(col)
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.title(f'{model_type} 模型特征统计分布')
        plt.ylabel('值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_boxplot.png'))
        plt.close()
    
    # 2. 绘制缺失值的条形图
    missing_values = {}
    for stat in all_stats:
        for col, count in stat['missing_values'].items():
            if col not in missing_values:
                missing_values[col] = []
            missing_values[col].append(count)
    
    if missing_values:
        plt.figure(figsize=(12, 6))
        for col, counts in missing_values.items():
            if sum(counts) > 0:  # 只绘制有缺失值的列
                plt.bar(col, sum(counts))
        plt.title(f'{model_type} 模型缺失值总数')
        plt.ylabel('缺失值计数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'missing_values.png'))
        plt.close()
    
    # 3. 绘制异常值的条形图
    outliers = {}
    for stat in all_stats:
        for col, count in stat['outliers'].items():
            if col not in outliers:
                outliers[col] = []
            outliers[col].append(count)
    
    if outliers:
        plt.figure(figsize=(12, 6))
        for col, counts in outliers.items():
            if sum(counts) > 0:  # 只绘制有异常值的列
                plt.bar(col, sum(counts))
        plt.title(f'{model_type} 模型异常值总数')
        plt.ylabel('异常值计数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'outliers.png'))
        plt.close()

def check_sample_data():
    """检查数据样本的分布和特征"""
    config = load_config()
    if not config:
        return
    
    # 获取数据路径和列名
    nonsentiment_config = config.get('nonsentiment_model', {})
    data_path = nonsentiment_config.get('data_path', 'data')
    columns = nonsentiment_config.get('columns', [])
    
    # 如果是目录，获取第一个CSV文件
    if os.path.isdir(data_path):
        csv_files = glob(os.path.join(data_path, '*.csv'))
        if not csv_files:
            print(f"警告: 在 {data_path} 中未找到CSV文件")
            return
        sample_file = csv_files[0]
    else:
        sample_file = data_path if data_path.endswith('.csv') else None
        if not sample_file:
            print(f"警告: 未找到有效的样本文件")
            return
    
    try:
        # 读取样本文件
        print(f"\n分析样本文件: {sample_file}")
        df = pd.read_csv(sample_file)
        
        # 检查是否包含所需列
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"警告: 文件中缺少列: {missing_cols}")
            return
        
        # 提取所需列
        df = df[columns].copy()
        
        # 保存详细信息
        save_dir = 'data_check_results/sample_analysis'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 基本信息
        with open(os.path.join(save_dir, 'basic_info.txt'), 'w') as f:
            f.write(f"样本文件: {os.path.basename(sample_file)}\n")
            f.write(f"数据形状: {df.shape}\n\n")
            f.write("数据类型:\n")
            for col in df.columns:
                f.write(f"{col}: {df[col].dtype}\n")
            f.write("\n")
            f.write("缺失值:\n")
            for col in df.columns:
                f.write(f"{col}: {df[col].isnull().sum()}\n")
            f.write("\n")
            f.write("统计信息:\n")
            f.write(df.describe().to_string())
        
        # 相关性分析
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('特征相关性')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation.png'))
        plt.close()
        
        # 时间序列图
        plt.figure(figsize=(15, 5 * len(columns)))
        for i, col in enumerate(columns):
            if np.issubdtype(df[col].dtype, np.number):
                plt.subplot(len(columns), 1, i+1)
                plt.plot(df[col].values)
                plt.title(f'{col} 时间序列')
                plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series.png'))
        plt.close()
        
        # 分布图
        plt.figure(figsize=(15, 5 * len(columns)))
        for i, col in enumerate(columns):
            if np.issubdtype(df[col].dtype, np.number):
                plt.subplot(len(columns), 1, i+1)
                sns.histplot(df[col].values, kde=True)
                plt.title(f'{col} 分布')
                plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distributions.png'))
        plt.close()
        
        print(f"样本分析完成，结果保存在 {save_dir}")
        
    except Exception as e:
        print(f"分析样本文件时出错: {e}")

if __name__ == '__main__':
    print("开始数据检查...")
    run_checks()
    check_sample_data()
    print("数据检查完成，结果保存在 data_check_results 目录") 