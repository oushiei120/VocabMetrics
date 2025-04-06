#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算H-point和R1词汇多样性指标脚本
连接SQLite数据库，为每个版本-章节组合计算指标
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from math import log
import time
import spacy
import warnings

warnings.filterwarnings("ignore")

# 数据库路径
DB_PATH = "/Users/oushiei/Documents/GitHub/phd_code/002尝试分词/02spacy处理好的修正版1.sqlite"
# 输出文件夹
OUTPUT_DIR = "/Users/oushiei/Documents/GitHub/phd_code/002尝试分词/04各种特征统计"
# 输出结果文件名
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "h_point_r1_metrics.txt")
# 是否生成图表
GENERATE_PLOTS = True
# 限制章节数量(设为0则不限制)
LIMIT_CHAPTERS = 80
# 分析的版本
VERSIONS = ["伊藤1997", "井波2013"]
# 指定文件类型 (surface/lemma)
TOKEN_TYPES = ["lemma"]
# 定义内容词的词性标记
CONTENT_POS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']


class HPointR1Calculator:
    """计算文本的H-point和R1指标"""
    
    def __init__(self, wordlist):
        """初始化计算器
        
        Args:
            wordlist: 文本的词语列表
        """
        self.wordlist = wordlist
        self.words = len(wordlist)
        self.term_freq = Counter(wordlist)
        self.terms = len(self.term_freq)
    
    def h_point(self, plot=False, plot_filename=None):
        """计算H-point
        
        H-point是指满足rank=frequency的点，即词频与词频排名相等的点
        
        Args:
            plot: 是否生成图表
            plot_filename: 图表保存路径
            
        Returns:
            float: H-point值
        """
        # 按词频降序排列
        sorted_freq = sorted(self.term_freq.values(), reverse=True)
        # 创建排名列表
        ranks = list(range(1, len(sorted_freq) + 1))
        
        # 查找rank≈frequency的点
        h_point_value = None
        for i in range(len(sorted_freq) - 1):
            if sorted_freq[i] >= ranks[i] and sorted_freq[i+1] <= ranks[i+1]:
                # 线性插值找到更精确的H-point
                r_i = ranks[i]
                r_i1 = ranks[i+1]
                f_i = sorted_freq[i]
                f_i1 = sorted_freq[i+1]
                
                # 按照公式计算H-point: h = (r_i * f(r_{i+1}) - r_{i+1} * f(r_i)) / (f(r_{i+1}) - f(r_i) - r_{i+1} + r_i)
                h_point_value = (r_i * f_i1 - r_i1 * f_i) / (f_i1 - f_i - r_i1 + r_i)
                break
        
        # 如果没有找到交叉点，使用最接近的点
        if h_point_value is None:
            diff = [abs(ranks[i] - sorted_freq[i]) for i in range(len(ranks))]
            min_idx = diff.index(min(diff))
            h_point_value = (ranks[min_idx] + sorted_freq[min_idx]) / 2
        
        # 生成图表（如果需要）
        if plot:
            plt.figure(figsize=(10, 6), facecolor="white")
            plt.plot(ranks[:100], sorted_freq[:100], marker='.', linestyle='-', alpha=0.7, 
                     label='词频-排名分布')
            plt.plot(ranks[:100], ranks[:100], 'r--', alpha=0.5, label='r = f(r)')
            
            # 标记H-point
            plt.scatter([h_point_value], [h_point_value], color='red', s=100, 
                        label=f'H-point = {h_point_value:.2f}')
            
            plt.xlabel('排名 (r)', fontweight='bold')
            plt.ylabel('词频 f(r)', fontweight='bold')
            plt.title('词频-排名分布中的H-point', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            if plot_filename:
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        return h_point_value
    
    def r1_metric(self, pos_tags=None, content_pos=None):
        """计算R1指标
        
        R1是基于H-point前的词型计算的词汇多样性指标，计算公式：
        R1_content = (T_h_content / N_content) * (log(N_content) / log(h))
        
        其中:
        - T_h_content: 前h个词型中实词的tokens总数
        - N_content: 文本中实词的总tokens数
        - h: H-point值
        
        Args:
            pos_tags: 与wordlist对应的词性标签列表
            content_pos: 被视为内容词的词性列表
            
        Returns:
            dict: 包含h、R1、T_h、N等指标的字典
        """
        # 计算H-point
        h = self.h_point()
        h_int = int(h)
        
        # 获取词频
        sorted_words = sorted(self.term_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 获取前h个词型
        top_h_words = [word for word, _ in sorted_words[:h_int]]
        
        # 计算前h个词型的总token数
        T_h = sum(self.term_freq[word] for word in top_h_words)
        
        # 计算累积频率F(h)
        F_h = T_h / self.words
        
        # 如果有词性标记，计算内容词的R1
        if pos_tags is not None and content_pos is not None:
            # 获取内容词索引
            content_indices = [i for i, tag in enumerate(pos_tags) if tag in content_pos]
            
            # 获取内容词列表
            content_words = [self.wordlist[i] for i in content_indices]
            
            # 计算内容词的tokens总数
            N_content = len(content_words)
            
            # 计算内容词的词频
            content_word_freq = Counter(content_words)
            
            # 计算前h个词型中的内容词tokens
            T_h_content = sum(min(self.term_freq[word], content_word_freq.get(word, 0)) 
                             for word in top_h_words)
            
            # 计算内容词的R1
            if N_content > 0 and h > 0:
                R1_content = (T_h_content / N_content) * (log(N_content) / log(h))
            else:
                R1_content = 0
                
            return {
                'h': h,
                'F_h': F_h,
                'R1_content': R1_content,
                'T_h_content': T_h_content,
                'N_content': N_content
            }
        
        # 如果没有词性标记，计算一般R1
        N = self.words
        
        # 计算一般R1
        if N > 0 and h > 0:
            R1 = (T_h / N) * (log(N) / log(h))
        else:
            R1 = 0
        
        return {
            'h': h,
            'F_h': F_h,
            'R1': R1,
            'T_h': T_h,
            'N': N
        }

def main():
    """主函数"""
    
    print(f"开始计算H-point和R1指标...")
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 连接到SQLite数据库
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    print(f"成功连接到数据库: {DB_PATH}")
    
    # 获取可用章节列表
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT 章节1 FROM token_features")
    chapters = [str(row[0]) for row in cursor.fetchall()]  # 确保所有章节都是字符串类型
    
    # 应用章节限制
    if LIMIT_CHAPTERS > 0 and LIMIT_CHAPTERS < len(chapters):
        chapters = chapters[:LIMIT_CHAPTERS]
        print(f"已限制分析前{LIMIT_CHAPTERS}章")
    
    print(f"将分析以下章节: {', '.join(chapters)}")
    
    # 准备结果字典
    results = []
    
    # 打开输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("H-point和R1指标计算结果\n")
        f_out.write("=" * 50 + "\n\n")
        f_out.write("计算时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # 处理每个版本和章节
        for version in VERSIONS:
            f_out.write(f"\n版本: {version}\n")
            f_out.write("-" * 30 + "\n")
            
            for chapter in chapters:
                print(f"处理版本 {version} 的章节 {chapter}...")
                
                # 对每种token类型进行处理
                for token_type in TOKEN_TYPES:
                    f_out.write(f"\n章节: {chapter} - {token_type}\n")
                    
                    # 获取该版本-章节的tokens
                    query = f"""
                    SELECT {token_type}, pos 
                    FROM token_features 
                    WHERE version = ? AND 章节1 = ? 
                    ORDER BY sentence_id, token_index
                    """
                    
                    cursor.execute(query, (version, chapter))
                    rows = cursor.fetchall()
                    
                    if not rows:
                        f_out.write(f"  无数据\n")
                        continue
                    
                    # 提取tokens和词性
                    tokens = [row[0] for row in rows]
                    pos_tags = [row[1] for row in rows]
                    
                    # 计算H-point和R1
                    calculator = HPointR1Calculator(tokens)
                    
                    # 生成图表（可选）
                    plot_path = None
                    if GENERATE_PLOTS:
                        plot_dir = os.path.join(OUTPUT_DIR, "plots")
                        os.makedirs(plot_dir, exist_ok=True)
                        plot_path = os.path.join(
                            plot_dir, 
                            f"h_point_{version}_{chapter}_{token_type}.png".replace(" ", "_")
                        )
                    
                    # 计算H-point
                    h_point = calculator.h_point(plot=GENERATE_PLOTS, plot_filename=plot_path)
                    
                    # 计算R1指标
                    r1_results = calculator.r1_metric(pos_tags=pos_tags, content_pos=CONTENT_POS)
                    
                    # 将结果写入文件
                    f_out.write(f"  H-point: {h_point:.4f}\n")
                    if 'R1' in r1_results:
                        f_out.write(f"  R1: {r1_results['R1']:.4f}\n")
                    if 'R1_content' in r1_results:
                        f_out.write(f"  R1(内容词): {r1_results['R1_content']:.4f}\n")
                    if 'F_h' in r1_results:
                        f_out.write(f"  累积频率F(h): {r1_results['F_h']:.4f}\n")
                    
                    # 其他统计信息
                    f_out.write(f"  词元总数: {calculator.words}\n")
                    f_out.write(f"  词型总数: {calculator.terms}\n")
                    if 'N_content' in r1_results:
                        f_out.write(f"  内容词词元数: {r1_results['N_content']}\n")
                    if 'T_h_content' in r1_results:
                        f_out.write(f"  前h个词型中内容词词元数: {r1_results['T_h_content']}\n")
                    
                    # 保存结果
                    result_row = {
                        'version': version,
                        'chapter': chapter,
                        'token_type': token_type,
                        'h_point': h_point,
                        'tokens': calculator.words,
                        'types': calculator.terms
                    }
                    
                    # 添加R1结果
                    result_row.update(r1_results)
                    results.append(result_row)
    
    # 将结果保存为CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(OUTPUT_DIR, "h_point_r1_metrics.csv")
    results_df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # 关闭数据库连接
    conn.close()
    
    end_time = time.time()
    print(f"计算完成，结果已保存到 {OUTPUT_FILE} 和 {csv_file}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    # 可视化部分
    if GENERATE_PLOTS and len(results) > 0:
        print("\n##=========================")
        print("开始生成对比图...")
        create_comparison_plots(results_df, OUTPUT_DIR)
        print("对比图生成完成")
        print("##=========================")
    
    # 生成GPT分析用的数据文本
    print("\n##=========================")
    print("生成GPT分析用数据...")
    create_gpt_analysis_text(results_df, OUTPUT_DIR)
    print("GPT分析数据生成完成")
    print("##=========================")

def create_gpt_analysis_text(results_df, output_dir):
    """创建适合GPT分析的文本数据
    
    Args:
        results_df: 包含指标结果的DataFrame
        output_dir: 输出目录
    """
    if len(results_df) == 0:
        print("没有足够的数据生成分析文本")
        return
    
    # 创建输出文件
    output_file = os.path.join(output_dir, "gpt_analysis_data.txt")
    
    # 获取版本列表
    versions = results_df['version'].unique()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# H-point和R1指标对比分析数据\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 基本统计信息
        f.write("## 基本统计信息\n\n")
        
        for version in versions:
            version_data = results_df[results_df['version'] == version]
            f.write(f"### {version} 版本\n\n")
            
            # 计算H-point和R1的均值和标准差
            h_mean = version_data['h_point'].mean()
            h_std = version_data['h_point'].std()
            
            f.write(f"- 章节数: {len(version_data)}\n")
            f.write(f"- H-point平均值: {h_mean:.4f}\n")
            f.write(f"- H-point标准差: {h_std:.4f}\n")
            
            if 'R1_content' in version_data.columns:
                r1_mean = version_data['R1_content'].mean()
                r1_std = version_data['R1_content'].std()
                f.write(f"- R1(内容词)平均值: {r1_mean:.4f}\n")
                f.write(f"- R1(内容词)标准差: {r1_std:.4f}\n")
            
            if 'F_h' in version_data.columns:
                fh_mean = version_data['F_h'].mean()
                fh_std = version_data['F_h'].std()
                f.write(f"- 累积频率F(h)平均值: {fh_mean:.4f}\n")
                f.write(f"- 累积频率F(h)标准差: {fh_std:.4f}\n")
            
            f.write("\n")
        
        # 2. 版本间对比
        if len(versions) >= 2:
            f.write("## 版本对比\n\n")
            
            # 计算H-point差异
            version1, version2 = versions[0], versions[1]
            pivot_df = results_df.pivot(index='chapter', columns='version', values=['h_point', 'R1_content', 'F_h'])
            
            # 只保留两个版本都有数据的章节
            common_data = pivot_df.dropna()
            
            if len(common_data) > 0:
                f.write(f"### {version1} vs {version2} (共{len(common_data)}个章节)\n\n")
                
                # H-point对比
                h_diff = common_data[('h_point', version1)] - common_data[('h_point', version2)]
                h_diff_mean = h_diff.mean()
                h_diff_std = h_diff.std()
                
                f.write(f"#### H-point差异\n\n")
                f.write(f"- 平均差异: {h_diff_mean:.4f}\n")
                f.write(f"- 差异标准差: {h_diff_std:.4f}\n")
                f.write(f"- 最大差异章节: {h_diff.idxmax()} ({h_diff.max():.4f})\n")
                f.write(f"- 最小差异章节: {h_diff.idxmin()} ({h_diff.min():.4f})\n\n")
                
                # R1对比
                if ('R1_content', version1) in common_data.columns and ('R1_content', version2) in common_data.columns:
                    r1_diff = common_data[('R1_content', version1)] - common_data[('R1_content', version2)]
                    r1_diff_mean = r1_diff.mean()
                    r1_diff_std = r1_diff.std()
                    
                    f.write(f"#### R1(内容词)差异\n\n")
                    f.write(f"- 平均差异: {r1_diff_mean:.4f}\n")
                    f.write(f"- 差异标准差: {r1_diff_std:.4f}\n")
                    f.write(f"- 最大差异章节: {r1_diff.idxmax()} ({r1_diff.max():.4f})\n")
                    f.write(f"- 最小差异章节: {r1_diff.idxmin()} ({r1_diff.min():.4f})\n\n")
                
                # F(h)对比
                if ('F_h', version1) in common_data.columns and ('F_h', version2) in common_data.columns:
                    fh_diff = common_data[('F_h', version1)] - common_data[('F_h', version2)]
                    fh_diff_mean = fh_diff.mean()
                    fh_diff_std = fh_diff.std()
                    
                    f.write(f"#### 累积频率F(h)差异\n\n")
                    f.write(f"- 平均差异: {fh_diff_mean:.4f}\n")
                    f.write(f"- 差异标准差: {fh_diff_std:.4f}\n")
                    f.write(f"- 最大差异章节: {fh_diff.idxmax()} ({fh_diff.max():.4f})\n")
                    f.write(f"- 最小差异章节: {fh_diff.idxmin()} ({fh_diff.min():.4f})\n\n")
        
        # 3. 前10个差异最大的章节详细数据
        if len(versions) >= 2:
            f.write("## 差异最大的章节\n\n")
            
            # H-point差异排序
            if len(common_data) > 0:
                h_diff_abs = h_diff.abs()
                top_diff_chapters = h_diff_abs.sort_values(ascending=False).head(10).index
                
                f.write("### H-point差异最大的章节\n\n")
                f.write("| 章节 | " + f"{version1} H-point | " + f"{version2} H-point | 差异 |\n")
                f.write("|------|" + "--------------|" + "--------------|------|\n")
                
                for chapter in top_diff_chapters:
                    v1 = common_data.loc[chapter, ('h_point', version1)]
                    v2 = common_data.loc[chapter, ('h_point', version2)]
                    diff = v1 - v2
                    f.write(f"| {chapter} | {v1:.4f} | {v2:.4f} | {diff:.4f} |\n")
                
                f.write("\n")
                
                # R1差异排序
                if ('R1_content', version1) in common_data.columns and ('R1_content', version2) in common_data.columns:
                    r1_diff_abs = r1_diff.abs()
                    top_r1_diff_chapters = r1_diff_abs.sort_values(ascending=False).head(10).index
                    
                    f.write("### R1(内容词)差异最大的章节\n\n")
                    f.write("| 章节 | " + f"{version1} R1 | " + f"{version2} R1 | 差异 |\n")
                    f.write("|------|" + "----------|" + "----------|------|\n")
                    
                    for chapter in top_r1_diff_chapters:
                        v1 = common_data.loc[chapter, ('R1_content', version1)]
                        v2 = common_data.loc[chapter, ('R1_content', version2)]
                        diff = v1 - v2
                        f.write(f"| {chapter} | {v1:.4f} | {v2:.4f} | {diff:.4f} |\n")
                    
                    f.write("\n")
        
        # 4. 相关性分析
        f.write("## 指标间相关性分析\n\n")
        
        for version in versions:
            version_data = results_df[results_df['version'] == version]
            
            if 'R1_content' in version_data.columns and 'h_point' in version_data.columns:
                h_r1_corr = version_data['h_point'].corr(version_data['R1_content'])
                f.write(f"### {version}\n\n")
                f.write(f"- H-point与R1(内容词)的相关系数: {h_r1_corr:.4f}\n")
                
                if 'F_h' in version_data.columns:
                    h_fh_corr = version_data['h_point'].corr(version_data['F_h'])
                    r1_fh_corr = version_data['R1_content'].corr(version_data['F_h'])
                    f.write(f"- H-point与累积频率F(h)的相关系数: {h_fh_corr:.4f}\n")
                    f.write(f"- R1(内容词)与累积频率F(h)的相关系数: {r1_fh_corr:.4f}\n")
                
                f.write("\n")
    
    print(f"GPT分析数据已保存至 {output_file}")

def create_comparison_plots(results_df, output_dir):
    """创建版本之间的对比图
    
    Args:
        results_df: 包含指标结果的DataFrame
        output_dir: 输出目录
    """
    # 创建对比图目录
    comparison_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 检查数据框是否有足够的数据
    if len(results_df) == 0:
        print("没有足够的数据生成对比图")
        return
    
    # 获取版本列表和章节列表
    versions = results_df['version'].unique()
    chapters = results_df['chapter'].unique()
    
    if len(versions) < 2:
        print("至少需要两个版本才能生成对比图")
        return
    
    # 1. H-point对比图
    plt.figure(figsize=(12, 8), facecolor="white")
    for version in versions:
        version_data = results_df[results_df['version'] == version]
        plt.plot(version_data['chapter'], version_data['h_point'], 
                 marker='o', linestyle='-', linewidth=2, alpha=0.7, label=version)
    
    plt.title('不同版本的H-point对比', fontsize=16, fontweight='bold')
    plt.xlabel('章节', fontsize=14, fontweight='bold')
    plt.ylabel('H-point值', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'h_point_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. R1(内容词)对比图
    if 'R1_content' in results_df.columns:
        plt.figure(figsize=(12, 8), facecolor="white")
        for version in versions:
            version_data = results_df[results_df['version'] == version]
            plt.plot(version_data['chapter'], version_data['R1_content'], 
                     marker='o', linestyle='-', linewidth=2, alpha=0.7, label=version)
        
        plt.title('不同版本的R1(内容词)对比', fontsize=16, fontweight='bold')
        plt.xlabel('章节', fontsize=14, fontweight='bold')
        plt.ylabel('R1(内容词)值', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'r1_content_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 累积频率F(h)对比图
    if 'F_h' in results_df.columns:
        plt.figure(figsize=(12, 8), facecolor="white")
        for version in versions:
            version_data = results_df[results_df['version'] == version]
            plt.plot(version_data['chapter'], version_data['F_h'], 
                     marker='o', linestyle='-', linewidth=2, alpha=0.7, label=version)
        
        plt.title('不同版本的累积频率F(h)对比', fontsize=16, fontweight='bold')
        plt.xlabel('章节', fontsize=14, fontweight='bold')
        plt.ylabel('累积频率F(h)值', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'f_h_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 创建散点图比较两个版本的H-point
    if len(versions) >= 2:
        # 创建数据集
        pivot_df = results_df.pivot(index='chapter', columns='version', values='h_point')
        version1, version2 = versions[0], versions[1]
        
        # 筛选两个版本都有数据的章节
        common_chapters = pivot_df.dropna().index
        
        if len(common_chapters) > 0:
            x = pivot_df.loc[common_chapters, version1].values
            y = pivot_df.loc[common_chapters, version2].values
            
            # 创建散点图
            plt.figure(figsize=(10, 8), facecolor="white")
            plt.scatter(x, y, s=80, alpha=0.6)
            
            # 添加对角线
            min_val = min(min(x), min(y))
            max_val = max(max(x), max(y))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            # 标注章节
            for i, chapter in enumerate(common_chapters):
                plt.annotate(chapter, (x[i], y[i]), fontsize=8, alpha=0.8)
            
            plt.title(f'H-point对比: {version1} vs {version2}', fontsize=16, fontweight='bold')
            plt.xlabel(f'{version1} H-point值', fontsize=14, fontweight='bold')
            plt.ylabel(f'{version2} H-point值', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'h_point_scatter_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. 创建箱线图比较各版本的分布
    plt.figure(figsize=(10, 8), facecolor="white")
    metrics = ['h_point']
    if 'R1_content' in results_df.columns:
        metrics.append('R1_content')
    if 'F_h' in results_df.columns:
        metrics.append('F_h')
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        sns_data = []
        
        for version in versions:
            version_data = results_df[results_df['version'] == version][metric].dropna()
            if len(version_data) > 0:
                sns_data.append(version_data.values)
        
        if len(sns_data) > 0:
            plt.boxplot(sns_data, labels=versions)
            plt.title(f'{metric} 分布对比', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
    
    plt.savefig(os.path.join(comparison_dir, 'metrics_boxplot_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存至 {comparison_dir}")

if __name__ == "__main__":
    main()