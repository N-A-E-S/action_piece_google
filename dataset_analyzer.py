#!/usr/bin/env python3
"""
数据集形状和内容分析脚本
用于分析Beauty和Sports数据集的详细统计信息
"""

import json
import os
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast

def parse_gz(path):
    """解析压缩文件"""
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as g:
        for line_num, line in enumerate(g, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError) as e:
                    if line_num <= 5:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

class DatasetAnalyzer:
    def __init__(self, category, cache_dir="cache"):
        self.category = category
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / "AmazonReviews2014" / category / "processed"
        
    def load_processed_data(self):
        """加载已处理的数据"""
        seq_file = self.data_dir / 'all_item_seqs.json'
        id_mapping_file = self.data_dir / 'id_mapping.json'
        
        if seq_file.exists() and id_mapping_file.exists():
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping
        else:
            print(f"Processed data not found for {self.category}")
            return None, None
    
    def load_raw_data(self):
        """加载原始数据"""
        reviews_file = self.data_dir / f'reviews_{self.category}_5.json.gz'
        metadata_file = self.data_dir / f'meta_{self.category}.json.gz'
        
        reviews = []
        metadata = {}
        
        if reviews_file.exists():
            print(f"Loading reviews from {reviews_file}")
            for review in parse_gz(reviews_file):
                reviews.append(review)
        
        if metadata_file.exists():
            print(f"Loading metadata from {metadata_file}")
            for item in parse_gz(metadata_file):
                if 'asin' in item:
                    metadata[item['asin']] = item
                    
        return reviews, metadata
    
    def basic_statistics(self, all_item_seqs, id_mapping):
        """计算基本统计信息"""
        n_users = len(all_item_seqs)
        n_items = len(id_mapping['item2id'])
        
        # 序列长度统计
        seq_lengths = [len(seq) for seq in all_item_seqs.values()]
        total_interactions = sum(seq_lengths)
        
        stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': total_interactions,
            'avg_seq_length': np.mean(seq_lengths),
            'median_seq_length': np.median(seq_lengths),
            'min_seq_length': np.min(seq_lengths),
            'max_seq_length': np.max(seq_lengths),
            'std_seq_length': np.std(seq_lengths),
            'sparsity': 1 - (total_interactions / (n_users * n_items))
        }
        
        return stats, seq_lengths
    
    def item_popularity_analysis(self, all_item_seqs):
        """分析物品流行度分布"""
        item_counts = Counter()
        for seq in all_item_seqs.values():
            item_counts.update(seq)
        
        popularity_stats = {
            'unique_items_in_sequences': len(item_counts),
            'most_popular_item_count': max(item_counts.values()) if item_counts else 0,
            'least_popular_item_count': min(item_counts.values()) if item_counts else 0,
            'avg_item_popularity': np.mean(list(item_counts.values())) if item_counts else 0
        }
        
        # 流行度分布
        popularity_distribution = Counter(item_counts.values())
        
        return popularity_stats, item_counts, popularity_distribution
    
    def feature_analysis(self, metadata, id_mapping):
        """分析物品特征"""
        if not metadata:
            return None
            
        # 只分析出现在序列中的物品
        items_in_seqs = set(id_mapping['item2id'].keys())
        relevant_metadata = {k: v for k, v in metadata.items() if k in items_in_seqs}
        
        feature_stats = {}
        
        # 分析各个特征的覆盖率和分布
        features_to_analyze = ['title', 'price', 'brand', 'categories', 'description', 'feature']
        
        for feature in features_to_analyze:
            values = []
            missing_count = 0
            
            for item_data in relevant_metadata.values():
                if feature in item_data and item_data[feature]:
                    if feature == 'categories':
                        # 类别数量
                        if isinstance(item_data[feature], list):
                            values.append(len(item_data[feature]))
                        else:
                            values.append(1)
                    elif feature == 'feature':
                        # 特征数量
                        if isinstance(item_data[feature], list):
                            values.append(len(item_data[feature]))
                        else:
                            values.append(1)
                    elif feature == 'title' or feature == 'description':
                        # 文本长度
                        if isinstance(item_data[feature], str):
                            values.append(len(item_data[feature].split()))
                    elif feature == 'price':
                        # 价格值
                        try:
                            if isinstance(item_data[feature], (int, float)):
                                values.append(float(item_data[feature]))
                            elif isinstance(item_data[feature], str):
                                # 尝试从字符串中提取价格
                                price_str = item_data[feature].replace('$', '').replace(',', '')
                                values.append(float(price_str))
                        except:
                            missing_count += 1
                    else:
                        values.append(1)  # 其他特征存在即计1
                else:
                    missing_count += 1
            
            if values:
                feature_stats[feature] = {
                    'coverage': (len(relevant_metadata) - missing_count) / len(relevant_metadata),
                    'mean': np.mean(values) if values else 0,
                    'median': np.median(values) if values else 0,
                    'std': np.std(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0
                }
            else:
                feature_stats[feature] = {
                    'coverage': 0,
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0
                }
        
        return feature_stats
    
    def temporal_analysis(self, reviews):
        """时间分析"""
        if not reviews:
            return None
            
        timestamps = [review.get('unixReviewTime', 0) for review in reviews if 'unixReviewTime' in review]
        
        if not timestamps:
            return None
            
        temporal_stats = {
            'time_span_days': (max(timestamps) - min(timestamps)) / (24 * 3600),
            'earliest_review': min(timestamps),
            'latest_review': max(timestamps),
            'total_reviews': len(timestamps)
        }
        
        return temporal_stats
    
    def create_visualizations(self, seq_lengths, item_counts, category):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{category} Dataset Analysis', fontsize=16)
        
        # 1. 序列长度分布
        axes[0, 0].hist(seq_lengths, bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(seq_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(seq_lengths):.2f}')
        axes[0, 0].legend()
        
        # 2. 物品流行度分布 (log scale)
        popularity_values = list(item_counts.values())
        axes[0, 1].hist(popularity_values, bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Item Popularity Distribution')
        axes[0, 1].set_xlabel('Item Frequency')
        axes[0, 1].set_ylabel('Number of Items')
        axes[0, 1].set_yscale('log')
        
        # 3. 序列长度累积分布
        sorted_lengths = np.sort(seq_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        axes[1, 0].plot(sorted_lengths, cumulative, color='orange')
        axes[1, 0].set_title('Cumulative Distribution of Sequence Lengths')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top 20 最流行物品
        top_items = item_counts.most_common(20)
        if top_items:
            items, counts = zip(*top_items)
            axes[1, 1].bar(range(len(counts)), counts, color='coral')
            axes[1, 1].set_title('Top 20 Most Popular Items')
            axes[1, 1].set_xlabel('Item Rank')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_xticks(range(0, len(counts), 2))
        
        plt.tight_layout()
        plt.savefig(f'{category}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_datasets(self, other_analyzer):
        """比较两个数据集"""
        print("=" * 60)
        print("DATASET COMPARISON")
        print("=" * 60)
        
        # 加载两个数据集的数据
        data1 = self.load_processed_data()
        data2 = other_analyzer.load_processed_data()
        
        if data1[0] is None or data2[0] is None:
            print("Cannot compare: missing processed data")
            return
        
        # 计算统计信息
        stats1, lengths1 = self.basic_statistics(*data1)
        stats2, lengths2 = other_analyzer.basic_statistics(*data2)
        
        # 打印比较结果
        print(f"\n{'Metric':<25} {'Beauty':<15} {'Sports':<15} {'Ratio (S/B)':<15}")
        print("-" * 70)
        
        metrics = ['n_users', 'n_items', 'n_interactions', 'avg_seq_length', 'sparsity']
        
        for metric in metrics:
            val1 = stats1[metric] if self.category == 'Beauty' else stats2[metric]
            val2 = stats2[metric] if self.category == 'Beauty' else stats1[metric]
            ratio = val2 / val1 if val1 != 0 else float('inf')
            
            if metric == 'sparsity':
                print(f"{metric:<25} {val1:<15.4f} {val2:<15.4f} {ratio:<15.4f}")
            elif metric == 'avg_seq_length':
                print(f"{metric:<25} {val1:<15.2f} {val2:<15.2f} {ratio:<15.4f}")
            else:
                print(f"{metric:<25} {val1:<15} {val2:<15} {ratio:<15.4f}")
    
    def full_analysis(self):
        """完整分析流程"""
        print(f"Analyzing {self.category} dataset...")
        print("=" * 50)
        
        # 1. 加载处理后的数据
        all_item_seqs, id_mapping = self.load_processed_data()
        if all_item_seqs is None:
            print("No processed data found. Please run training first.")
            return
        
        # 2. 基本统计
        stats, seq_lengths = self.basic_statistics(all_item_seqs, id_mapping)
        print("\nBASIC STATISTICS:")
        print("-" * 30)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # 3. 物品流行度分析
        pop_stats, item_counts, pop_dist = self.item_popularity_analysis(all_item_seqs)
        print("\nITEM POPULARITY:")
        print("-" * 30)
        for key, value in pop_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # 4. 加载原始数据进行特征分析
        reviews, metadata = self.load_raw_data()
        
        # 5. 特征分析
        if metadata:
            feature_stats = self.feature_analysis(metadata, id_mapping)
            print("\nFEATURE COVERAGE:")
            print("-" * 30)
            for feature, stats in feature_stats.items():
                print(f"{feature}: {stats['coverage']:.4f} coverage, mean: {stats['mean']:.2f}")
        
        # 6. 时间分析
        if reviews:
            temporal_stats = self.temporal_analysis(reviews)
            if temporal_stats:
                print("\nTEMPORAL STATISTICS:")
                print("-" * 30)
                print(f"Time span: {temporal_stats['time_span_days']:.1f} days")
                print(f"Total reviews: {temporal_stats['total_reviews']}")
        
        # 7. 创建可视化
        self.create_visualizations(seq_lengths, item_counts, self.category)
        
        return {
            'basic_stats': stats,
            'popularity_stats': pop_stats,
            'feature_stats': feature_stats if metadata else None,
            'temporal_stats': temporal_stats if reviews else None
        }

# 使用示例
if __name__ == "__main__":
    # 分析Beauty数据集
    beauty_analyzer = DatasetAnalyzer("Beauty")
    beauty_results = beauty_analyzer.full_analysis()
    
    print("\n" + "="*80 + "\n")
    
    # 分析Sports数据集
    sports_analyzer = DatasetAnalyzer("Sports_and_Outdoors")
    sports_results = sports_analyzer.full_analysis()
    
    print("\n" + "="*80 + "\n")
    
    # 比较两个数据集
    beauty_analyzer.compare_datasets(sports_analyzer)