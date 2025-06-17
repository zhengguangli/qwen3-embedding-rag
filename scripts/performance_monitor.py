#!/usr/bin/env python3
"""
RAG 系统性能监控脚本
"""

import time
import json
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    question: str
    total_time: float
    num_candidates: int
    num_reranked: int
    success: bool
    error_message: str = ""


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_file: str = "performance_log.json"):
        self.log_file = log_file
        self.metrics: List[PerformanceMetrics] = []
        self._load_existing_metrics()
    
    def _load_existing_metrics(self):
        """加载现有的性能指标"""
        if Path(self.log_file).exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics = [PerformanceMetrics(**item) for item in data]
                logger.info(f"加载了 {len(self.metrics)} 条性能记录")
            except Exception as e:
                logger.error(f"加载性能记录失败: {str(e)}")
    
    def add_metric(self, metric: PerformanceMetrics):
        """添加性能指标"""
        self.metrics.append(metric)
        self._save_metrics()
    
    def _save_metrics(self):
        """保存性能指标"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(m) for m in self.metrics], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存性能记录失败: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.metrics:
            return {}
        
        successful_metrics = [m for m in self.metrics if m.success]
        
        if not successful_metrics:
            return {"error": "没有成功的性能记录"}
        
        stats = {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics),
            "average_total_time": statistics.mean([m.total_time for m in successful_metrics]),
            "average_candidates": statistics.mean([m.num_candidates for m in successful_metrics]),
            "average_reranked": statistics.mean([m.num_reranked for m in successful_metrics]),
            "min_total_time": min([m.total_time for m in successful_metrics]),
            "max_total_time": max([m.total_time for m in successful_metrics]),
            "median_total_time": statistics.median([m.total_time for m in successful_metrics]),
        }
        
        return stats
    
    def print_report(self):
        """打印性能报告"""
        stats = self.get_statistics()
        
        if not stats:
            print("没有性能数据")
            return
        
        print("=" * 60)
        print("RAG 系统性能报告")
        print("=" * 60)
        print(f"总请求数: {stats.get('total_requests', 0)}")
        print(f"成功请求数: {stats.get('successful_requests', 0)}")
        print(f"成功率: {stats.get('success_rate', 0):.2%}")
        print()
        print("平均响应时间:")
        print(f"  总时间: {stats.get('average_total_time', 0):.2f}s")
        print()
        print("候选文档统计:")
        print(f"  平均候选数: {stats.get('average_candidates', 0):.1f}")
        print(f"  平均重排序数: {stats.get('average_reranked', 0):.1f}")
        print()
        print("响应时间分布:")
        print(f"  最短: {stats.get('min_total_time', 0):.2f}s")
        print(f"  最长: {stats.get('max_total_time', 0):.2f}s")
        print(f"  中位数: {stats.get('median_total_time', 0):.2f}s")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="性能监控工具")
    parser.add_argument("--report", action="store_true", help="生成性能报告")
    parser.add_argument("--clear", action="store_true", help="清除性能记录")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.clear:
        monitor.metrics = []
        monitor._save_metrics()
        print("性能记录已清除")
    
    if args.report:
        monitor.print_report()


if __name__ == "__main__":
    main() 