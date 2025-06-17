#!/usr/bin/env python3
"""
性能监控脚本

监控RAG系统的性能指标，包括：
- 响应时间
- 成功率
- 资源使用情况
- 服务质量指标
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    average_search_time: float
    average_llm_time: float
    success_rate: float
    collection_size: int
    memory_usage: float
    cpu_usage: float

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config_path: str):
        self.config = RAGConfig.from_file(config_path)
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        self.metrics_history: List[PerformanceMetrics] = []
        
    def setup_pipeline(self):
        """设置RAG管道"""
            try:
            self.pipeline = RAGPipeline(self.config)
            self.logger.info("RAG管道初始化成功")
            except Exception as e:
            self.logger.error(f"RAG管道初始化失败: {str(e)}")
            raise
    
    def collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        try:
            # 获取系统统计信息
            pipeline_stats = self.pipeline.get_statistics()
            
            # 获取Milvus统计信息
            milvus_stats = self.pipeline.milvus_service.get_statistics()
            
            # 获取系统资源使用情况
            system_stats = self._get_system_stats()
            
            # 计算指标
            total_requests = pipeline_stats.get("total_requests", 0)
            successful_requests = pipeline_stats.get("successful_requests", 0)
            failed_requests = pipeline_stats.get("failed_requests", 0)
            
            success_rate = 0.0
            if total_requests > 0:
                success_rate = successful_requests / total_requests
            
            avg_response_time = pipeline_stats.get("average_response_time", 0.0)
            avg_search_time = milvus_stats.get("searches", {}).get("average_time", 0.0)
            avg_llm_time = pipeline_stats.get("average_llm_time", 0.0)
            
            collection_size = milvus_stats.get("collection_info", {}).get("row_count", 0)
            
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=avg_response_time,
                average_search_time=avg_search_time,
                average_llm_time=avg_llm_time,
                success_rate=success_rate,
                collection_size=collection_size,
                memory_usage=system_stats.get("memory_usage", 0.0),
                cpu_usage=system_stats.get("cpu_usage", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {str(e)}")
            # 返回默认指标
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0,
                average_search_time=0.0,
                average_llm_time=0.0,
                success_rate=0.0,
                collection_size=0,
                memory_usage=0.0,
                cpu_usage=0.0
            )
    
    def _get_system_stats(self) -> Dict[str, float]:
        """获取系统资源使用情况"""
        try:
            import psutil
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            return {
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage
            }
            
        except ImportError:
            self.logger.warning("psutil未安装，无法获取系统资源信息")
            return {
                "memory_usage": 0.0,
                "cpu_usage": 0.0
            }
        except Exception as e:
            self.logger.error(f"获取系统资源信息失败: {str(e)}")
            return {
                "memory_usage": 0.0,
                "cpu_usage": 0.0
            }
    
    def run_load_test(self, questions: List[str], duration: int = 60) -> List[PerformanceMetrics]:
        """运行负载测试"""
        self.logger.info(f"开始负载测试，持续 {duration} 秒")
        
        start_time = time.time()
        metrics_list = []
        
        # 重置统计信息
        self.pipeline.reset_statistics()
        
        while time.time() - start_time < duration:
            try:
                # 随机选择问题
                import random
                question = random.choice(questions)
                
                # 执行查询
                start_query = time.time()
                response = self.pipeline.run(question)
                query_time = time.time() - start_query
                
                self.logger.debug(f"查询完成: {question[:50]}... 耗时: {query_time:.3f}秒")
                
                # 收集指标
                metrics = self.collect_metrics()
                metrics_list.append(metrics)
                
                # 短暂休息
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"负载测试中的查询失败: {str(e)}")
                continue
        
        self.logger.info(f"负载测试完成，共执行 {len(metrics_list)} 次查询")
        return metrics_list
    
    def generate_report(self, metrics_list: List[PerformanceMetrics], output_file: str):
        """生成性能报告"""
        if not metrics_list:
            self.logger.warning("没有指标数据，无法生成报告")
            return
        
        try:
            # 计算统计信息
            total_requests = sum(m.total_requests for m in metrics_list)
            avg_response_time = sum(m.average_response_time for m in metrics_list) / len(metrics_list)
            avg_success_rate = sum(m.success_rate for m in metrics_list) / len(metrics_list)
            avg_memory_usage = sum(m.memory_usage for m in metrics_list) / len(metrics_list)
            avg_cpu_usage = sum(m.cpu_usage for m in metrics_list) / len(metrics_list)
            
            # 生成报告
            report = {
                "test_info": {
                    "start_time": metrics_list[0].timestamp,
                    "end_time": metrics_list[-1].timestamp,
                    "duration": len(metrics_list),
                    "total_metrics_collected": len(metrics_list)
                },
                "performance_summary": {
                    "total_requests": total_requests,
                    "average_response_time": avg_response_time,
                    "average_success_rate": avg_success_rate,
                    "average_memory_usage": avg_memory_usage,
                    "average_cpu_usage": avg_cpu_usage
                },
                "detailed_metrics": [asdict(m) for m in metrics_list]
            }
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"性能报告已保存到: {output_file}")
            
            # 打印摘要
            print("\n=== 性能测试摘要 ===")
            print(f"总请求数: {total_requests}")
            print(f"平均响应时间: {avg_response_time:.3f}秒")
            print(f"平均成功率: {avg_success_rate:.2%}")
            print(f"平均内存使用率: {avg_memory_usage:.1f}%")
            print(f"平均CPU使用率: {avg_cpu_usage:.1f}%")
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {str(e)}")
    
    def monitor_realtime(self, interval: int = 30, duration: int = 3600):
        """实时监控"""
        self.logger.info(f"开始实时监控，间隔 {interval} 秒，持续 {duration} 秒")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # 打印当前状态
                print(f"\n[{metrics.timestamp}] 系统状态:")
                print(f"  总请求数: {metrics.total_requests}")
                print(f"  成功率: {metrics.success_rate:.2%}")
                print(f"  平均响应时间: {metrics.average_response_time:.3f}秒")
                print(f"  集合大小: {metrics.collection_size}")
                print(f"  内存使用率: {metrics.memory_usage:.1f}%")
                print(f"  CPU使用率: {metrics.cpu_usage:.1f}%")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("监控被用户中断")
                break
            except Exception as e:
                self.logger.error(f"监控过程中出错: {str(e)}")
                time.sleep(interval)
        
        self.logger.info("实时监控结束")
    
    def cleanup(self):
        """清理资源"""
        if self.pipeline:
            self.pipeline.cleanup()

def main():
    parser = argparse.ArgumentParser(description="RAG系统性能监控")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--mode", choices=["monitor", "load-test"], default="monitor", help="运行模式")
    parser.add_argument("--interval", type=int, default=30, help="监控间隔（秒）")
    parser.add_argument("--duration", type=int, default=3600, help="运行时长（秒）")
    parser.add_argument("--output", default="performance_report.json", help="报告输出文件")
    parser.add_argument("--questions", nargs="+", help="负载测试问题列表")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = PerformanceMonitor(args.config)
    
    try:
        monitor.setup_pipeline()
        
        if args.mode == "monitor":
            monitor.monitor_realtime(args.interval, args.duration)
        elif args.mode == "load-test":
            if not args.questions:
                # 默认测试问题
                test_questions = [
                    "什么是RAG系统？",
                    "如何优化向量搜索性能？",
                    "Milvus的主要特点是什么？",
                    "如何选择合适的嵌入模型？",
                    "RAG系统的应用场景有哪些？"
                ]
            else:
                test_questions = args.questions
            
            metrics_list = monitor.run_load_test(test_questions, args.duration)
            monitor.generate_report(metrics_list, args.output)
    
    except Exception as e:
        logging.error(f"性能监控失败: {str(e)}")
    finally:
        monitor.cleanup()

if __name__ == "__main__":
    main() 