#!/usr/bin/env python3
"""
RAG系统命令行接口
使用Click库提供现代化的CLI体验
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
import rich.console
import rich.table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax

from src.rag.config import RAGConfig
from src.rag.exceptions import RAGException, handle_exception
from .pipeline import RAGPipeline
from .utils import setup_logging, check_dependencies

# 设置Rich控制台
console = rich.console.Console()

def print_banner():
    """打印系统横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Qwen3 Embedding RAG 系统                    ║
    ║                基于 Milvus 的检索增强生成系统                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))

@click.group()
@click.version_option(version="1.0.0", prog_name="RAG系统")
@click.option("--config", "-c", type=click.Path(exists=True), help="配置文件路径")
@click.option("--env", default=None, help="指定环境（如dev/prod/test），优先于RAG_ENV环境变量")
@click.option("--log-level", default="INFO", help="日志级别")
@click.pass_context
def cli(ctx, config: Optional[str], env: Optional[str], log_level: str):
    """Qwen3 Embedding RAG 系统 - 基于 Milvus 的检索增强生成系统"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["env"] = env
    ctx.obj["log_level"] = log_level
    
    # 设置日志
    logger = setup_logging(log_level)
    ctx.obj["logger"] = logger

@cli.command()
@click.option("--question", "-q", help="要回答的问题")
@click.option("--output-file", "-o", default="answer.txt", help="答案输出文件路径")
@click.option("--force-recreate", "-f", is_flag=True, help="强制重建 Milvus 集合")
@click.pass_context
def ask(ctx, question: Optional[str], output_file: str, force_recreate: bool):
    """提问并获取答案"""
    print_banner()
    
    try:
        # 检查依赖
        check_dependencies()
        
        # 加载配置
        config = RAGConfig(ctx.obj["config"], env=ctx.obj["env"])
        ctx.obj["logger"].info("配置加载成功")
        
        # 显示配置摘要
        show_config_summary(config)
        
        # 初始化管道
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("初始化RAG管道...", total=None)
            pipeline = RAGPipeline(config)
            progress.update(task, description="RAG管道初始化完成")
        
        # 强制重建集合
        if force_recreate:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("重建Milvus集合...", total=None)
                pipeline.setup_collection(force_recreate=True)
                progress.update(task, description="Milvus集合重建完成")
        
        # 处理问题
        if question:
            process_single_question(pipeline, question, output_file, ctx.obj["logger"])
        else:
            interactive_mode(pipeline, ctx.obj["logger"])
            
    except Exception as e:
        console.print(f"[red]错误: {str(e)}[/red]")
        ctx.obj["logger"].error(f"程序运行出错: {str(e)}")
        sys.exit(1)

@cli.command()
@click.pass_context
def setup(ctx):
    """设置和初始化系统"""
    print_banner()
    
    try:
        # 检查依赖
        check_dependencies()
        
        # 加载配置
        config = RAGConfig(ctx.obj["config"], env=ctx.obj["env"])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("初始化系统...", total=None)
            
            # 初始化管道
            pipeline = RAGPipeline(config)
            progress.update(task, description="RAG管道初始化完成")
            
            # 设置集合
            task = progress.add_task("设置Milvus集合...", total=None)
            pipeline.setup_collection(force_recreate=False)
            progress.update(task, description="Milvus集合设置完成")
        
        console.print("[green]系统设置完成！[/green]")
        
    except Exception as e:
        console.print(f"[red]设置失败: {str(e)}[/red]")
        ctx.obj["logger"].error(f"系统设置失败: {str(e)}")
        sys.exit(1)

@cli.command()
@click.pass_context
def config(ctx):
    """显示和验证配置"""
    print_banner()
    
    try:
        config = RAGConfig(ctx.obj["config"], env=ctx.obj["env"])
        show_detailed_config(config)
        
    except Exception as e:
        console.print(f"[red]配置错误: {str(e)}[/red]")
        ctx.obj["logger"].error(f"配置验证失败: {str(e)}")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """检查系统状态"""
    print_banner()
    
    try:
        config = RAGConfig(ctx.obj["config"], env=ctx.obj["env"])
        
        # 检查各个组件状态
        status_table = rich.table.Table(title="系统状态")
        status_table.add_column("组件", style="cyan")
        status_table.add_column("状态", style="green")
        status_table.add_column("详情", style="white")
        
        # 检查配置
        try:
            config.validate()
            status_table.add_row("配置", "✅ 正常", "配置验证通过")
        except Exception as e:
            status_table.add_row("配置", "❌ 错误", str(e))
        
        # 检查Milvus连接
        try:
            from .milvus_service import MilvusService
            milvus_service = MilvusService(config)
            milvus_service.check_connection()
            status_table.add_row("Milvus", "✅ 正常", "连接成功")
        except Exception as e:
            status_table.add_row("Milvus", "❌ 错误", str(e))
        
        # 检查API连接
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config.api.openai_api_key,
                base_url=config.api.openai_base_url
            )
            # 简单测试连接
            client.models.list()
            status_table.add_row("API", "✅ 正常", "连接成功")
        except Exception as e:
            status_table.add_row("API", "❌ 错误", str(e))
        
        console.print(status_table)
        
    except Exception as e:
        console.print(f"[red]状态检查失败: {str(e)}[/red]")
        ctx.obj["logger"].error(f"状态检查失败: {str(e)}")
        sys.exit(1)

def show_config_summary(config: RAGConfig):
    """显示配置摘要"""
    summary_table = rich.table.Table(title="配置摘要")
    summary_table.add_column("配置项", style="cyan")
    summary_table.add_column("值", style="white")
    
    summary_table.add_row("API基础URL", config.api.openai_base_url)
    summary_table.add_row("Milvus URI", config.database.milvus_uri)
    summary_table.add_row("集合名称", config.database.collection_name)
    summary_table.add_row("嵌入模型", config.models.embedding.name)
    summary_table.add_row("LLM模型", config.models.llm.name)
    summary_table.add_row("数据路径", config.data.data_path_glob)
    
    console.print(summary_table)

def show_detailed_config(config: RAGConfig):
    """显示详细配置"""
    config_dict = config.to_dict()
    config_json = json.dumps(config_dict, indent=2, ensure_ascii=False)
    
    syntax = Syntax(config_json, "json", theme="monokai")
    console.print(Panel(syntax, title="详细配置", border_style="blue"))

def process_single_question(pipeline: RAGPipeline, question: str, output_file: str, logger):
    """处理单个问题"""
    console.print(f"\n[bold cyan]问题:[/bold cyan] {question}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("正在生成答案...", total=None)
        answer = pipeline.run(question)
        progress.update(task, description="答案生成完成")
    
    # 保存答案
    output_path = Path("answers") / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"问题: {question}\n")
        f.write(f"答案: {answer}\n")
    
    # 显示结果
    console.print(f"\n[bold green]答案已保存到:[/bold green] {output_path}")
    console.print(f"[bold green]答案长度:[/bold green] {len(answer)} 字符")
    
    # 显示答案摘要
    answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
    console.print(f"\n[bold yellow]答案摘要:[/bold yellow]")
    console.print(Panel(answer_preview, border_style="yellow"))

def interactive_mode(pipeline: RAGPipeline, logger):
    """交互模式"""
    console.print("\n[bold green]进入交互模式[/bold green]")
    console.print("输入问题开始对话，输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            question = Prompt.ask("[bold cyan]请输入问题[/bold cyan]")
            
            if question.lower() in ['quit', 'exit', '退出']:
                console.print("[yellow]再见！[/yellow]")
                break
            
            if not question.strip():
                continue
            
            process_single_question(pipeline, question, f"answer_{len(question)}.txt", logger)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]退出程序...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]处理问题时出错: {str(e)}[/red]")
            logger.error(f"处理问题时出错: {str(e)}")

def main():
    """主入口函数"""
    cli()

# 移除重复的入口点，只保留main函数供外部调用 