# RAG系统开发工具

.PHONY: help install install-dev test lint format clean build docs

# 默认目标
help:
	@echo "可用的命令:"
	@echo "  install      - 安装生产依赖"
	@echo "  install-dev  - 安装开发依赖"
	@echo "  test         - 运行测试"
	@echo "  lint         - 代码检查"
	@echo "  format       - 代码格式化"
	@echo "  clean        - 清理临时文件"
	@echo "  build        - 构建包"
	@echo "  docs         - 生成文档"
	@echo "  run          - 运行系统"
	@echo "  setup        - 设置系统"

# 安装依赖
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# 测试
test:
	pytest tests/ -v --cov=rag --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x

# 代码检查
lint:
	flake8 rag/ tests/
	mypy rag/

# 代码格式化
format:
	black rag/ tests/
	isort rag/ tests/

format-check:
	black --check rag/ tests/
	isort --check-only rag/ tests/

# 清理
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# 构建
build: clean
	python -m build

# 文档
docs:
	@echo "生成文档..."
	# 这里可以添加文档生成命令

# 运行系统
run:
	python main.py

# 设置系统
setup:
	python main.py setup

# 预提交检查
pre-commit: format lint test

# 开发环境设置
dev-setup: install-dev
	pre-commit install

# 性能测试
benchmark:
	python scripts/performance_monitor.py

# 配置管理
config:
	python main.py config

# 状态检查
status:
	python main.py status 