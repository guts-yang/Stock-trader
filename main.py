#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票交易系统主程序入口

这个文件是股票交易系统的主入口点，提供命令行接口来运行系统的各种功能，
包括回测交易策略、风险分析、数据获取等。
"""

import argparse
import logging
import sys
import os
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.backtesting_engine.backtester import Backtester
from src.backtesting_engine.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy
from src.risk_management.risk_metrics import RiskMetrics
from src.risk_management.portfolio_optimizer import PortfolioOptimizer
from src.risk_management.drawdown_manager import DrawdownManager
from src.data_acquisition.data_fetcher import DataFetcher  # 假设有这个模块

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stocktrader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockTrader")


def load_config(config_path='config/config.yaml'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def run_backtest(args, config):
    """运行回测"""
    logger.info(f"开始回测: {args.strategy}")
    
    # 根据策略名称创建策略实例
    strategy_map = {
        'ma_cross': MovingAverageCrossStrategy,
        'rsi': RSIStrategy,
        'macd': MACDStrategy
    }
    
    if args.strategy not in strategy_map:
        logger.error(f"不支持的策略: {args.strategy}")
        return
    
    # 初始化回测器
    try:
        backtester = Backtester(
            initial_capital=args.capital or config.get('backtesting', {}).get('initial_capital', 100000),
            commission=args.commission or config.get('backtesting', {}).get('commission', 0.001),
            slippage=args.slippage or config.get('backtesting', {}).get('slippage', 0.0005)
        )
        
        # 创建策略实例
        strategy_class = strategy_map[args.strategy]
        strategy = strategy_class()
        
        # 加载数据
        if args.data_file:
            backtester.load_data(args.data_file)
        else:
            # 使用默认数据或从配置中获取
            default_data = config.get('data', {}).get('default_file', 'data/stock_data.csv')
            backtester.load_data(default_data)
        
        # 设置策略
        backtester.set_strategy(strategy)
        
        # 运行回测
        results = backtester.run()
        
        # 生成报告
        if args.report:
            backtester.generate_report(args.report)
        
        logger.info("回测完成")
        return results
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}")
        return None


def analyze_risk(args, config):
    """进行风险分析"""
    logger.info("开始风险分析")
    
    try:
        # 初始化风险指标计算器
        risk_metrics = RiskMetrics()
        
        # 加载数据
        if args.data_file:
            risk_metrics.load_data(args.data_file)
        else:
            default_data = config.get('data', {}).get('default_file', 'data/stock_data.csv')
            risk_metrics.load_data(default_data)
        
        # 计算风险指标
        metrics = risk_metrics.calculate_all_metrics()
        
        # 打印或保存结果
        if args.report:
            risk_metrics.generate_report(args.report)
        else:
            # 打印结果到控制台
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        logger.info("风险分析完成")
        return metrics
        
    except Exception as e:
        logger.error(f"风险分析过程中出错: {e}")
        return None


def optimize_portfolio(args, config):
    """优化投资组合"""
    logger.info(f"开始投资组合优化: {args.method}")
    
    try:
        # 初始化投资组合优化器
        portfolio_optimizer = PortfolioOptimizer()
        
        # 加载数据
        if args.data_file:
            portfolio_optimizer.load_data(args.data_file)
        else:
            default_data = config.get('data', {}).get('portfolio_file', 'data/portfolio_data.csv')
            portfolio_optimizer.load_data(default_data)
        
        # 执行优化
        weights = portfolio_optimizer.optimize(method=args.method)
        
        # 计算优化后的组合指标
        metrics = portfolio_optimizer.evaluate_portfolio(weights)
        
        # 打印或保存结果
        if args.report:
            portfolio_optimizer.generate_report(args.report, weights)
        else:
            print("优化后的权重:")
            for asset, weight in weights.items():
                print(f"{asset}: {weight:.4f}")
            print("\n组合指标:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        logger.info("投资组合优化完成")
        return {'weights': weights, 'metrics': metrics}
        
    except Exception as e:
        logger.error(f"投资组合优化过程中出错: {e}")
        return None


def run_example(args, config):
    """运行示例"""
    logger.info(f"运行示例: {args.example}")
    
    # 运行特定的示例
    example_path = Path(f"examples/{args.example}.py")
    
    if not example_path.exists():
        logger.error(f"示例文件不存在: {example_path}")
        return False
    
    try:
        # 导入并运行示例
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("example_module", example_path)
        example_module = importlib.util.module_from_spec(spec)
        sys.modules["example_module"] = example_module
        spec.loader.exec_module(example_module)
        
        # 如果示例模块有main函数，则调用它
        if hasattr(example_module, 'main'):
            example_module.main()
            
        logger.info("示例运行完成")
        return True
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='股票交易系统')
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='运行策略回测')
    backtest_parser.add_argument('--strategy', type=str, required=True, 
                                choices=['ma_cross', 'rsi', 'macd'], help='回测策略')
    backtest_parser.add_argument('--capital', type=float, help='初始资金')
    backtest_parser.add_argument('--commission', type=float, help='佣金率')
    backtest_parser.add_argument('--slippage', type=float, help='滑点')
    backtest_parser.add_argument('--data-file', type=str, help='数据文件路径')
    backtest_parser.add_argument('--report', type=str, help='报告输出文件路径')
    
    # 风险分析命令
    risk_parser = subparsers.add_parser('risk', help='进行风险分析')
    risk_parser.add_argument('--data-file', type=str, help='数据文件路径')
    risk_parser.add_argument('--report', type=str, help='报告输出文件路径')
    
    # 投资组合优化命令
    portfolio_parser = subparsers.add_parser('portfolio', help='优化投资组合')
    portfolio_parser.add_argument('--method', type=str, required=True, 
                                 choices=['mean_variance', 'risk_parity', 'max_sharpe'], 
                                 help='优化方法')
    portfolio_parser.add_argument('--data-file', type=str, help='数据文件路径')
    portfolio_parser.add_argument('--report', type=str, help='报告输出文件路径')
    
    # 运行示例命令
    example_parser = subparsers.add_parser('example', help='运行示例')
    example_parser.add_argument('--example', type=str, default='backtest_example', 
                               help='示例名称')
    
    # 解析参数
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 根据命令执行相应的功能
    if args.command == 'backtest':
        run_backtest(args, config)
    elif args.command == 'risk':
        analyze_risk(args, config)
    elif args.command == 'portfolio':
        optimize_portfolio(args, config)
    elif args.command == 'example':
        run_example(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()