"""回测引擎模块
提供交易策略的历史回测功能，支持多种回测模式和性能评估指标
"""

from .backtester import Backtester
from .strategy_evaluator import StrategyEvaluator
from .performance_analyzer import PerformanceAnalyzer
from .statistical_analyzer import StatisticalAnalyzer
from .data_handler import DataHandler
from .order_manager import OrderManager
from .trade_simulator import TradeSimulator
from .strategy import Strategy
from .utils import *

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['Backtester', 'StrategyEvaluator', 'PerformanceAnalyzer', 'StatisticalAnalyzer', 'DataHandler', 'OrderManager', 'TradeSimulator', 'Strategy']

# 延迟导入机制
import sys

def __getattr__(name):
    if name == 'Backtester':
        from .backtester import Backtester
        return Backtester
    elif name == 'StrategyEvaluator':
        from .strategy_evaluator import StrategyEvaluator
        return StrategyEvaluator
    elif name == 'TransactionCost':
        from .transaction_cost import TransactionCost
        return TransactionCost
    elif name == 'SlippageModel':
        from .slippage_model import SlippageModel
        return SlippageModel
    raise AttributeError(f"模块 '{__name__}' 没有属性 '{name}'")