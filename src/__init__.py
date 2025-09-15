"""PyQuantTrader 量化交易系统主模块"""

__version__ = "0.1.0"
__author__ = "PyQuantTrader Team"
__description__ = "基于深度学习的多因子量化交易系统"

# 导出主要模块
from .data_acquisition import *
from .feature_engineering import *
from .prediction_model import *
from .backtesting_engine import *
from .trading_execution import *
from .risk_management import *
from .visualization import *