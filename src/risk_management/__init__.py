"""风险管理模块
包含风险度量、风险控制和资产配置等功能
"""

# 导入子模块和类
def __getattr__(name):
    if name == 'RiskManager':
        from .risk_manager import RiskManager
        return RiskManager
    elif name == 'PositionSizer':
        from .position_sizer import PositionSizer
        return PositionSizer
    elif name == 'PortfolioOptimizer':
        from .portfolio_optimizer import PortfolioOptimizer
        return PortfolioOptimizer
    elif name == 'DrawdownManager':
        from .drawdown_manager import DrawdownManager
        return DrawdownManager
    elif name == 'RiskMetrics':
        from .risk_metrics import RiskMetrics
        return RiskMetrics
    raise AttributeError(f"模块 {__name__} 没有属性 {name}")

# 导出列表
__all__ = [
    'RiskManager',
    'PositionSizer', 
    'PortfolioOptimizer',
    'DrawdownManager',
    'RiskMetrics'
]

# 模块版本
__version__ = '0.1.0'