import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入回测引擎组件
from src.backtesting_engine.backtester import Backtester
from src.backtesting_engine.strategy import MovingAverageCrossStrategy, RSIStrategy, MACDStrategy
from src.backtesting_engine.data_handler import CSVDataHandler
from src.backtesting_engine.order_manager import Order, OrderManager, PositionManager
from src.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from src.backtesting_engine.statistical_analyzer import StatisticalAnalyzer
from src.backtesting_engine.utils import BacktestUtils

# 导入风险管理组件
from src.risk_management.risk_manager import RiskManager
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.portfolio_optimizer import PortfolioOptimizer
from src.risk_management.drawdown_manager import DrawdownManager
from src.risk_management.risk_metrics import RiskMetrics

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class BacktestExample:
    """回测引擎示例类
    展示如何使用回测引擎的各个组件进行完整的交易策略回测流程
    """
    def __init__(self):
        """初始化回测示例"""
        # 设置数据目录
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        # 设置结果目录
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        # 确保结果目录存在
        BacktestUtils.create_directory_if_not_exists(self.results_dir)
        
        # 初始化回测引擎组件
        self.backtester = None
        self.strategy = None
        self.data_handler = None
        self.risk_manager = None
        self.position_sizer = None
        self.performance_analyzer = None
        self.statistical_analyzer = None
        
        # 设置日志
        self._setup_logger()
        
        logger.info("回测示例初始化完成")
    
    def _setup_logger(self):
        """设置日志记录器"""
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        BacktestUtils.create_directory_if_not_exists(log_dir)
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f"backtest_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def prepare_data(self, symbol: str = 'AAPL', start_date: str = '2010-01-01', end_date: str = '2023-12-31'):
        """准备回测数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info(f"准备回测数据: {symbol}, 日期范围: {start_date} 至 {end_date}")
        
        # 创建数据处理器
        self.data_handler = CSVDataHandler(data_dir=self.data_dir)
        
        try:
            # 尝试加载数据
            data = self.data_handler.load_data(symbol=symbol)
            
            # 如果数据不存在或不完整，生成模拟数据
            if data is None or data.empty:
                logger.warning(f"未找到 {symbol} 的数据，生成模拟数据")
                data = self._generate_sample_data(symbol=symbol, start_date=start_date, end_date=end_date)
                
                # 保存模拟数据
                data_file = os.path.join(self.data_dir, f'{symbol}.csv')
                data.to_csv(data_file)
                logger.info(f"模拟数据已保存到: {data_file}")
            else:
                # 筛选日期范围
                data = data.loc[start_date:end_date]
                
            logger.info(f"数据准备完成，共有 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"准备数据时发生异常: {str(e)}")
            # 生成模拟数据作为备选
            logger.info("生成模拟数据作为备选")
            return self._generate_sample_data(symbol=symbol, start_date=start_date, end_date=end_date)
    
    def _generate_sample_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            模拟的OHLCV数据
        """
        # 创建日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(date_range)
        
        # 生成随机价格数据（使用几何布朗运动）
        returns = np.random.normal(0, 0.01, n-1)
        price = 100.0  # 初始价格
        prices = [price]
        
        for r in returns:
            price *= (1 + r)
            prices.append(price)
        
        # 生成OHLC数据
        open_prices = prices[:-1]
        close_prices = prices[1:]
        high_prices = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(open_prices, close_prices)]
        low_prices = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(open_prices, close_prices)]
        
        # 生成成交量数据
        volumes = np.random.randint(1000000, 10000000, n-1)
        
        # 创建数据框
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=date_range[1:])  # 注意索引从第二个日期开始
        
        return data
    
    def setup_strategy(self, strategy_type: str = 'ma_cross', params: Optional[Dict[str, Any]] = None):
        """设置交易策略
        
        Args:
            strategy_type: 策略类型 ('ma_cross', 'rsi', 'macd')
            params: 策略参数
        """
        logger.info(f"设置交易策略: {strategy_type}")
        
        # 默认参数
        if params is None:
            if strategy_type == 'ma_cross':
                params = {'short_window': 50, 'long_window': 200}
            elif strategy_type == 'rsi':
                params = {'rsi_period': 14, 'overbought_threshold': 70, 'oversold_threshold': 30}
            elif strategy_type == 'macd':
                params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            else:
                params = {}
        
        # 创建策略实例
        if strategy_type == 'ma_cross':
            self.strategy = MovingAverageCrossStrategy(**params)
        elif strategy_type == 'rsi':
            self.strategy = RSIStrategy(**params)
        elif strategy_type == 'macd':
            self.strategy = MACDStrategy(**params)
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")
        
        logger.info(f"策略设置完成，参数: {params}")
        return self.strategy
    
    def setup_backtester(self, 
                        data: pd.DataFrame,
                        initial_capital: float = 100000.0,
                        commission: float = 0.001,
                        slippage: float = 0.0005):
        """设置回测器
        
        Args:
            data: 回测数据
            initial_capital: 初始资金
            commission: 佣金率
            slippage: 滑点
        """
        logger.info(f"设置回测器: 初始资金={initial_capital}, 佣金率={commission}, 滑点={slippage}")
        
        # 创建回测器配置
        config = {
            'initial_capital': initial_capital,
            'commission': commission,
            'slippage': slippage,
            'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        }
        
        # 创建回测器
        self.backtester = Backtester(config=config)
        
        # 设置数据
        self.backtester.set_data(data)
        
        # 设置策略
        if self.strategy is not None:
            self.backtester.set_strategy(self.strategy)
        
        # 创建并设置风险管理器
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        self.backtester.set_risk_manager(self.risk_manager)
        
        # 创建并设置头寸计算器
        self.position_sizer = PositionSizer(strategy='fixed_percent', risk_per_trade=0.02)
        self.backtester.set_position_sizer(self.position_sizer)
        
        logger.info("回测器设置完成")
        return self.backtester
    
    def run_backtest(self):
        """运行回测
        
        Returns:
            回测结果
        """
        if self.backtester is None:
            logger.error("回测器尚未设置，无法运行回测")
            return None
        
        logger.info("开始回测")
        
        # 运行回测
        self.backtester.run()
        
        logger.info("回测完成")
        
        # 获取回测结果
        results = self.backtester.get_results()
        
        # 保存回测结果
        results_file = os.path.join(self.results_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        BacktestUtils.save_results(results, results_file)
        logger.info(f"回测结果已保存到: {results_file}")
        
        return results
    
    def analyze_performance(self):
        """分析回测绩效
        
        Returns:
            绩效分析结果
        """
        if self.backtester is None:
            logger.error("回测器尚未设置，无法分析绩效")
            return None
        
        logger.info("开始绩效分析")
        
        # 创建绩效分析器
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 获取回测器的收益率和交易记录
        returns = self.backtester.returns
        trades = self.backtester.trades
        
        # 设置数据
        self.performance_analyzer.set_data(returns=returns, trades=trades)
        
        # 计算绩效指标
        metrics = self.performance_analyzer.calculate_performance_metrics()
        
        # 生成绩效报告
        report_file = os.path.join(self.results_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        self.performance_analyzer.generate_performance_report(save_path=report_file)
        logger.info(f"绩效报告已保存到: {report_file}")
        
        # 生成可视化图表
        self._generate_performance_charts()
        
        logger.info("绩效分析完成")
        
        return metrics
    
    def _generate_performance_charts(self):
        """生成绩效可视化图表"""
        if self.performance_analyzer is None:
            logger.error("绩效分析器尚未设置，无法生成图表")
            return
        
        # 确保图表目录存在
        charts_dir = os.path.join(self.results_dir, 'charts')
        BacktestUtils.create_directory_if_not_exists(charts_dir)
        
        # 生成权益曲线图
        equity_chart_file = os.path.join(charts_dir, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        self.performance_analyzer.plot_equity_curve(save_path=equity_chart_file)
        
        # 生成回撤曲线图
        drawdown_chart_file = os.path.join(charts_dir, f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        self.performance_analyzer.plot_drawdown(save_path=drawdown_chart_file)
        
        # 生成月度收益热力图
        monthly_heatmap_file = os.path.join(charts_dir, f"monthly_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        self.performance_analyzer.plot_monthly_returns_heatmap(save_path=monthly_heatmap_file)
        
        # 生成收益分布图
        distribution_file = os.path.join(charts_dir, f"return_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        self.performance_analyzer.plot_return_distribution(save_path=distribution_file)
        
        logger.info("绩效可视化图表生成完成")
    
    def run_statistical_analysis(self):
        """运行统计分析
        
        Returns:
            统计分析结果
        """
        if self.backtester is None:
            logger.error("回测器尚未设置，无法运行统计分析")
            return None
        
        logger.info("开始统计分析")
        
        # 创建统计分析器
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # 获取回测器的收益率和交易记录
        returns = self.backtester.returns
        trades = self.backtester.trades
        
        # 设置数据
        self.statistical_analyzer.set_data(returns=returns, trades=trades)
        
        # 进行蒙特卡洛模拟
        monte_carlo_results = self.statistical_analyzer.monte_carlo_simulation(
            num_simulations=1000,
            time_horizon=252
        )
        
        # 绘制蒙特卡洛模拟结果
        mc_chart_file = os.path.join(self.results_dir, f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        self.statistical_analyzer.plot_monte_carlo_results(monte_carlo_results, save_path=mc_chart_file)
        
        # 计算概率指标
        probabilistic_metrics = self.statistical_analyzer.calculate_probabilistic_metrics()
        
        # 运行假设检验
        hypothesis_results = self.statistical_analyzer.run_hypothesis_test(hypothesis='normality')
        
        # 保存统计分析结果
        stats_results = {
            'monte_carlo': monte_carlo_results,
            'probabilistic_metrics': probabilistic_metrics,
            'hypothesis_test': hypothesis_results
        }
        
        stats_file = os.path.join(self.results_dir, f"statistical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        BacktestUtils.save_results(stats_results, stats_file)
        logger.info(f"统计分析结果已保存到: {stats_file}")
        
        logger.info("统计分析完成")
        
        return stats_results
    
    def run_risk_analysis(self):
        """运行风险分析
        
        Returns:
            风险分析结果
        """
        if self.backtester is None:
            logger.error("回测器尚未设置，无法运行风险分析")
            return None
        
        logger.info("开始风险分析")
        
        # 创建风险指标计算器
        risk_metrics = RiskMetrics()
        
        # 获取回测器的收益率
        returns = self.backtester.returns
        
        # 设置数据
        risk_metrics.set_data(returns=returns)
        
        # 计算各类风险指标
        all_risk_metrics = {
            'volatility': risk_metrics.calculate_volatility(),
            'var': risk_metrics.calculate_var(),
            'cvar': risk_metrics.calculate_cvar(),
            'sharpe_ratio': risk_metrics.calculate_sharpe_ratio(),
            'sortino_ratio': risk_metrics.calculate_sortino_ratio(),
            'drawdown_stats': risk_metrics.calculate_drawdown_stats(),
            'risk_contribution': risk_metrics.calculate_risk_contribution(),
            'tail_risk': risk_metrics.calculate_tail_risk()
        }
        
        # 生成风险报告
        risk_report_file = os.path.join(self.results_dir, f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        risk_metrics.generate_risk_report(save_path=risk_report_file)
        logger.info(f"风险报告已保存到: {risk_report_file}")
        
        # 保存风险分析结果
        risk_results_file = os.path.join(self.results_dir, f"risk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        BacktestUtils.save_results(all_risk_metrics, risk_results_file)
        logger.info(f"风险分析结果已保存到: {risk_results_file}")
        
        logger.info("风险分析完成")
        
        return all_risk_metrics
    
    def run_full_analysis(self, 
                         symbol: str = 'AAPL',
                         strategy_type: str = 'ma_cross',
                         initial_capital: float = 100000.0):
        """运行完整的回测和分析流程
        
        Args:
            symbol: 股票代码
            strategy_type: 策略类型
            initial_capital: 初始资金
        
        Returns:
            完整分析结果
        """
        logger.info(f"运行完整分析: 标的={symbol}, 策略={strategy_type}, 初始资金={initial_capital}")
        
        try:
            # 1. 准备数据
            data = self.prepare_data(symbol=symbol)
            
            # 2. 设置策略
            self.setup_strategy(strategy_type=strategy_type)
            
            # 3. 设置回测器
            self.setup_backtester(data=data, initial_capital=initial_capital)
            
            # 4. 运行回测
            backtest_results = self.run_backtest()
            
            if backtest_results is None:
                logger.error("回测失败，无法继续分析")
                return None
            
            # 5. 分析绩效
            performance_results = self.analyze_performance()
            
            # 6. 运行统计分析
            statistical_results = self.run_statistical_analysis()
            
            # 7. 运行风险分析
            risk_results = self.run_risk_analysis()
            
            # 8. 整合所有结果
            full_results = {
                'backtest': backtest_results,
                'performance': performance_results,
                'statistical': statistical_results,
                'risk': risk_results,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'symbol': symbol,
                    'strategy_type': strategy_type,
                    'initial_capital': initial_capital
                }
            }
            
            # 保存完整结果
            full_results_file = os.path.join(self.results_dir, f"full_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            BacktestUtils.save_results(full_results, full_results_file)
            logger.info(f"完整分析结果已保存到: {full_results_file}")
            
            # 打印主要结果摘要
            self._print_results_summary(performance_results)
            
            logger.info("完整分析流程完成")
            return full_results
        except Exception as e:
            logger.error(f"运行完整分析时发生异常: {str(e)}")
            return None
    
    def _print_results_summary(self, performance_results: Dict[str, float]):
        """打印结果摘要
        
        Args:
            performance_results: 绩效分析结果
        """
        if not performance_results:
            return
        
        print("\n===== 回测结果摘要 =====")
        print(f"总收益率: {performance_results.get('total_return', 0) * 100:.2f}%")
        print(f"年化收益率: {performance_results.get('annualized_return', 0) * 100:.2f}%")
        print(f"最大回撤: {performance_results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"夏普比率: {performance_results.get('sharpe_ratio', 0):.2f}")
        print(f"索提诺比率: {performance_results.get('sortino_ratio', 0):.2f}")
        print(f"胜率: {performance_results.get('win_rate', 0) * 100:.2f}%")
        print(f"盈利因子: {performance_results.get('profit_factor', 0):.2f}")
        print(f"交易次数: {performance_results.get('total_trades', 0)}")
        print(f"平均盈亏比: {performance_results.get('avg_win_loss_ratio', 0):.2f}")
        print("========================\n")

if __name__ == "__main__":
    # 创建回测示例实例
    example = BacktestExample()
    
    # 运行完整分析
    try:
        # 您可以根据需要修改参数
        full_results = example.run_full_analysis(
            symbol='AAPL',
            strategy_type='ma_cross',
            initial_capital=100000.0
        )
        
        # 您还可以单独运行各个步骤
        # data = example.prepare_data(symbol='AAPL')
        # example.setup_strategy(strategy_type='ma_cross')
        # example.setup_backtester(data=data)
        # backtest_results = example.run_backtest()
        # performance_results = example.analyze_performance()
    except KeyboardInterrupt:
        logger.info("用户中断了回测")
    except Exception as e:
        logger.error(f"运行回测示例时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()