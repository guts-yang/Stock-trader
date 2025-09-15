import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import json
import os

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StrategyEvaluator:
    """策略评估器
    用于评估交易策略的性能，提供多种评估指标和可视化功能
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 risk_free_rate: float = 0.03,
                 benchmark_returns: Optional[pd.Series] = None,
                 log_level: int = logging.INFO):
        """初始化策略评估器
        
        Args:
            config: 配置字典
            risk_free_rate: 无风险利率
            benchmark_returns: 基准收益率
            log_level: 日志级别
        """
        self.config = config or {}
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        
        # 评估结果
        self.evaluation_results = {}
        self.performance_metrics = {}
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("StrategyEvaluator 初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"strategy_evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def evaluate_performance(self, 
                            equity_curve: pd.Series,
                            trades: List[Dict[str, Any]] = None,
                            positions: pd.DataFrame = None) -> Dict[str, Any]:
        """评估策略性能
        
        Args:
            equity_curve: 权益曲线
            trades: 交易记录列表
            positions: 头寸记录DataFrame
        
        Returns:
            性能评估结果字典
        """
        try:
            logger.info("开始评估策略性能")
            
            # 验证输入数据
            if not isinstance(equity_curve, pd.Series) or len(equity_curve) < 2:
                logger.error("无效的权益曲线数据")
                return {}
            
            # 确保权益曲线有日期索引
            if not isinstance(equity_curve.index, pd.DatetimeIndex):
                logger.warning("权益曲线索引不是日期类型，尝试转换")
                try:
                    equity_curve.index = pd.to_datetime(equity_curve.index)
                except Exception as e:
                    logger.error(f"转换索引为日期类型失败: {str(e)}")
                    return {}
            
            # 计算基本性能指标
            metrics = self._calculate_basic_metrics(equity_curve)
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(equity_curve)
            metrics.update(risk_metrics)
            
            # 如果有交易记录，计算交易相关指标
            if trades:
                trade_metrics = self._calculate_trade_metrics(trades)
                metrics.update(trade_metrics)
            
            # 如果有基准收益率，计算相对性能指标
            if self.benchmark_returns is not None:
                relative_metrics = self._calculate_relative_metrics(equity_curve, self.benchmark_returns)
                metrics.update(relative_metrics)
            
            # 保存评估结果
            self.performance_metrics = metrics
            self.evaluation_results = {
                'metrics': metrics,
                'equity_curve': equity_curve,
                'trades': trades,
                'positions': positions
            }
            
            logger.info("策略性能评估完成")
            return metrics
        except Exception as e:
            logger.error(f"评估策略性能时发生异常: {str(e)}")
            return {}
    
    def _calculate_basic_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """计算基本性能指标
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            基本性能指标字典
        """
        # 计算总收益率
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # 计算年化收益率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (252 / days) - 1
        else:
            annualized_return = 0.0
        
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算平均日收益率
        avg_daily_return = daily_returns.mean()
        
        # 计算中位数日收益率
        median_daily_return = daily_returns.median()
        
        # 计算累计收益率曲线
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # 构建基本指标字典
        basic_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_daily_return': avg_daily_return,
            'median_daily_return': median_daily_return,
            'start_date': equity_curve.index[0],
            'end_date': equity_curve.index[-1],
            'duration_days': days,
            'n_periods': len(equity_curve)
        }
        
        return basic_metrics
    
    def _calculate_risk_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """计算风险指标
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            风险指标字典
        """
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        sharpe_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / volatility if volatility != 0 else 0
        
        # 计算索提诺比率（使用下行风险）
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / downside_volatility
        
        # 计算最大回撤和回撤持续时间
        drawdown, max_drawdown, max_drawdown_days = self._calculate_drawdown(equity_curve)
        
        # 计算Calmar比率（年化收益率除以最大回撤）
        calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 计算Omega比率（假设门槛收益率为无风险利率）
        threshold_return = self.risk_free_rate / 252  # 日度门槛收益率
        upside_returns = daily_returns[daily_returns > threshold_return]
        downside_returns = threshold_return - daily_returns[daily_returns < threshold_return]
        omega_ratio = upside_returns.sum() / downside_returns.sum() if len(downside_returns) > 0 else 0
        
        # 计算收益率的偏度和峰度
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        # 构建风险指标字典
        risk_metrics = {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_days': max_drawdown_days,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_volatility': downside_volatility
        }
        
        return risk_metrics
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[pd.Series, float, int]:
        """计算回撤
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            (回撤序列, 最大回撤, 最大回撤持续天数)
        """
        # 计算累积最大权益
        cumulative_max = equity_curve.cummax()
        
        # 计算回撤
        drawdown = (equity_curve / cumulative_max) - 1
        
        # 计算最大回撤
        max_drawdown = drawdown.min()
        
        # 找到最大回撤的开始和结束日期
        max_drawdown_end_idx = drawdown.idxmin()
        if isinstance(max_drawdown_end_idx, pd.Timestamp):
            # 单资产情况
            cumulative_max_before = cumulative_max.loc[:max_drawdown_end_idx]
            max_drawdown_start_idx = cumulative_max_before.idxmax()
            
            # 计算最大回撤持续天数
            max_drawdown_days = (max_drawdown_end_idx - max_drawdown_start_idx).days
        else:
            # 多资产情况（不常见）
            max_drawdown_days = 0
        
        return drawdown, max_drawdown, max_drawdown_days
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算交易相关指标
        
        Args:
            trades: 交易记录列表
        
        Returns:
            交易指标字典
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit_per_trade': 0,
                'avg_loss_per_trade': 0,
                'max_win_trade': 0,
                'max_loss_trade': 0,
                'avg_holding_period_days': 0
            }
        
        # 计算交易利润
        profits = []
        wins = 0
        total_profit = 0
        total_loss = 0
        holding_periods = []
        
        # 按日期排序交易记录
        sorted_trades = sorted(trades, key=lambda x: x['date'])
        
        # 跟踪每个资产的持仓
        positions = {}
        
        for i, trade in enumerate(sorted_trades):
            asset = trade['asset']
            quantity = trade['quantity']
            price = trade['price']
            date = trade['date']
            
            # 如果是新的资产或平仓交易，计算利润
            if asset not in positions or (positions[asset]['quantity'] > 0 and quantity < 0) or (positions[asset]['quantity'] < 0 and quantity > 0):
                if asset in positions:
                    # 计算持仓期间
                    holding_period = (date - positions[asset]['entry_date']).days
                    holding_periods.append(holding_period)
                    
                    # 计算利润
                    entry_price = positions[asset]['entry_price']
                    trade_quantity = abs(positions[asset]['quantity'])
                    
                    if positions[asset]['quantity'] > 0:
                        # 多头平仓
                        profit = trade_quantity * (price - entry_price)
                    else:
                        # 空头平仓
                        profit = trade_quantity * (entry_price - price)
                    
                    profits.append(profit)
                    
                    if profit > 0:
                        wins += 1
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
                    
                    # 移除已平仓的资产
                    del positions[asset]
                
                # 如果有新的持仓，记录入场信息
                if quantity != 0:
                    positions[asset] = {
                        'entry_date': date,
                        'entry_price': price,
                        'quantity': quantity
                    }
        
        # 计算各项交易指标
        total_trades = len(profits)
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        avg_profit_per_trade = sum(p for p in profits if p > 0) / wins if wins > 0 else 0
        avg_loss_per_trade = sum(abs(p) for p in profits if p < 0) / (total_trades - wins) if (total_trades - wins) > 0 else 0
        max_win_trade = max(profits) if profits else 0
        max_loss_trade = min(profits) if profits else 0
        avg_holding_period_days = sum(holding_periods) / len(holding_periods) if holding_periods else 0
        
        # 构建交易指标字典
        trade_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_loss_per_trade': avg_loss_per_trade,
            'max_win_trade': max_win_trade,
            'max_loss_trade': max_loss_trade,
            'avg_holding_period_days': avg_holding_period_days
        }
        
        return trade_metrics
    
    def _calculate_relative_metrics(self, 
                                  equity_curve: pd.Series,
                                  benchmark_returns: pd.Series) -> Dict[str, float]:
        """计算相对基准的性能指标
        
        Args:
            equity_curve: 权益曲线
            benchmark_returns: 基准收益率
        
        Returns:
            相对性能指标字典
        """
        # 计算策略收益率
        strategy_returns = equity_curve.pct_change().dropna()
        
        # 确保基准收益率和策略收益率有相同的时间范围
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            logger.warning("没有足够的共同数据点来计算相对性能指标")
            return {
                'alpha': 0,
                'beta': 0,
                'tracking_error': 0,
                'information_ratio': 0
            }
        
        # 对齐数据
        aligned_strategy = strategy_returns.loc[common_index]
        aligned_benchmark = benchmark_returns.loc[common_index]
        
        # 计算Alpha和Beta
        # 使用线性回归计算Beta
        X = aligned_benchmark.values.reshape(-1, 1)
        y = aligned_strategy.values
        
        # 添加截距项
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        
        # 执行线性回归
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            beta = model.coef_[0]
            alpha = model.intercept_ * 252  # 年化Alpha
        except ImportError:
            # 如果没有sklearn，使用numpy计算
            beta, alpha = np.polyfit(X.flatten(), y, 1)
            alpha = alpha * 252  # 年化Alpha
        
        # 计算跟踪误差
        tracking_error = (aligned_strategy - aligned_benchmark).std() * np.sqrt(252)
        
        # 计算信息比率
        information_ratio = (aligned_strategy.mean() - aligned_benchmark.mean()) * 252 / tracking_error if tracking_error != 0 else 0
        
        # 构建相对指标字典
        relative_metrics = {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
        
        return relative_metrics
    
    def evaluate_risk_adjusted_performance(self, 
                                          equity_curve: pd.Series,
                                          risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """评估风险调整后收益
        
        Args:
            equity_curve: 权益曲线
            risk_free_rate: 无风险利率
        
        Returns:
            风险调整后收益指标字典
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算年化收益率
        annualized_return = daily_returns.mean() * 252
        
        # 计算波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 计算最大回撤
        _, max_drawdown, _ = self._calculate_drawdown(equity_curve)
        
        # 计算各种风险调整后收益指标
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(daily_returns, risk_free_rate)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        omega_ratio = self._calculate_omega_ratio(daily_returns, risk_free_rate)
        
        # 构建风险调整后收益指标字典
        risk_adjusted_metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown
        }
        
        return risk_adjusted_metrics
    
    def _calculate_sortino_ratio(self, 
                               returns: pd.Series,
                               risk_free_rate: float) -> float:
        """计算索提诺比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
        
        Returns:
            索提诺比率
        """
        # 计算超额收益率
        excess_returns = returns - (risk_free_rate / 252)  # 假设日度无风险利率
        
        # 计算下行收益率
        downside_returns = excess_returns[excess_returns < 0]
        
        # 计算下行波动率
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        
        # 计算年化平均超额收益率
        annualized_excess_return = excess_returns.mean() * 252
        
        # 计算索提诺比率
        sortino_ratio = annualized_excess_return / downside_volatility
        
        return sortino_ratio
    
    def _calculate_omega_ratio(self, 
                             returns: pd.Series,
                             risk_free_rate: float) -> float:
        """计算Omega比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
        
        Returns:
            Omega比率
        """
        # 计算门槛收益率（日度）
        threshold_return = risk_free_rate / 252
        
        # 分离上行和下行收益率
        upside_returns = returns[returns > threshold_return]
        downside_returns = threshold_return - returns[returns < threshold_return]
        
        # 计算Omega比率
        if len(downside_returns) > 0:
            omega_ratio = upside_returns.sum() / downside_returns.sum()
        else:
            omega_ratio = float('inf')  # 没有下行风险
        
        return omega_ratio
    
    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """计算月度收益率
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            月度收益率DataFrame
        """
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算月度收益率
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 转换为DataFrame并添加年份和月份列
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Month_Name': monthly_returns.index.month_name(),
            'Return': monthly_returns.values,
            'Return_Pct': monthly_returns.values * 100
        })
        
        return monthly_data
    
    def calculate_rolling_metrics(self, 
                                 equity_curve: pd.Series,
                                 window: int = 252) -> pd.DataFrame:
        """计算滚动性能指标
        
        Args:
            equity_curve: 权益曲线
            window: 滚动窗口大小（交易日）
        
        Returns:
            滚动指标DataFrame
        """
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 计算滚动年化收益率
        rolling_return = daily_returns.rolling(window=window).apply(lambda x: (1 + x).prod() ** (252 / len(x)) - 1)
        
        # 计算滚动波动率
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)
        
        # 计算滚动夏普比率
        rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol
        
        # 构建滚动指标DataFrame
        rolling_metrics = pd.DataFrame({
            'Rolling_Return': rolling_return,
            'Rolling_Volatility': rolling_vol,
            'Rolling_Sharpe': rolling_sharpe
        })
        
        return rolling_metrics
    
    def generate_evaluation_report(self, 
                                  output_file: Optional[str] = None,
                                  detailed: bool = True) -> Dict[str, Any]:
        """生成评估报告
        
        Args:
            output_file: 输出文件路径
            detailed: 是否生成详细报告
        
        Returns:
            格式化的评估报告数据
        """
        if not self.evaluation_results:
            logger.warning("没有评估结果可生成报告")
            return {}
        
        # 构建报告
        report = {
            'report_date': datetime.now(),
            'summary': {
                'total_return_pct': self.performance_metrics.get('total_return', 0) * 100,
                'annualized_return_pct': self.performance_metrics.get('annualized_return', 0) * 100,
                'volatility_pct': self.performance_metrics.get('volatility', 0) * 100,
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': self.performance_metrics.get('max_drawdown', 0) * 100,
                'total_trades': self.performance_metrics.get('total_trades', 0),
                'win_rate_pct': self.performance_metrics.get('win_rate', 0) * 100,
                'backtest_period': f"{self.performance_metrics.get('start_date', '')} to {self.performance_metrics.get('end_date', '')}",
                'duration_days': self.performance_metrics.get('duration_days', 0)
            }
        }
        
        # 如果需要详细报告
        if detailed:
            report['detailed_metrics'] = self.performance_metrics
            report['equity_curve'] = self.evaluation_results['equity_curve'].to_dict() if self.evaluation_results['equity_curve'] is not None else {}
            
            # 如果有交易记录，添加交易统计
            if self.evaluation_results['trades']:
                # 计算月度交易统计
                trades_df = pd.DataFrame(self.evaluation_results['trades'])
                if not trades_df.empty and 'date' in trades_df.columns:
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    monthly_trades = trades_df.resample('M', on='date').size()
                    report['monthly_trades'] = monthly_trades.to_dict()
        
        # 如果指定了输出文件，保存报告
        if output_file:
            try:
                # 确保目录存在
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 保存为JSON文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    # 将datetime对象和DataFrame转换为可序列化的格式
                    report_serializable = report.copy()
                    if 'report_date' in report_serializable and isinstance(report_serializable['report_date'], datetime):
                        report_serializable['report_date'] = report_serializable['report_date'].strftime('%Y-%m-%d %H:%M:%S')
                    if 'start_date' in report_serializable.get('summary', {}) and isinstance(report_serializable['summary']['start_date'], datetime):
                        report_serializable['summary']['start_date'] = report_serializable['summary']['start_date'].strftime('%Y-%m-%d')
                    if 'end_date' in report_serializable.get('summary', {}) and isinstance(report_serializable['summary']['end_date'], datetime):
                        report_serializable['summary']['end_date'] = report_serializable['summary']['end_date'].strftime('%Y-%m-%d')
                    
                    json.dump(report_serializable, f, indent=2, ensure_ascii=False)
                
                logger.info(f"评估报告已保存到: {output_file}")
            except Exception as e:
                logger.error(f"保存评估报告时发生异常: {str(e)}")
        
        logger.info("评估报告生成完成")
        
        return report
    
    def plot_performance_summary(self, 
                                figsize: Tuple[int, int] = (12, 10),
                                show: bool = True,
                                save_path: Optional[str] = None) -> Any:
        """绘制性能摘要图表
        
        Args:
            figsize: 图表尺寸
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib图表对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not self.evaluation_results or 'equity_curve' not in self.evaluation_results:
                logger.warning("没有评估结果可绘制性能摘要")
                return None
            
            equity_curve = self.evaluation_results['equity_curve']
            
            # 创建图表
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # 绘制权益曲线
            ax1.plot(equity_curve.index, equity_curve, label='Equity Curve')
            ax1.set_ylabel('Equity ($)')
            ax1.set_title('Strategy Performance Summary')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 绘制回撤曲线
            drawdown, _, _ = self._calculate_drawdown(equity_curve)
            ax2.fill_between(drawdown.index, drawdown * 100, 0, where=drawdown < 0, 
                           facecolor='red', alpha=0.3, label='Drawdown (%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 绘制月度收益率柱状图
            monthly_returns = self.calculate_monthly_returns(equity_curve)
            if not monthly_returns.empty:
                monthly_returns['date'] = pd.to_datetime(monthly_returns[['Year', 'Month']].assign(day=1))
                ax3.bar(monthly_returns['date'], monthly_returns['Return_Pct'], label='Monthly Return (%)')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Monthly Return (%)')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            # 设置日期格式化
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"性能摘要图表已保存到: {save_path}")
            
            # 显示图表
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"绘制性能摘要图表时发生异常: {str(e)}")
            return None
    
    def plot_risk_return_scatter(self, 
                                strategies: Dict[str, Tuple[pd.Series, str]],
                                figsize: Tuple[int, int] = (10, 8),
                                show: bool = True,
                                save_path: Optional[str] = None) -> Any:
        """绘制风险-收益散点图
        
        Args:
            strategies: 策略字典，键为策略名称，值为(权益曲线, 颜色)元组
            figsize: 图表尺寸
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib图表对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines
            
            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)
            
            # 为每个策略计算风险和收益并绘图
            for name, (equity_curve, color) in strategies.items():
                # 计算年化收益率和波动率
                daily_returns = equity_curve.pct_change().dropna()
                annualized_return = daily_returns.mean() * 252
                volatility = daily_returns.std() * np.sqrt(252)
                
                # 绘制散点
                scatter = ax.scatter(volatility, annualized_return, c=color, label=name, s=100, alpha=0.7)
                
                # 添加标签
                ax.annotate(name, (volatility, annualized_return), 
                           xytext=(5, 5), textcoords='offset points')
            
            # 添加无风险利率参考线
            rf_line = mlines.Line2D([0, 1], [self.risk_free_rate, self.risk_free_rate], 
                                   color='black', linestyle='--', label=f'Risk-Free Rate ({self.risk_free_rate:.1%})')
            ax.add_line(rf_line)
            
            # 设置图表属性
            ax.set_title('Risk-Return Analysis')
            ax.set_xlabel('Annualized Volatility')
            ax.set_ylabel('Annualized Return')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置坐标轴范围
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=self.risk_free_rate - 0.1)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"风险-收益散点图已保存到: {save_path}")
            
            # 显示图表
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"绘制风险-收益散点图时发生异常: {str(e)}")
            return None
    
    def plot_monthly_heatmap(self, 
                           figsize: Tuple[int, int] = (12, 8),
                           show: bool = True,
                           save_path: Optional[str] = None) -> Any:
        """绘制月度收益率热力图
        
        Args:
            figsize: 图表尺寸
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib图表对象
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.evaluation_results or 'equity_curve' not in self.evaluation_results:
                logger.warning("没有评估结果可绘制月度收益率热力图")
                return None
            
            equity_curve = self.evaluation_results['equity_curve']
            
            # 计算月度收益率
            monthly_returns = self.calculate_monthly_returns(equity_curve)
            
            if monthly_returns.empty:
                logger.warning("没有足够的月度数据绘制热力图")
                return None
            
            # 创建透视表
            pivot_table = monthly_returns.pivot('Year', 'Month_Name', 'Return_Pct')
            
            # 确保月份顺序正确
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            pivot_table = pivot_table.reindex(columns=month_order)
            
            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制热力图
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", 
                       center=0, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
            
            # 设置图表属性
            ax.set_title('Monthly Returns Heatmap')
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"月度收益率热力图已保存到: {save_path}")
            
            # 显示图表
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"绘制月度收益率热力图时发生异常: {str(e)}")
            return None
    
    def save_evaluation_results(self, 
                               output_dir: Optional[str] = None,
                               prefix: str = 'strategy_evaluation') -> Dict[str, str]:
        """保存评估结果
        
        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        
        Returns:
            保存的文件路径字典
        """
        try:
            logger.info("保存评估结果")
            
            if not self.evaluation_results:
                logger.warning("没有评估结果可保存")
                return {}
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存目录
            if output_dir is None:
                output_dir = os.path.join(self.config.get('results_dir', './results'), 'strategy_evaluation')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 准备保存的文件路径
            saved_files = {}
            
            # 保存性能指标
            metrics_file = os.path.join(output_dir, f"{prefix}_{timestamp}_metrics.json")
            # 将datetime对象转换为字符串
            metrics_serializable = self.performance_metrics.copy()
            for key, value in metrics_serializable.items():
                if isinstance(value, datetime):
                    metrics_serializable[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
            
            saved_files['metrics'] = metrics_file
            
            # 保存权益曲线
            if 'equity_curve' in self.evaluation_results and self.evaluation_results['equity_curve'] is not None:
                equity_file = os.path.join(output_dir, f"{prefix}_{timestamp}_equity.csv")
                self.evaluation_results['equity_curve'].to_csv(equity_file)
                saved_files['equity_curve'] = equity_file
            
            # 保存交易记录
            if 'trades' in self.evaluation_results and self.evaluation_results['trades']:
                trades_file = os.path.join(output_dir, f"{prefix}_{timestamp}_trades.csv")
                pd.DataFrame(self.evaluation_results['trades']).to_csv(trades_file, index=False)
                saved_files['trades'] = trades_file
            
            # 保存头寸记录
            if 'positions' in self.evaluation_results and self.evaluation_results['positions'] is not None and not self.evaluation_results['positions'].empty:
                positions_file = os.path.join(output_dir, f"{prefix}_{timestamp}_positions.csv")
                self.evaluation_results['positions'].to_csv(positions_file)
                saved_files['positions'] = positions_file
            
            logger.info(f"评估结果已保存到: {output_dir}")
            
            return saved_files
        except Exception as e:
            logger.error(f"保存评估结果时发生异常: {str(e)}")
            raise
    
    def compare_strategies(self, 
                          strategies: Dict[str, pd.Series],
                          benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
        """比较多个策略的性能
        
        Args:
            strategies: 策略字典，键为策略名称，值为权益曲线
            benchmark: 基准权益曲线（可选）
        
        Returns:
            策略比较DataFrame
        """
        try:
            logger.info("比较多个策略的性能")
            
            # 准备比较结果
            comparison_data = []
            
            # 比较每个策略
            for name, equity_curve in strategies.items():
                # 评估策略性能
                metrics = self.evaluate_performance(equity_curve)
                
                # 提取关键指标
                strategy_data = {
                    'Strategy': name,
                    'Total Return (%)': metrics.get('total_return', 0) * 100,
                    'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                    'Volatility (%)': metrics.get('volatility', 0) * 100,
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                    'Calmar Ratio': metrics.get('calmar_ratio', 0),
                    'Sortino Ratio': metrics.get('sortino_ratio', 0),
                    'Total Trades': metrics.get('total_trades', 0),
                    'Win Rate (%)': metrics.get('win_rate', 0) * 100
                }
                
                comparison_data.append(strategy_data)
            
            # 如果有基准，添加基准比较
            if benchmark is not None:
                # 评估基准性能
                benchmark_metrics = self.evaluate_performance(benchmark)
                
                # 提取关键指标
                benchmark_data = {
                    'Strategy': 'Benchmark',
                    'Total Return (%)': benchmark_metrics.get('total_return', 0) * 100,
                    'Annualized Return (%)': benchmark_metrics.get('annualized_return', 0) * 100,
                    'Volatility (%)': benchmark_metrics.get('volatility', 0) * 100,
                    'Sharpe Ratio': benchmark_metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': benchmark_metrics.get('max_drawdown', 0) * 100,
                    'Calmar Ratio': benchmark_metrics.get('calmar_ratio', 0),
                    'Sortino Ratio': benchmark_metrics.get('sortino_ratio', 0),
                    'Total Trades': 0,
                    'Win Rate (%)': 0
                }
                
                comparison_data.append(benchmark_data)
            
            # 创建比较DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # 设置策略列为索引
            comparison_df.set_index('Strategy', inplace=True)
            
            # 按年化收益率排序
            comparison_df.sort_values('Annualized Return (%)', ascending=False, inplace=True)
            
            logger.info("策略比较完成")
            
            return comparison_df
        except Exception as e:
            logger.error(f"比较策略性能时发生异常: {str(e)}")
            return pd.DataFrame()

# 模块版本
__version__ = '0.1.0'