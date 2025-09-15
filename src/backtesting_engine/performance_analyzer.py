import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
import math
import copy

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PerformanceAnalyzer:
    """绩效分析器
    提供全面的交易策略绩效分析功能
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化绩效分析器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        self.returns = None  # 收益率数据
        self.equity_curve = None  # 权益曲线
        self.trades = None  # 交易记录
        self.positions = None  # 头寸数据
        self.risk_free_rate = self.config.get('risk_free_rate', 0.0)  # 无风险利率
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("绩效分析器初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"performance_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def set_data(self, 
                returns: Optional[pd.Series] = None,
                equity_curve: Optional[pd.Series] = None,
                trades: Optional[pd.DataFrame] = None,
                positions: Optional[Dict[str, pd.DataFrame]] = None):
        """设置分析数据
        
        Args:
            returns: 收益率数据
            equity_curve: 权益曲线
            trades: 交易记录
            positions: 头寸数据
        """
        # 设置收益率数据
        if returns is not None:
            self.returns = returns
            # 确保索引是datetime类型
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                try:
                    self.returns.index = pd.to_datetime(self.returns.index)
                except:
                    logger.warning("无法将收益率索引转换为datetime类型")
        
        # 设置权益曲线
        if equity_curve is not None:
            self.equity_curve = equity_curve
            # 确保索引是datetime类型
            if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
                try:
                    self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
                except:
                    logger.warning("无法将权益曲线索引转换为datetime类型")
        
        # 如果有权益曲线但没有收益率，计算收益率
        if self.equity_curve is not None and self.returns is None:
            self.returns = self.equity_curve.pct_change().fillna(0)
        
        # 设置交易记录
        if trades is not None:
            self.trades = trades
        
        # 设置头寸数据
        if positions is not None:
            self.positions = positions
        
        logger.info("分析数据已设置")
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """计算基本绩效指标
        
        Returns:
            基本绩效指标字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算基本绩效指标")
            return {}
        
        try:
            # 累计收益率
            total_return = (self.returns + 1).prod() - 1
            
            # 年化收益率
            years = len(self.returns) / 252  # 假设一年252个交易日
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # 最大回撤
            cum_returns = (self.returns + 1).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 波动率
            volatility = self.returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # 索提诺比率
            downside_returns = self.returns[self.returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # 胜率
            win_rate = len(self.returns[self.returns > 0]) / len(self.returns) if len(self.returns) > 0 else 0
            
            # 平均盈亏比
            avg_win = self.returns[self.returns > 0].mean()
            avg_loss = self.returns[self.returns < 0].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trading_days': len(self.returns)
            }
            
            logger.info("基本绩效指标计算完成")
            return metrics
        except Exception as e:
            logger.error(f"计算基本绩效指标时发生异常: {str(e)}")
            return {}
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """计算风险指标
        
        Returns:
            风险指标字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算风险指标")
            return {}
        
        try:
            # VaR (Value at Risk) - 历史模拟法
            var_95 = self.returns.quantile(0.05)
            var_99 = self.returns.quantile(0.01)
            
            # CVaR (Conditional Value at Risk)
            cvar_95 = self.returns[self.returns <= var_95].mean()
            cvar_99 = self.returns[self.returns <= var_99].mean()
            
            # 最大回撤和回撤持续时间
            cum_returns = (self.returns + 1).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            
            # 找到最大回撤的开始和结束时间
            max_drawdown_end_idx = drawdown.idxmin()
            if pd.isna(max_drawdown_end_idx):
                max_drawdown_start_date = None
                max_drawdown_end_date = None
                max_drawdown_duration = 0
            else:
                # 找到最大回撤开始的位置
                max_drawdown_start_idx = cum_returns[:max_drawdown_end_idx].idxmax()
                
                # 计算回撤持续时间
                max_drawdown_start_date = max_drawdown_start_idx
                max_drawdown_end_date = max_drawdown_end_idx
                max_drawdown_duration = (max_drawdown_end_date - max_drawdown_start_date).days
            
            # 计算MAR比率 (年化收益率/最大回撤)
            annualized_return = self.calculate_basic_metrics().get('annualized_return', 0)
            max_drawdown = drawdown.min()
            mar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Calmar比率 (同MAR比率，但更严格的定义)
            calmar_ratio = mar_ratio
            
            # 计算omega比率
            threshold = self.risk_free_rate / 252  # 日无风险利率
            upside = self.returns[self.returns > threshold].sum()
            downside = self.returns[self.returns < threshold].abs().sum()
            omega_ratio = upside / downside if downside != 0 else 0
            
            metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'max_drawdown_start_date': max_drawdown_start_date,
                'max_drawdown_end_date': max_drawdown_end_date,
                'max_drawdown_duration': max_drawdown_duration,
                'mar_ratio': mar_ratio,
                'calmar_ratio': calmar_ratio,
                'omega_ratio': omega_ratio
            }
            
            logger.info("风险指标计算完成")
            return metrics
        except Exception as e:
            logger.error(f"计算风险指标时发生异常: {str(e)}")
            return {}
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """计算交易指标
        
        Returns:
            交易指标字典
        """
        if self.trades is None:
            logger.warning("没有设置交易记录，无法计算交易指标")
            return {}
        
        try:
            # 交易次数
            num_trades = len(self.trades)
            
            if num_trades == 0:
                return {
                    'num_trades': 0,
                    'win_rate': 0,
                    'avg_profit_per_trade': 0,
                    'avg_loss_per_trade': 0,
                    'profit_factor': 0,
                    'max_win_trade': 0,
                    'max_loss_trade': 0,
                    'avg_trade_duration': 0
                }
            
            # 胜率
            winning_trades = self.trades[self.trades['profit'] > 0]
            losing_trades = self.trades[self.trades['profit'] <= 0]
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            
            # 平均盈利和平均亏损
            avg_profit_per_trade = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss_per_trade = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
            
            # 盈亏比
            profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else 0
            
            # 最大盈利和最大亏损交易
            max_win_trade = winning_trades['profit'].max() if len(winning_trades) > 0 else 0
            max_loss_trade = losing_trades['profit'].min() if len(losing_trades) > 0 else 0
            
            # 平均交易持续时间
            if 'entry_time' in self.trades.columns and 'exit_time' in self.trades.columns:
                # 确保时间列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(self.trades['entry_time']):
                    self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])
                if not pd.api.types.is_datetime64_any_dtype(self.trades['exit_time']):
                    self.trades['exit_time'] = pd.to_datetime(self.trades['exit_time'])
                
                # 计算交易持续时间
                trade_durations = (self.trades['exit_time'] - self.trades['entry_time']).dt.total_seconds() / (60 * 60 * 24)  # 转换为天
                avg_trade_duration = trade_durations.mean()
            else:
                avg_trade_duration = 0
            
            # 连续盈利和连续亏损
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for _, trade in self.trades.iterrows():
                if trade['profit'] > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            metrics = {
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_loss_per_trade': avg_loss_per_trade,
                'profit_factor': profit_factor,
                'max_win_trade': max_win_trade,
                'max_loss_trade': max_loss_trade,
                'avg_trade_duration': avg_trade_duration,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses
            }
            
            logger.info("交易指标计算完成")
            return metrics
        except Exception as e:
            logger.error(f"计算交易指标时发生异常: {str(e)}")
            return {}
    
    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """计算高级绩效指标
        
        Returns:
            高级绩效指标字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算高级绩效指标")
            return {}
        
        try:
            # 计算Kappa指标 (Kindleberger's Kappa)
            # 衡量收益率分布的偏度
            kappa = stats.skew(self.returns)
            
            # 计算峰度
            kurtosis = stats.kurtosis(self.returns)
            
            # 计算Omega比率
            # 已经在风险指标中计算过了
            
            # 计算信息比率
            # 假设没有基准收益率数据
            information_ratio = 0
            
            # 计算跟踪误差
            # 假设没有基准收益率数据
            tracking_error = 0
            
            # 计算Beta值
            # 假设没有基准收益率数据
            beta = 0
            
            # 计算Alpha值
            annualized_return = self.calculate_basic_metrics().get('annualized_return', 0)
            alpha = annualized_return - self.risk_free_rate - beta * (annualized_return - self.risk_free_rate)
            
            # 计算Treynor比率
            treynor_ratio = (annualized_return - self.risk_free_rate) / beta if beta != 0 else 0
            
            # 计算Calmar比率
            # 已经在风险指标中计算过了
            
            # 计算Martin比率
            # (年化收益率 - 无风险利率) / 最大回撤
            max_drawdown = self.calculate_risk_metrics().get('max_drawdown', 0)
            martin_ratio = (annualized_return - self.risk_free_rate) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 计算Sterling比率
            # (年化收益率 - 无风险利率) / 平均最大回撤
            # 计算过去3年的平均最大回撤（简化计算）
            avg_max_drawdown = abs(max_drawdown)  # 简化计算
            sterling_ratio = (annualized_return - self.risk_free_rate) / avg_max_drawdown if avg_max_drawdown != 0 else 0
            
            metrics = {
                'kappa': kappa,
                'kurtosis': kurtosis,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'treynor_ratio': treynor_ratio,
                'martin_ratio': martin_ratio,
                'sterling_ratio': sterling_ratio
            }
            
            logger.info("高级绩效指标计算完成")
            return metrics
        except Exception as e:
            logger.error(f"计算高级绩效指标时发生异常: {str(e)}")
            return {}
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """计算月度收益率
        
        Returns:
            月度收益率数据框
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算月度收益率")
            return pd.DataFrame()
        
        try:
            # 确保索引是datetime类型
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                try:
                    returns_data = self.returns.copy()
                    returns_data.index = pd.to_datetime(returns_data.index)
                except:
                    logger.error("无法将收益率索引转换为datetime类型")
                    return pd.DataFrame()
            else:
                returns_data = self.returns.copy()
            
            # 计算月度收益率
            monthly_returns = returns_data.resample('M').apply(lambda x: (x + 1).prod() - 1)
            
            logger.info("月度收益率计算完成")
            return monthly_returns
        except Exception as e:
            logger.error(f"计算月度收益率时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rolling_metrics(self, window: int = 252) -> Dict[str, pd.Series]:
        """计算滚动绩效指标
        
        Args:
            window: 滚动窗口大小（交易日）
        
        Returns:
            滚动绩效指标字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算滚动绩效指标")
            return {}
        
        try:
            rolling_metrics = {}
            
            # 滚动收益率
            rolling_returns = self.returns.rolling(window).apply(lambda x: (x + 1).prod() - 1)
            rolling_metrics['rolling_returns'] = rolling_returns
            
            # 滚动波动率
            rolling_volatility = self.returns.rolling(window).std() * np.sqrt(252)
            rolling_metrics['rolling_volatility'] = rolling_volatility
            
            # 滚动夏普比率
            risk_free_rate_daily = self.risk_free_rate / 252
            excess_returns = self.returns - risk_free_rate_daily
            rolling_sharpe = excess_returns.rolling(window).mean() / self.returns.rolling(window).std() * np.sqrt(252)
            rolling_metrics['rolling_sharpe'] = rolling_sharpe
            
            # 滚动最大回撤
            cum_returns = (self.returns + 1).cumprod()
            rolling_cum_returns = cum_returns.rolling(window)
            rolling_max = rolling_cum_returns.max()
            rolling_drawdown = (cum_returns / rolling_max) - 1
            rolling_metrics['rolling_drawdown'] = rolling_drawdown
            
            logger.info(f"滚动绩效指标计算完成，窗口大小: {window}")
            return rolling_metrics
        except Exception as e:
            logger.error(f"计算滚动绩效指标时发生异常: {str(e)}")
            return {}
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """生成完整的绩效摘要
        
        Returns:
            完整的绩效摘要字典
        """
        try:
            # 计算各类指标
            basic_metrics = self.calculate_basic_metrics()
            risk_metrics = self.calculate_risk_metrics()
            trade_metrics = self.calculate_trade_metrics()
            advanced_metrics = self.calculate_advanced_metrics()
            monthly_returns = self.calculate_monthly_returns()
            
            # 构建完整摘要
            summary = {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'trade_metrics': trade_metrics,
                'advanced_metrics': advanced_metrics,
                'monthly_returns': monthly_returns.to_dict() if not monthly_returns.empty else {},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("绩效摘要生成完成")
            return summary
        except Exception as e:
            logger.error(f"生成绩效摘要时发生异常: {str(e)}")
            return {}
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制权益曲线
        
        Args:
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        if self.equity_curve is None:
            logger.warning("没有设置权益曲线数据，无法绘制权益曲线")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制权益曲线
            ax.plot(self.equity_curve.index, self.equity_curve, label='Equity Curve')
            
            # 设置标题和标签
            ax.set_title('Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.grid(True)
            ax.legend()
            
            # 自动调整日期标签
            fig.autofmt_xdate()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"权益曲线图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制权益曲线时发生异常: {str(e)}")
            return None
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制回撤曲线
        
        Args:
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法绘制回撤曲线")
            return None
        
        try:
            # 计算累积收益率和回撤
            cum_returns = (self.returns + 1).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制回撤曲线
            ax.plot(drawdown.index, drawdown, label='Drawdown')
            
            # 填充负值区域
            ax.fill_between(drawdown.index, drawdown, alpha=0.3)
            
            # 设置标题和标签
            ax.set_title('Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.grid(True)
            ax.legend()
            
            # 设置y轴为百分比格式
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            # 自动调整日期标签
            fig.autofmt_xdate()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回撤曲线图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制回撤曲线时发生异常: {str(e)}")
            return None
    
    def plot_monthly_returns_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制月度收益率热力图
        
        Args:
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        monthly_returns = self.calculate_monthly_returns()
        if monthly_returns.empty:
            logger.warning("没有月度收益率数据，无法绘制热力图")
            return None
        
        try:
            # 构建月度/年度的透视表
            monthly_returns_df = monthly_returns.to_frame(name='return')
            monthly_returns_df['year'] = monthly_returns_df.index.year
            monthly_returns_df['month'] = monthly_returns_df.index.month
            
            # 创建透视表
            heatmap_data = monthly_returns_df.pivot('year', 'month', 'return')
            
            # 重命名列以显示月份名称
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = [month_names[i] for i in heatmap_data.columns]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制热力图
            sns.heatmap(heatmap_data, annot=True, fmt='.2%', center=0, cmap='RdYlGn', ax=ax)
            
            # 设置标题
            ax.set_title('Monthly Returns Heatmap')
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"月度收益率热力图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制月度收益率热力图时发生异常: {str(e)}")
            return None
    
    def plot_rolling_metrics(self, window: int = 252, save_path: Optional[str] = None) -> plt.Figure:
        """绘制滚动绩效指标
        
        Args:
            window: 滚动窗口大小
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        rolling_metrics = self.calculate_rolling_metrics(window)
        if not rolling_metrics:
            logger.warning("没有滚动绩效指标数据，无法绘制图表")
            return None
        
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # 绘制滚动收益率
            if 'rolling_returns' in rolling_metrics:
                ax1.plot(rolling_metrics['rolling_returns'].index, rolling_metrics['rolling_returns'])
                ax1.set_title(f'Rolling {window}-Day Returns')
                ax1.set_ylabel('Return')
                ax1.grid(True)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            # 绘制滚动波动率
            if 'rolling_volatility' in rolling_metrics:
                ax2.plot(rolling_metrics['rolling_volatility'].index, rolling_metrics['rolling_volatility'])
                ax2.set_title(f'Rolling {window}-Day Volatility')
                ax2.set_ylabel('Volatility')
                ax2.grid(True)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            # 绘制滚动夏普比率
            if 'rolling_sharpe' in rolling_metrics:
                ax3.plot(rolling_metrics['rolling_sharpe'].index, rolling_metrics['rolling_sharpe'])
                ax3.set_title(f'Rolling {window}-Day Sharpe Ratio')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.grid(True)
            
            # 自动调整日期标签
            fig.autofmt_xdate()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"滚动绩效指标图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制滚动绩效指标时发生异常: {str(e)}")
            return None
    
    def plot_monthly_returns_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制月度收益率分布
        
        Args:
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        monthly_returns = self.calculate_monthly_returns()
        if monthly_returns.empty:
            logger.warning("没有月度收益率数据，无法绘制分布")
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 绘制直方图
            ax1.hist(monthly_returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('Monthly Returns Distribution')
            ax1.set_xlabel('Return')
            ax1.set_ylabel('Frequency')
            ax1.grid(True)
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
            # 绘制箱线图
            ax2.boxplot(monthly_returns)
            ax2.set_title('Monthly Returns Box Plot')
            ax2.set_ylabel('Return')
            ax2.grid(True)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"月度收益率分布图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制月度收益率分布时发生异常: {str(e)}")
            return None
    
    def save_performance_report(self, report_path: str) -> bool:
        """保存绩效报告
        
        Args:
            report_path: 报告保存路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # 生成绩效摘要
            performance_summary = self.generate_performance_summary()
            
            # 保存为JSON文件
            with open(report_path, 'w') as f:
                json.dump(performance_summary, f, indent=4, default=str)
            
            logger.info(f"绩效报告已保存到: {report_path}")
            return True
        except Exception as e:
            logger.error(f"保存绩效报告时发生异常: {str(e)}")
            return False
    
    def generate_comprehensive_report(self, report_dir: str) -> Dict[str, Any]:
        """生成综合绩效报告，包括数据和图表
        
        Args:
            report_dir: 报告目录
        
        Returns:
            报告信息字典
        """
        try:
            # 确保目录存在
            os.makedirs(report_dir, exist_ok=True)
            
            # 创建图表目录
            charts_dir = os.path.join(report_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # 生成绩效摘要
            performance_summary = self.generate_performance_summary()
            
            # 保存绩效报告
            report_path = os.path.join(report_dir, 'performance_summary.json')
            with open(report_path, 'w') as f:
                json.dump(performance_summary, f, indent=4, default=str)
            
            # 生成并保存所有图表
            charts = {
                'equity_curve': os.path.join(charts_dir, 'equity_curve.png'),
                'drawdown': os.path.join(charts_dir, 'drawdown.png'),
                'monthly_returns_heatmap': os.path.join(charts_dir, 'monthly_returns_heatmap.png'),
                'rolling_metrics_252': os.path.join(charts_dir, 'rolling_metrics_252.png'),
                'rolling_metrics_60': os.path.join(charts_dir, 'rolling_metrics_60.png'),
                'monthly_returns_distribution': os.path.join(charts_dir, 'monthly_returns_distribution.png')
            }
            
            # 绘制并保存图表
            self.plot_equity_curve(charts['equity_curve'])
            self.plot_drawdown(charts['drawdown'])
            self.plot_monthly_returns_heatmap(charts['monthly_returns_heatmap'])
            self.plot_rolling_metrics(window=252, save_path=charts['rolling_metrics_252'])
            self.plot_rolling_metrics(window=60, save_path=charts['rolling_metrics_60'])
            self.plot_monthly_returns_distribution(charts['monthly_returns_distribution'])
            
            # 构建报告信息
            report_info = {
                'report_path': report_path,
                'charts': charts,
                'timestamp': datetime.now().isoformat(),
                'summary': performance_summary
            }
            
            # 生成HTML报告
            html_report_path = os.path.join(report_dir, 'performance_report.html')
            self._generate_html_report(report_info, html_report_path)
            
            logger.info(f"综合绩效报告已生成到: {report_dir}")
            return report_info
        except Exception as e:
            logger.error(f"生成综合绩效报告时发生异常: {str(e)}")
            return {}
    
    def _generate_html_report(self, report_info: Dict[str, Any], html_path: str) -> None:
        """生成HTML格式的绩效报告
        
        Args:
            report_info: 报告信息
            html_path: HTML报告路径
        """
        try:
            # 构建HTML内容
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-name {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .chart {{ flex: 1 1 45%; min-width: 300px; }}
        .chart img {{ width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #777; }}
    </style>
</head>
<body>
    <h1>Trading Strategy Performance Report</h1>
    <p>Generated on: {report_info['timestamp']}</p>
    
    <h2>Basic Metrics</h2>
    <div class="metrics">
"""
            
            # 添加基本指标卡片
            basic_metrics = report_info['summary'].get('basic_metrics', {})
            if basic_metrics:
                html_content += f"""
        <div class="metric-card">
            <div class="metric-name">Total Return</div>
            <div class="metric-value">{(basic_metrics.get('total_return', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Annualized Return</div>
            <div class="metric-value">{(basic_metrics.get('annualized_return', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Max Drawdown</div>
            <div class="metric-value">{(basic_metrics.get('max_drawdown', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Volatility</div>
            <div class="metric-value">{(basic_metrics.get('volatility', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Sharpe Ratio</div>
            <div class="metric-value">{basic_metrics.get('sharpe_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Sortino Ratio</div>
            <div class="metric-value">{basic_metrics.get('sortino_ratio', 0):.2f}</div>
        </div>
"""
            
            html_content += """
    </div>
    
    <h2>Risk Metrics</h2>
    <div class="metrics">
"""
            
            # 添加风险指标卡片
            risk_metrics = report_info['summary'].get('risk_metrics', {})
            if risk_metrics:
                html_content += f"""
        <div class="metric-card">
            <div class="metric-name">VaR 95%</div>
            <div class="metric-value">{(risk_metrics.get('var_95', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">CVaR 95%</div>
            <div class="metric-value">{(risk_metrics.get('cvar_95', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">MAR Ratio</div>
            <div class="metric-value">{risk_metrics.get('mar_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Omega Ratio</div>
            <div class="metric-value">{risk_metrics.get('omega_ratio', 0):.2f}</div>
        </div>
"""
            
            html_content += """
    </div>
    
    <h2>Trade Metrics</h2>
    <div class="metrics">
"""
            
            # 添加交易指标卡片
            trade_metrics = report_info['summary'].get('trade_metrics', {})
            if trade_metrics:
                html_content += f"""
        <div class="metric-card">
            <div class="metric-name">Number of Trades</div>
            <div class="metric-value">{trade_metrics.get('num_trades', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Win Rate</div>
            <div class="metric-value">{(trade_metrics.get('win_rate', 0) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Profit Factor</div>
            <div class="metric-value">{trade_metrics.get('profit_factor', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Avg. Trade Duration (days)</div>
            <div class="metric-value">{trade_metrics.get('avg_trade_duration', 0):.1f}</div>
        </div>
"""
            
            html_content += """
    </div>
    
    <h2>Performance Charts</h2>
    <div class="charts">
"""
            
            # 添加图表
            charts = report_info.get('charts', {})
            if charts:
                # 计算相对路径（假设HTML报告在report_dir，图表在report_dir/charts）
                relative_charts = {k: os.path.basename(v) for k, v in charts.items()}
                
                html_content += f"""
        <div class="chart">
            <h3>Equity Curve</h3>
            <img src="charts/{relative_charts.get('equity_curve', '')}" alt="Equity Curve">
        </div>
        <div class="chart">
            <h3>Drawdown</h3>
            <img src="charts/{relative_charts.get('drawdown', '')}" alt="Drawdown">
        </div>
        <div class="chart">
            <h3>Monthly Returns Heatmap</h3>
            <img src="charts/{relative_charts.get('monthly_returns_heatmap', '')}" alt="Monthly Returns Heatmap">
        </div>
        <div class="chart">
            <h3>Monthly Returns Distribution</h3>
            <img src="charts/{relative_charts.get('monthly_returns_distribution', '')}" alt="Monthly Returns Distribution">
        </div>
        <div class="chart" style="flex: 1 1 100%;">
            <h3>Rolling Metrics (252-day window)</h3>
            <img src="charts/{relative_charts.get('rolling_metrics_252', '')}" alt="Rolling Metrics">
        </div>
"""
            
            html_content += """
    </div>
    
    <div class="footer">
        <p>Performance Report generated by StockTrader Backtesting Engine</p>
    </div>
</body>
</html>
"""
            
            # 保存HTML报告
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML绩效报告已生成到: {html_path}")
        except Exception as e:
            logger.error(f"生成HTML绩效报告时发生异常: {str(e)}")
    
    def compare_strategies(self, 
                         other_analyzers: List['PerformanceAnalyzer'],
                         strategy_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """比较多个策略的绩效
        
        Args:
            other_analyzers: 其他绩效分析器实例列表
            strategy_names: 策略名称列表
        
        Returns:
            比较结果字典
        """
        try:
            # 确保策略名称列表长度与分析器列表长度匹配
            if strategy_names and len(strategy_names) != len(other_analyzers) + 1:
                raise ValueError("策略名称列表长度必须与分析器列表长度+1（包括当前分析器）匹配")
            
            # 创建所有分析器的列表（包括当前分析器）
            all_analyzers = [self] + other_analyzers
            
            # 创建策略名称列表
            if not strategy_names:
                strategy_names = [f"Strategy_{i+1}" for i in range(len(all_analyzers))]
            else:
                # 确保策略名称列表包含当前分析器
                strategy_names = [strategy_names[0]] + strategy_names[1:]
            
            # 收集每个策略的绩效指标
            comparison_results = {}
            
            for i, analyzer in enumerate(all_analyzers):
                strategy_name = strategy_names[i]
                
                # 生成绩效摘要
                performance_summary = analyzer.generate_performance_summary()
                
                # 保存结果
                comparison_results[strategy_name] = performance_summary
            
            logger.info(f"已完成 {len(all_analyzers)} 个策略的绩效比较")
            return comparison_results
        except Exception as e:
            logger.error(f"比较策略绩效时发生异常: {str(e)}")
            return {}
    
    def plot_strategy_comparison(self, 
                               comparison_results: Dict[str, Any],
                               metrics_to_compare: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制策略比较图表
        
        Args:
            comparison_results: 比较结果字典
            metrics_to_compare: 要比较的指标列表
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            图表对象
        """
        try:
            # 默认要比较的指标
            if not metrics_to_compare:
                metrics_to_compare = ['annualized_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio']
            
            # 准备数据
            data = {}
            for strategy_name, results in comparison_results.items():
                data[strategy_name] = {}
                
                # 从基本指标中获取
                basic_metrics = results.get('basic_metrics', {})
                for metric in metrics_to_compare:
                    if metric in basic_metrics:
                        data[strategy_name][metric] = basic_metrics[metric]
                
                # 从风险指标中获取
                risk_metrics = results.get('risk_metrics', {})
                for metric in metrics_to_compare:
                    if metric in risk_metrics:
                        data[strategy_name][metric] = risk_metrics[metric]
            
            # 转换为数据框
            df = pd.DataFrame(data).T
            
            # 创建图表
            fig, axes = plt.subplots(math.ceil(len(metrics_to_compare) / 2), 2, figsize=(15, 5 * math.ceil(len(metrics_to_compare) / 2)))
            axes = axes.flatten()
            
            # 绘制每个指标的条形图
            for i, metric in enumerate(metrics_to_compare):
                if metric in df.columns:
                    # 绘制条形图
                    df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
                    
                    # 设置标题和标签
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_ylabel(metric)
                    axes[i].grid(True, axis='y')
                    
                    # 对于百分比指标，格式化y轴
                    if metric in ['annualized_return', 'max_drawdown', 'win_rate']:
                        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                    
                    # 在条形图上显示数值
                    for j, v in enumerate(df[metric]):
                        if metric in ['annualized_return', 'max_drawdown', 'win_rate']:
                            axes[i].text(j, v, f'{v:.1%}', ha='center', va='bottom')
                        else:
                            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
            
            # 移除多余的子图
            for i in range(len(metrics_to_compare), len(axes)):
                fig.delaxes(axes[i])
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"策略比较图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制策略比较图时发生异常: {str(e)}")
            return None

# 模块版本
__version__ = '0.1.0'