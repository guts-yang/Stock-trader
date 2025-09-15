import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any, Set
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
import inspect
from collections import defaultdict
import traceback
import re
import math

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BacktestUtils:
    """回测引擎工具类
    提供各种辅助功能，支持回测引擎的各个组件
    """
    @staticmethod
    def setup_logger(name: str, 
                    log_dir: str = './logs',
                    log_level: int = logging.INFO) -> logging.Logger:
        """设置日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
        
        Returns:
            配置好的日志记录器
        """
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 创建文件处理器
            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 设置日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return {}
            
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        import yaml
                        config = yaml.safe_load(f)
                    except ImportError:
                        logger.error("加载YAML配置需要PyYAML库")
                        return {}
                else:
                    logger.error(f"不支持的配置文件格式: {config_path}")
                    return {}
            
            logger.info(f"配置文件已从 {config_path} 加载")
            return config
        except Exception as e:
            logger.error(f"加载配置文件时发生异常: {str(e)}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """保存配置文件
        
        Args:
            config: 配置字典
            config_path: 配置文件保存路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                if config_path.endswith('.json'):
                    json.dump(config, f, indent=4, default=str)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        import yaml
                        yaml.dump(config, f, default_flow_style=False)
                    except ImportError:
                        logger.error("保存YAML配置需要PyYAML库")
                        return False
                else:
                    logger.error(f"不支持的配置文件格式: {config_path}")
                    return False
            
            logger.info(f"配置已保存到 {config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件时发生异常: {str(e)}")
            return False
    
    @staticmethod
    def calculate_returns(prices: pd.Series, 
                        log_returns: bool = False) -> pd.Series:
        """计算收益率
        
        Args:
            prices: 价格序列
            log_returns: 是否计算对数收益率
        
        Returns:
            收益率序列
        """
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """计算累积收益率
        
        Args:
            returns: 收益率序列
        
        Returns:
            累积收益率序列
        """
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_drawdown(returns: pd.Series) -> pd.Series:
        """计算回撤
        
        Args:
            returns: 收益率序列
        
        Returns:
            回撤序列
        """
        # 计算累积收益
        cumulative = (1 + returns).cumprod()
        # 计算运行中的最大值
        running_max = cumulative.expanding().max()
        # 计算回撤
        drawdown = (cumulative / running_max) - 1
        return drawdown
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """计算最大回撤
        
        Args:
            returns: 收益率序列
        
        Returns:
            (最大回撤值, 回撤开始时间, 回撤结束时间)
        """
        drawdown = BacktestUtils.calculate_drawdown(returns)
        max_drawdown = drawdown.min()
        
        # 找到最大回撤的结束位置
        end_idx = drawdown.idxmin()
        
        # 找到最大回撤的开始位置
        if end_idx is not None:
            # 在结束位置之前的最大值
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            
            # 找到结束位置之前的峰值位置
            peak_idx = running_max[:end_idx].idxmax()
            return max_drawdown, peak_idx, end_idx
        else:
            return 0.0, None, None
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.0, 
                              annualized: bool = True) -> float:
        """计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            annualized: 是否年化
        
        Returns:
            夏普比率
        """
        # 计算超额收益
        excess_returns = returns - (risk_free_rate / 252)  # 假设每日数据
        
        # 计算夏普比率
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()
        
        if std_excess_return == 0:
            return 0.0
        
        sharpe = mean_excess_return / std_excess_return
        
        # 年化
        if annualized:
            sharpe *= math.sqrt(252)  # 假设每日数据
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, 
                               risk_free_rate: float = 0.0, 
                               annualized: bool = True) -> float:
        """计算索提诺比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            annualized: 是否年化
        
        Returns:
            索提诺比率
        """
        # 计算超额收益
        excess_returns = returns - (risk_free_rate / 252)  # 假设每日数据
        
        # 计算下行标准差
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        
        # 计算索提诺比率
        mean_excess_return = excess_returns.mean()
        
        if downside_std == 0:
            return float('inf') if mean_excess_return > 0 else float('-inf')
        
        sortino = mean_excess_return / downside_std
        
        # 年化
        if annualized:
            sortino *= math.sqrt(252)  # 假设每日数据
        
        return sortino
    
    @staticmethod
    def calculate_romad(returns: pd.Series) -> float:
        """计算收益率与最大回撤比率
        
        Args:
            returns: 收益率序列
        
        Returns:
            ROMAD比率
        """
        total_return = returns.sum()
        max_drawdown, _, _ = BacktestUtils.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else float('-inf')
        
        return total_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """计算胜率
        
        Args:
            trades: 交易记录数据框
        
        Returns:
            胜率
        """
        if len(trades) == 0:
            return 0.0
        
        # 假设trades有一个'profit'列
        if 'profit' in trades.columns:
            winning_trades = trades[trades['profit'] > 0]
            return len(winning_trades) / len(trades)
        else:
            logger.warning("交易记录中没有'profit'列，无法计算胜率")
            return 0.0
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """计算盈利因子
        
        Args:
            trades: 交易记录数据框
        
        Returns:
            盈利因子
        """
        if len(trades) == 0:
            return 0.0
        
        # 假设trades有一个'profit'列
        if 'profit' in trades.columns:
            winning_trades = trades[trades['profit'] > 0]
            losing_trades = trades[trades['profit'] <= 0]
            
            total_win = winning_trades['profit'].sum()
            total_loss = abs(losing_trades['profit'].sum())
            
            if total_loss == 0:
                return float('inf') if total_win > 0 else 0.0
            
            return total_win / total_loss
        else:
            logger.warning("交易记录中没有'profit'列，无法计算盈利因子")
            return 0.0
    
    @staticmethod
    def resample_data(data: pd.DataFrame, 
                     frequency: str = 'D') -> pd.DataFrame:
        """重采样数据
        
        Args:
            data: 原始数据
            frequency: 重采样频率
        
        Returns:
            重采样后的数据
        """
        try:
            # 确保索引是datetime类型
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    logger.warning("无法将索引转换为datetime类型")
                    return data
            
            # 重采样
            # 对于OHLC数据，使用相应的聚合函数
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                resampled = data.resample(frequency).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'  # 如果有成交量列
                })
            else:
                # 对于其他数据，使用最后一个值
                resampled = data.resample(frequency).last()
            
            return resampled
        except Exception as e:
            logger.error(f"重采样数据时发生异常: {str(e)}")
            return data
    
    @staticmethod
    def fill_missing_values(data: pd.DataFrame, 
                           method: str = 'forward') -> pd.DataFrame:
        """填充缺失值
        
        Args:
            data: 原始数据
            method: 填充方法 ('forward', 'backward', 'mean', 'median', 'interpolate')
        
        Returns:
            填充后的数据
        """
        try:
            data_filled = data.copy()
            
            if method == 'forward':
                data_filled = data_filled.ffill()
            elif method == 'backward':
                data_filled = data_filled.bfill()
            elif method == 'mean':
                data_filled = data_filled.fillna(data_filled.mean())
            elif method == 'median':
                data_filled = data_filled.fillna(data_filled.median())
            elif method == 'interpolate':
                data_filled = data_filled.interpolate()
            else:
                logger.warning(f"不支持的填充方法: {method}")
                return data
            
            # 填充剩余的缺失值（如果有的话）
            data_filled = data_filled.fillna(0)
            
            return data_filled
        except Exception as e:
            logger.error(f"填充缺失值时发生异常: {str(e)}")
            return data
    
    @staticmethod
    def detect_and_remove_outliers(data: pd.DataFrame, 
                                  columns: Optional[List[str]] = None, 
                                  method: str = 'zscore', 
                                  threshold: float = 3.0) -> pd.DataFrame:
        """检测并移除异常值
        
        Args:
            data: 原始数据
            columns: 要检测异常值的列
            method: 检测方法 ('zscore', 'iqr')
            threshold: 阈值
        
        Returns:
            移除异常值后的数据
        """
        try:
            data_clean = data.copy()
            
            if columns is None:
                # 默认处理所有数值列
                columns = data.select_dtypes(include=['number']).columns
            
            for col in columns:
                if col not in data_clean.columns:
                    continue
                
                if method == 'zscore':
                    # 使用Z-score方法
                    z_scores = np.abs(stats.zscore(data_clean[col].dropna()))
                    outliers = z_scores > threshold
                    data_clean.loc[outliers.index[outliers], col] = np.nan
                elif method == 'iqr':
                    # 使用IQR方法
                    Q1 = data_clean[col].quantile(0.25)
                    Q3 = data_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    data_clean.loc[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound), col] = np.nan
                else:
                    logger.warning(f"不支持的异常值检测方法: {method}")
                    continue
            
            # 填充移除异常值后的缺失值
            data_clean = BacktestUtils.fill_missing_values(data_clean)
            
            return data_clean
        except Exception as e:
            logger.error(f"检测并移除异常值时发生异常: {str(e)}")
            return data
    
    @staticmethod
    def generate_equity_curve(returns: pd.Series, 
                             initial_capital: float = 100000.0) -> pd.Series:
        """生成权益曲线
        
        Args:
            returns: 收益率序列
            initial_capital: 初始资金
        
        Returns:
            权益曲线
        """
        # 计算累积收益
        cumulative_returns = (1 + returns).cumprod()
        # 生成权益曲线
        equity_curve = initial_capital * cumulative_returns
        return equity_curve
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.Series, 
                         title: str = 'Equity Curve',
                         save_path: Optional[str] = None) -> plt.Figure:
        """绘制权益曲线
        
        Args:
            equity_curve: 权益曲线
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制权益曲线
            ax.plot(equity_curve.index, equity_curve, 'b-', linewidth=2)
            
            # 设置标题和标签
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            
            # 添加网格
            ax.grid(True)
            
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
    
    @staticmethod
    def plot_drawdown(drawdown: pd.Series, 
                     title: str = 'Drawdown',
                     save_path: Optional[str] = None) -> plt.Figure:
        """绘制回撤曲线
        
        Args:
            drawdown: 回撤序列
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 将回撤转换为百分比
            drawdown_pct = drawdown * 100
            
            # 绘制回撤曲线
            ax.fill_between(drawdown_pct.index, drawdown_pct, 0, where=drawdown_pct < 0, color='red', alpha=0.3)
            ax.plot(drawdown_pct.index, drawdown_pct, 'r-', linewidth=1)
            
            # 添加零基准线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 设置标题和标签
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            
            # 添加网格
            ax.grid(True)
            
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
    
    @staticmethod
    def plot_monthly_heatmap(returns: pd.Series, 
                           title: str = 'Monthly Returns Heatmap',
                           save_path: Optional[str] = None) -> plt.Figure:
        """绘制月度收益热力图
        
        Args:
            returns: 收益率序列
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            # 确保索引是datetime类型
            if not isinstance(returns.index, pd.DatetimeIndex):
                try:
                    returns.index = pd.to_datetime(returns.index)
                except:
                    logger.warning("无法将索引转换为datetime类型")
                    return None
            
            # 计算月度收益
            monthly_returns = returns.resample('M').sum() * 100  # 转换为百分比
            
            # 创建透视表
            monthly_data = monthly_returns.to_frame(name='Return')
            monthly_data['Year'] = monthly_data.index.year
            monthly_data['Month'] = monthly_data.index.month
            
            # 创建透视表
            heatmap_data = monthly_data.pivot(index='Year', columns='Month', values='Return')
            
            # 重命名列
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = month_names
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制热力图
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                        square=True, ax=ax, cbar_kws={'label': 'Return (%)'})
            
            # 设置标题
            ax.set_title(title)
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"月度收益热力图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制月度收益热力图时发生异常: {str(e)}")
            return None
    
    @staticmethod
    def parallelize(func: Callable, 
                   data: List[Any], 
                   num_jobs: int = -1) -> List[Any]:
        """并行执行函数
        
        Args:
            func: 要执行的函数
            data: 输入数据列表
            num_jobs: 并行任务数量 (-1 表示使用所有可用CPU)
        
        Returns:
            结果列表
        """
        try:
            # 限制工作进程数量
            if num_jobs < 0:
                num_jobs = mp.cpu_count()
            
            # 使用joblib进行并行处理
            with Parallel(n_jobs=num_jobs) as parallel:
                results = parallel(delayed(func)(item) for item in data)
            
            return results
        except Exception as e:
            logger.error(f"并行执行函数时发生异常: {str(e)}")
            # 出错时回退到串行执行
            return [func(item) for item in data]
    
    @staticmethod
    def timer(func: Callable) -> Callable:
        """装饰器：计时函数执行时间
        
        Args:
            func: 要计时的函数
        
        Returns:
            包装后的函数
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
            return result
        return wrapper
    
    @staticmethod
    def catch_exceptions(func: Callable) -> Callable:
        """装饰器：捕获函数异常
        
        Args:
            func: 要包装的函数
        
        Returns:
            包装后的函数
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行异常: {str(e)}")
                logger.debug(traceback.format_exc())
                return None
        return wrapper
    
    @staticmethod
    def generate_random_series(length: int, 
                             mean: float = 0.0, 
                             std: float = 0.01, 
                             distribution: str = 'normal') -> pd.Series:
        """生成随机序列
        
        Args:
            length: 序列长度
            mean: 均值
            std: 标准差
            distribution: 分布类型 ('normal', 'uniform', 'exponential')
        
        Returns:
            随机序列
        """
        try:
            if distribution == 'normal':
                data = np.random.normal(mean, std, length)
            elif distribution == 'uniform':
                data = np.random.uniform(mean - std, mean + std, length)
            elif distribution == 'exponential':
                data = np.random.exponential(mean, length)
            else:
                logger.warning(f"不支持的分布类型: {distribution}")
                data = np.random.normal(mean, std, length)
            
            # 创建时间索引
            date_index = pd.date_range(start=datetime.now(), periods=length)
            
            return pd.Series(data, index=date_index)
        except Exception as e:
            logger.error(f"生成随机序列时发生异常: {str(e)}")
            return pd.Series()
    
    @staticmethod
    def calculate_portfolio_metrics(returns: pd.DataFrame, 
                                  weights: Optional[List[float]] = None, 
                                  risk_free_rate: float = 0.0) -> Dict[str, float]:
        """计算投资组合指标
        
        Args:
            returns: 资产收益率数据框
            weights: 资产权重
            risk_free_rate: 无风险利率
        
        Returns:
            投资组合指标字典
        """
        try:
            # 如果没有提供权重，使用等权重
            if weights is None:
                weights = [1/len(returns.columns)] * len(returns.columns)
            
            # 计算投资组合收益率
            portfolio_returns = returns.dot(weights)
            
            # 计算指标
            metrics = {
                'total_return': portfolio_returns.sum(),
                'mean_daily_return': portfolio_returns.mean(),
                'std_daily_return': portfolio_returns.std(),
                'sharpe_ratio': BacktestUtils.calculate_sharpe_ratio(portfolio_returns, risk_free_rate),
                'sortino_ratio': BacktestUtils.calculate_sortino_ratio(portfolio_returns, risk_free_rate),
            }
            
            # 计算最大回撤
            max_drawdown, _, _ = BacktestUtils.calculate_max_drawdown(portfolio_returns)
            metrics['max_drawdown'] = max_drawdown
            
            # 计算ROMAD
            metrics['romad'] = BacktestUtils.calculate_romad(portfolio_returns)
            
            return metrics
        except Exception as e:
            logger.error(f"计算投资组合指标时发生异常: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, 
                                window: int = 252, 
                                metrics: List[str] = None) -> pd.DataFrame:
        """计算滚动指标
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            metrics: 要计算的指标列表
        
        Returns:
            滚动指标数据框
        """
        try:
            # 默认指标
            if metrics is None:
                metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown']
            
            # 创建结果数据框
            rolling_metrics_df = pd.DataFrame(index=returns.index)
            
            # 计算每个滚动指标
            for metric in metrics:
                if metric == 'sharpe_ratio':
                    # 计算滚动夏普比率
                    rolling_mean = returns.rolling(window=window).mean()
                    rolling_std = returns.rolling(window=window).std()
                    rolling_sharpe = rolling_mean / rolling_std * math.sqrt(252)  # 年化
                    rolling_metrics_df['rolling_sharpe_ratio'] = rolling_sharpe
                elif metric == 'sortino_ratio':
                    # 计算滚动索提诺比率
                    def calculate_rolling_sortino(window_data):
                        downside_returns = window_data[window_data < 0]
                        if len(downside_returns) == 0 or downside_returns.std() == 0:
                            return np.nan
                        return window_data.mean() / downside_returns.std() * math.sqrt(252)  # 年化
                    
                    rolling_sortino = returns.rolling(window=window).apply(calculate_rolling_sortino)
                    rolling_metrics_df['rolling_sortino_ratio'] = rolling_sortino
                elif metric == 'max_drawdown':
                    # 计算滚动最大回撤
                    def calculate_rolling_max_drawdown(window_data):
                        cumulative = (1 + window_data).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative / running_max) - 1
                        return drawdown.min()
                    
                    rolling_max_dd = returns.rolling(window=window).apply(calculate_rolling_max_drawdown)
                    rolling_metrics_df['rolling_max_drawdown'] = rolling_max_dd
            
            return rolling_metrics_df
        except Exception as e:
            logger.error(f"计算滚动指标时发生异常: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def save_results(results: Dict[str, Any], 
                    save_path: str) -> bool:
        """保存结果
        
        Args:
            results: 结果字典
            save_path: 保存路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 转换不支持JSON序列化的数据类型
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    # 将DataFrame/Series转换为字典
                    if isinstance(value, pd.DataFrame):
                        # 保存为CSV文件
                        csv_path = save_path.replace('.json', f'_{key}.csv')
                        value.to_csv(csv_path)
                        serializable_results[key] = csv_path
                    else:
                        # 对于Series，保存其值和索引
                        serializable_results[key] = {
                            'values': value.tolist(),
                            'index': value.index.tolist()
                        }
                elif isinstance(value, np.ndarray):
                    # 将ndarray转换为列表
                    serializable_results[key] = value.tolist()
                elif isinstance(value, plt.Figure):
                    # 保存图表
                    fig_path = save_path.replace('.json', f'_{key}.png')
                    value.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(value)  # 关闭图表以释放内存
                    serializable_results[key] = fig_path
                else:
                    # 尝试直接保存
                    try:
                        json.dumps(value)
                        serializable_results[key] = value
                    except:
                        # 如果无法序列化，保存其字符串表示
                        serializable_results[key] = str(value)
            
            # 保存为JSON文件
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)
            
            logger.info(f"结果已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存结果时发生异常: {str(e)}")
            return False
    
    @staticmethod
    def load_results(load_path: str) -> Dict[str, Any]:
        """加载结果
        
        Args:
            load_path: 加载路径
        
        Returns:
            结果字典
        """
        try:
            if not os.path.exists(load_path):
                logger.error(f"结果文件不存在: {load_path}")
                return {}
            
            # 从文件加载
            with open(load_path, 'r') as f:
                results = json.load(f)
            
            # 尝试加载引用的CSV文件和图表
            for key, value in results.items():
                if isinstance(value, str) and value.endswith('.csv') and os.path.exists(value):
                    # 加载CSV文件为DataFrame
                    results[key] = pd.read_csv(value, index_col=0)
                    # 如果索引看起来像日期，尝试转换
                    try:
                        results[key].index = pd.to_datetime(results[key].index)
                    except:
                        pass
                elif isinstance(value, dict) and 'values' in value and 'index' in value:
                    # 重新构建Series
                    results[key] = pd.Series(value['values'], index=value['index'])
                    # 如果索引看起来像日期，尝试转换
                    try:
                        results[key].index = pd.to_datetime(results[key].index)
                    except:
                        pass
            
            logger.info(f"结果已从: {load_path} 加载")
            return results
        except Exception as e:
            logger.error(f"加载结果时发生异常: {str(e)}")
            return {}
    
    @staticmethod
    def create_directory_if_not_exists(directory: str) -> bool:
        """如果目录不存在则创建
        
        Args:
            directory: 目录路径
        
        Returns:
            是否创建成功
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"目录已创建: {directory}")
            return True
        except Exception as e:
            logger.error(f"创建目录时发生异常: {str(e)}")
            return False
    
    @staticmethod
    def get_file_paths(directory: str, 
                     extensions: Optional[List[str]] = None) -> List[str]:
        """获取目录下所有文件的路径
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名列表（可选）
        
        Returns:
            文件路径列表
        """
        try:
            file_paths = []
            
            for root, _, files in os.walk(directory):
                for file in files:
                    if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                        file_paths.append(os.path.join(root, file))
            
            return file_paths
        except Exception as e:
            logger.error(f"获取文件路径时发生异常: {str(e)}")
            return []
    
    @staticmethod
    def get_class_attributes(class_obj: Any) -> List[str]:
        """获取类的属性列表
        
        Args:
            class_obj: 类对象
        
        Returns:
            属性列表
        """
        try:
            # 获取类的属性，但排除方法和私有属性
            attributes = [attr for attr in dir(class_obj) 
                         if not attr.startswith('__') 
                         and not inspect.ismethod(getattr(class_obj, attr)) 
                         and not inspect.isfunction(getattr(class_obj, attr))]
            return attributes
        except Exception as e:
            logger.error(f"获取类属性时发生异常: {str(e)}")
            return []

# 导入stats模块（需要在函数内部导入以避免循环依赖）
from scipy import stats

# 模块版本
__version__ = '0.1.0'