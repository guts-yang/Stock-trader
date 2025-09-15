import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any, Set
import os
import abc
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Strategy(abc.ABC):
    """交易策略抽象基类
    所有交易策略都应该继承这个基类
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化交易策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.symbols: Set[str] = set()
        self.positions: Dict[str, float] = {}
        self.signals: Dict[str, Dict[datetime, float]] = {}
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info(f"{self.name} 策略初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"strategy_{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    @abc.abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        pass
    
    def add_symbol(self, symbol: str) -> None:
        """添加要交易的符号
        
        Args:
            symbol: 交易符号
        """
        self.symbols.add(symbol)
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        if symbol not in self.signals:
            self.signals[symbol] = {}
        
        logger.info(f"已添加交易符号: {symbol}")
    
    def remove_symbol(self, symbol: str) -> None:
        """移除交易符号
        
        Args:
            symbol: 交易符号
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
        if symbol in self.positions:
            del self.positions[symbol]
        if symbol in self.signals:
            del self.signals[symbol]
        
        logger.info(f"已移除交易符号: {symbol}")
    
    def get_symbols(self) -> List[str]:
        """获取所有交易符号
        
        Returns:
            交易符号列表
        """
        return list(self.symbols)
    
    def set_position(self, symbol: str, position: float) -> None:
        """设置持仓
        
        Args:
            symbol: 交易符号
            position: 持仓数量
        """
        if symbol not in self.symbols:
            self.add_symbol(symbol)
        
        self.positions[symbol] = position
        
        logger.debug(f"已设置 {symbol} 持仓为: {position}")
    
    def get_position(self, symbol: str) -> float:
        """获取持仓
        
        Args:
            symbol: 交易符号
        
        Returns:
            持仓数量
        """
        return self.positions.get(symbol, 0.0)
    
    def update_signals(self, symbol: str, timestamp: datetime, signal: float) -> None:
        """更新信号
        
        Args:
            symbol: 交易符号
            timestamp: 时间戳
            signal: 信号值
        """
        if symbol not in self.signals:
            self.signals[symbol] = {}
        
        self.signals[symbol][timestamp] = signal
    
    def get_signals(self, symbol: str) -> Dict[datetime, float]:
        """获取信号
        
        Args:
            symbol: 交易符号
        
        Returns:
            信号字典
        """
        return self.signals.get(symbol, {})
    
    def reset(self) -> None:
        """重置策略状态"""
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.signals = {symbol: {} for symbol in self.symbols}
        
        logger.info(f"{self.name} 策略已重置")
    
    def save_state(self, file_path: str) -> bool:
        """保存策略状态
        
        Args:
            file_path: 保存文件路径
        
        Returns:
            是否保存成功
        """
        try:
            state = {
                'name': self.name,
                'config': self.config,
                'symbols': list(self.symbols),
                'positions': self.positions,
                'signals': {}
            }
            
            # 转换datetime为字符串以便JSON序列化
            for symbol, sig_dict in self.signals.items():
                state['signals'][symbol] = {ts.strftime('%Y-%m-%d %H:%M:%S'): sig for ts, sig in sig_dict.items()}
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存到文件
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            
            logger.info(f"策略状态已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存策略状态时发生异常: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """加载策略状态
        
        Args:
            file_path: 加载文件路径
        
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"策略状态文件不存在: {file_path}")
                return False
            
            # 从文件加载
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # 恢复状态
            self.config = state.get('config', {})
            self.symbols = set(state.get('symbols', []))
            self.positions = state.get('positions', {})
            
            # 恢复信号，将字符串转换回datetime
            self.signals = {}
            for symbol, sig_dict in state.get('signals', {}).items():
                self.signals[symbol] = {datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S'): sig for ts_str, sig in sig_dict.items()}
            
            logger.info(f"策略状态已从: {file_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载策略状态时发生异常: {str(e)}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置
        
        Returns:
            配置字典
        """
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新策略配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        logger.info(f"策略配置已更新")

class MovingAverageCrossStrategy(Strategy):
    """移动平均线交叉策略
    当短期移动平均线上穿长期移动平均线时买入，下穿时卖出
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化移动平均线交叉策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        # 设置默认配置
        default_config = {
            'short_window': 50,
            'long_window': 200,
            'price_column': 'close',
            'signal_threshold': 0.0
        }
        
        # 合并用户配置和默认配置
        if config:
            default_config.update(config)
        
        super().__init__(default_config, log_level)
        
        self.short_window = self.config['short_window']
        self.long_window = self.config['long_window']
        self.price_column = self.config['price_column']
        self.signal_threshold = self.config['signal_threshold']
        
        logger.info(f"移动平均线交叉策略初始化完成，短期窗口: {self.short_window}，长期窗口: {self.long_window}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        signals = {}
        
        try:
            for symbol, df in data.items():
                if symbol not in self.symbols:
                    continue
                
                # 确保数据包含所需的价格列
                if self.price_column not in df.columns:
                    logger.warning(f"数据中缺少价格列: {self.price_column}")
                    continue
                
                # 计算移动平均线
                df['short_mavg'] = df[self.price_column].rolling(window=self.short_window).mean()
                df['long_mavg'] = df[self.price_column].rolling(window=self.long_window).mean()
                
                # 生成信号
                # 1.0 表示买入信号，-1.0 表示卖出信号
                df['signal'] = 0.0
                df['signal'][self.short_window:] = np.where(
                    df['short_mavg'][self.short_window:] > df['long_mavg'][self.short_window:], 1.0, 0.0
                )
                
                # 计算交易订单
                df['positions'] = df['signal'].diff()
                
                # 收集交易信号
                symbol_signals = []
                for timestamp, row in df.iterrows():
                    if row['positions'] == 1.0:
                        # 买入信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'short_mavg': row['short_mavg'],
                            'long_mavg': row['long_mavg']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, 1.0)
                    elif row['positions'] == -1.0:
                        # 卖出信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'short_mavg': row['short_mavg'],
                            'long_mavg': row['long_mavg']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, -1.0)
                
                signals[symbol] = symbol_signals
                
                logger.info(f"为 {symbol} 生成了 {len(symbol_signals)} 个交易信号")
            
            return signals
        except Exception as e:
            logger.error(f"生成信号时发生异常: {str(e)}")
            return signals
    
    def plot_signals(self, data: Dict[str, pd.DataFrame], symbol: str, output_path: Optional[str] = None) -> None:
        """绘制策略信号图
        
        Args:
            data: 包含符号数据的字典
            symbol: 要绘制的符号
            output_path: 图像保存路径
        """
        try:
            if symbol not in data:
                logger.error(f"没有找到 {symbol} 的数据")
                return
            
            df = data[symbol].copy()
            
            # 计算移动平均线
            df['short_mavg'] = df[self.price_column].rolling(window=self.short_window).mean()
            df['long_mavg'] = df[self.price_column].rolling(window=self.long_window).mean()
            
            # 生成信号
            df['signal'] = 0.0
            df['signal'][self.short_window:] = np.where(
                df['short_mavg'][self.short_window:] > df['long_mavg'][self.short_window:], 1.0, 0.0
            )
            df['positions'] = df['signal'].diff()
            
            # 创建图形
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # 绘制价格和移动平均线
            ax1.plot(df.index, df[self.price_column], label='价格')
            ax1.plot(df.index, df['short_mavg'], label=f'{self.short_window}日移动平均')
            ax1.plot(df.index, df['long_mavg'], label=f'{self.long_window}日移动平均')
            
            # 标记买入信号
            buy_signals = df[df['positions'] == 1.0]
            ax1.scatter(buy_signals.index, buy_signals[self.price_column], 
                       marker='^', color='g', label='买入信号', alpha=1)
            
            # 标记卖出信号
            sell_signals = df[df['positions'] == -1.0]
            ax1.scatter(sell_signals.index, sell_signals[self.price_column], 
                       marker='v', color='r', label='卖出信号', alpha=1)
            
            # 设置图形属性
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.set_title(f'{symbol} - 移动平均线交叉策略信号')
            ax1.legend()
            ax1.grid(True)
            
            # 自动调整日期标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存或显示图形
            if output_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger.info(f"策略信号图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"绘制策略信号图时发生异常: {str(e)}")

class RSIStrategy(Strategy):
    """RSI策略
    基于相对强弱指标(RSI)生成交易信号
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化RSI策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        # 设置默认配置
        default_config = {
            'rsi_period': 14,
            'overbought_threshold': 70,
            'oversold_threshold': 30,
            'price_column': 'close'
        }
        
        # 合并用户配置和默认配置
        if config:
            default_config.update(config)
        
        super().__init__(default_config, log_level)
        
        self.rsi_period = self.config['rsi_period']
        self.overbought_threshold = self.config['overbought_threshold']
        self.oversold_threshold = self.config['oversold_threshold']
        self.price_column = self.config['price_column']
        
        logger.info(f"RSI策略初始化完成，周期: {self.rsi_period}，超买阈值: {self.overbought_threshold}，超卖阈值: {self.oversold_threshold}")
    
    def calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算RSI指标
        
        Args:
            df: 价格数据
            period: RSI计算周期
        
        Returns:
            包含RSI指标的数据
        """
        # 复制数据以避免修改原始数据
        result_df = df.copy()
        
        # 计算价格变化
        delta = result_df[self.price_column].diff()
        
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 计算RS和RSI
        rs = gain / loss
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        return result_df
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        signals = {}
        
        try:
            for symbol, df in data.items():
                if symbol not in self.symbols:
                    continue
                
                # 确保数据包含所需的价格列
                if self.price_column not in df.columns:
                    logger.warning(f"数据中缺少价格列: {self.price_column}")
                    continue
                
                # 计算RSI
                df = self.calculate_rsi(df, self.rsi_period)
                
                # 初始化信号列
                df['signal'] = 0.0
                
                # 生成信号
                # 当RSI从下方穿过超卖阈值时买入
                # 当RSI从上方穿过超买阈值时卖出
                df['signal'][self.rsi_period:] = np.where(
                    (df['rsi'][self.rsi_period:] <= self.oversold_threshold) & 
                    (df['rsi'].shift(1)[self.rsi_period:] > self.oversold_threshold),
                    1.0, 0.0
                )
                df['signal'][self.rsi_period:] = np.where(
                    (df['rsi'][self.rsi_period:] >= self.overbought_threshold) & 
                    (df['rsi'].shift(1)[self.rsi_period:] < self.overbought_threshold),
                    -1.0, df['signal'][self.rsi_period:]
                )
                
                # 收集交易信号
                symbol_signals = []
                for timestamp, row in df.iterrows():
                    if row['signal'] == 1.0:
                        # 买入信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'rsi': row['rsi']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, 1.0)
                    elif row['signal'] == -1.0:
                        # 卖出信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'rsi': row['rsi']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, -1.0)
                
                signals[symbol] = symbol_signals
                
                logger.info(f"为 {symbol} 生成了 {len(symbol_signals)} 个交易信号")
            
            return signals
        except Exception as e:
            logger.error(f"生成信号时发生异常: {str(e)}")
            return signals
    
    def plot_signals(self, data: Dict[str, pd.DataFrame], symbol: str, output_path: Optional[str] = None) -> None:
        """绘制策略信号图
        
        Args:
            data: 包含符号数据的字典
            symbol: 要绘制的符号
            output_path: 图像保存路径
        """
        try:
            if symbol not in data:
                logger.error(f"没有找到 {symbol} 的数据")
                return
            
            df = data[symbol].copy()
            
            # 计算RSI
            df = self.calculate_rsi(df, self.rsi_period)
            
            # 生成信号
            df['signal'] = 0.0
            df['signal'][self.rsi_period:] = np.where(
                (df['rsi'][self.rsi_period:] <= self.oversold_threshold) & 
                (df['rsi'].shift(1)[self.rsi_period:] > self.oversold_threshold),
                1.0, 0.0
            )
            df['signal'][self.rsi_period:] = np.where(
                (df['rsi'][self.rsi_period:] >= self.overbought_threshold) & 
                (df['rsi'].shift(1)[self.rsi_period:] < self.overbought_threshold),
                -1.0, df['signal'][self.rsi_period:]
            )
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制价格
            ax1.plot(df.index, df[self.price_column], label='价格')
            
            # 标记买入信号
            buy_signals = df[df['signal'] == 1.0]
            ax1.scatter(buy_signals.index, buy_signals[self.price_column], 
                       marker='^', color='g', label='买入信号', alpha=1)
            
            # 标记卖出信号
            sell_signals = df[df['signal'] == -1.0]
            ax1.scatter(sell_signals.index, sell_signals[self.price_column], 
                       marker='v', color='r', label='卖出信号', alpha=1)
            
            # 设置第一个子图属性
            ax1.set_ylabel('价格')
            ax1.set_title(f'{symbol} - RSI策略信号')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制RSI
            ax2.plot(df.index, df['rsi'], label=f'RSI ({self.rsi_period})')
            ax2.axhline(y=self.overbought_threshold, color='r', linestyle='--', label='超买阈值')
            ax2.axhline(y=self.oversold_threshold, color='g', linestyle='--', label='超卖阈值')
            
            # 设置第二个子图属性
            ax2.set_xlabel('日期')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True)
            
            # 自动调整日期标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存或显示图形
            if output_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger.info(f"策略信号图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"绘制策略信号图时发生异常: {str(e)}")

class MACDStrategy(Strategy):
    """MACD策略
    基于移动平均线收敛发散指标(MACD)生成交易信号
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化MACD策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        # 设置默认配置
        default_config = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'price_column': 'close'
        }
        
        # 合并用户配置和默认配置
        if config:
            default_config.update(config)
        
        super().__init__(default_config, log_level)
        
        self.fast_period = self.config['fast_period']
        self.slow_period = self.config['slow_period']
        self.signal_period = self.config['signal_period']
        self.price_column = self.config['price_column']
        
        logger.info(f"MACD策略初始化完成，快线周期: {self.fast_period}，慢线周期: {self.slow_period}，信号线周期: {self.signal_period}")
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 价格数据
        
        Returns:
            包含MACD指标的数据
        """
        # 复制数据以避免修改原始数据
        result_df = df.copy()
        
        # 计算快速和慢速移动平均线
        result_df['ema_fast'] = result_df[self.price_column].ewm(span=self.fast_period, adjust=False).mean()
        result_df['ema_slow'] = result_df[self.price_column].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算MACD线
        result_df['macd_line'] = result_df['ema_fast'] - result_df['ema_slow']
        
        # 计算信号线
        result_df['signal_line'] = result_df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        result_df['macd_hist'] = result_df['macd_line'] - result_df['signal_line']
        
        return result_df
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        signals = {}
        
        try:
            for symbol, df in data.items():
                if symbol not in self.symbols:
                    continue
                
                # 确保数据包含所需的价格列
                if self.price_column not in df.columns:
                    logger.warning(f"数据中缺少价格列: {self.price_column}")
                    continue
                
                # 计算MACD指标
                df = self.calculate_macd(df)
                
                # 初始化信号列
                df['signal'] = 0.0
                
                # 生成信号
                # 当MACD线上穿信号线时买入
                # 当MACD线下穿信号线时卖出
                df['signal'][self.slow_period:] = np.where(
                    (df['macd_line'][self.slow_period:] > df['signal_line'][self.slow_period:]) & 
                    (df['macd_line'].shift(1)[self.slow_period:] <= df['signal_line'].shift(1)[self.slow_period:]),
                    1.0, 0.0
                )
                df['signal'][self.slow_period:] = np.where(
                    (df['macd_line'][self.slow_period:] < df['signal_line'][self.slow_period:]) & 
                    (df['macd_line'].shift(1)[self.slow_period:] >= df['signal_line'].shift(1)[self.slow_period:]),
                    -1.0, df['signal'][self.slow_period:]
                )
                
                # 收集交易信号
                symbol_signals = []
                for timestamp, row in df.iterrows():
                    if row['signal'] == 1.0:
                        # 买入信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'macd_line': row['macd_line'],
                            'signal_line': row['signal_line'],
                            'macd_hist': row['macd_hist']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, 1.0)
                    elif row['signal'] == -1.0:
                        # 卖出信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'macd_line': row['macd_line'],
                            'signal_line': row['signal_line'],
                            'macd_hist': row['macd_hist']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, -1.0)
                
                signals[symbol] = symbol_signals
                
                logger.info(f"为 {symbol} 生成了 {len(symbol_signals)} 个交易信号")
            
            return signals
        except Exception as e:
            logger.error(f"生成信号时发生异常: {str(e)}")
            return signals
    
    def plot_signals(self, data: Dict[str, pd.DataFrame], symbol: str, output_path: Optional[str] = None) -> None:
        """绘制策略信号图
        
        Args:
            data: 包含符号数据的字典
            symbol: 要绘制的符号
            output_path: 图像保存路径
        """
        try:
            if symbol not in data:
                logger.error(f"没有找到 {symbol} 的数据")
                return
            
            df = data[symbol].copy()
            
            # 计算MACD指标
            df = self.calculate_macd(df)
            
            # 生成信号
            df['signal'] = 0.0
            df['signal'][self.slow_period:] = np.where(
                (df['macd_line'][self.slow_period:] > df['signal_line'][self.slow_period:]) & 
                (df['macd_line'].shift(1)[self.slow_period:] <= df['signal_line'].shift(1)[self.slow_period:]),
                1.0, 0.0
            )
            df['signal'][self.slow_period:] = np.where(
                (df['macd_line'][self.slow_period:] < df['signal_line'][self.slow_period:]) & 
                (df['macd_line'].shift(1)[self.slow_period:] >= df['signal_line'].shift(1)[self.slow_period:]),
                -1.0, df['signal'][self.slow_period:]
            )
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})
            
            # 绘制价格
            ax1.plot(df.index, df[self.price_column], label='价格')
            
            # 标记买入信号
            buy_signals = df[df['signal'] == 1.0]
            ax1.scatter(buy_signals.index, buy_signals[self.price_column], 
                       marker='^', color='g', label='买入信号', alpha=1)
            
            # 标记卖出信号
            sell_signals = df[df['signal'] == -1.0]
            ax1.scatter(sell_signals.index, sell_signals[self.price_column], 
                       marker='v', color='r', label='卖出信号', alpha=1)
            
            # 设置第一个子图属性
            ax1.set_ylabel('价格')
            ax1.set_title(f'{symbol} - MACD策略信号')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制MACD线和信号线
            ax2.plot(df.index, df['macd_line'], label='MACD线')
            ax2.plot(df.index, df['signal_line'], label='信号线')
            
            # 绘制MACD柱状图
            ax2.bar(df.index, df['macd_hist'], label='MACD柱状图', alpha=0.5)
            
            # 设置第二个子图属性
            ax2.set_xlabel('日期')
            ax2.set_ylabel('MACD值')
            ax2.legend()
            ax2.grid(True)
            
            # 自动调整日期标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存或显示图形
            if output_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger.info(f"策略信号图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"绘制策略信号图时发生异常: {str(e)}")

class BollingerBandsStrategy(Strategy):
    """布林带策略
    基于布林带指标生成交易信号
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化布林带策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        # 设置默认配置
        default_config = {
            'period': 20,
            'std_dev': 2,
            'price_column': 'close'
        }
        
        # 合并用户配置和默认配置
        if config:
            default_config.update(config)
        
        super().__init__(default_config, log_level)
        
        self.period = self.config['period']
        self.std_dev = self.config['std_dev']
        self.price_column = self.config['price_column']
        
        logger.info(f"布林带策略初始化完成，周期: {self.period}，标准差倍数: {self.std_dev}")
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标
        
        Args:
            df: 价格数据
        
        Returns:
            包含布林带指标的数据
        """
        # 复制数据以避免修改原始数据
        result_df = df.copy()
        
        # 计算移动平均线（中轨）
        result_df['middle_band'] = result_df[self.price_column].rolling(window=self.period).mean()
        
        # 计算标准差
        result_df['std_dev'] = result_df[self.price_column].rolling(window=self.period).std()
        
        # 计算上轨和下轨
        result_df['upper_band'] = result_df['middle_band'] + (result_df['std_dev'] * self.std_dev)
        result_df['lower_band'] = result_df['middle_band'] - (result_df['std_dev'] * self.std_dev)
        
        # 计算价格相对于布林带的位置
        result_df['position'] = (result_df[self.price_column] - result_df['middle_band']) / (self.std_dev * result_df['std_dev'])
        
        return result_df
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        signals = {}
        
        try:
            for symbol, df in data.items():
                if symbol not in self.symbols:
                    continue
                
                # 确保数据包含所需的价格列
                if self.price_column not in df.columns:
                    logger.warning(f"数据中缺少价格列: {self.price_column}")
                    continue
                
                # 计算布林带指标
                df = self.calculate_bollinger_bands(df)
                
                # 初始化信号列
                df['signal'] = 0.0
                
                # 生成信号
                # 当价格从下方穿过下轨时买入
                # 当价格从上方穿过上轨时卖出
                df['signal'][self.period:] = np.where(
                    (df[self.price_column][self.period:] <= df['lower_band'][self.period:]) & 
                    (df[self.price_column].shift(1)[self.period:] > df['lower_band'].shift(1)[self.period:]),
                    1.0, 0.0
                )
                df['signal'][self.period:] = np.where(
                    (df[self.price_column][self.period:] >= df['upper_band'][self.period:]) & 
                    (df[self.price_column].shift(1)[self.period:] < df['upper_band'].shift(1)[self.period:]),
                    -1.0, df['signal'][self.period:]
                )
                
                # 收集交易信号
                symbol_signals = []
                for timestamp, row in df.iterrows():
                    if row['signal'] == 1.0:
                        # 买入信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'middle_band': row['middle_band'],
                            'upper_band': row['upper_band'],
                            'lower_band': row['lower_band'],
                            'position': row['position']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, 1.0)
                    elif row['signal'] == -1.0:
                        # 卖出信号
                        signal = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'strength': 1.0,
                            'price': row[self.price_column],
                            'middle_band': row['middle_band'],
                            'upper_band': row['upper_band'],
                            'lower_band': row['lower_band'],
                            'position': row['position']
                        }
                        symbol_signals.append(signal)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, -1.0)
                
                signals[symbol] = symbol_signals
                
                logger.info(f"为 {symbol} 生成了 {len(symbol_signals)} 个交易信号")
            
            return signals
        except Exception as e:
            logger.error(f"生成信号时发生异常: {str(e)}")
            return signals
    
    def plot_signals(self, data: Dict[str, pd.DataFrame], symbol: str, output_path: Optional[str] = None) -> None:
        """绘制策略信号图
        
        Args:
            data: 包含符号数据的字典
            symbol: 要绘制的符号
            output_path: 图像保存路径
        """
        try:
            if symbol not in data:
                logger.error(f"没有找到 {symbol} 的数据")
                return
            
            df = data[symbol].copy()
            
            # 计算布林带指标
            df = self.calculate_bollinger_bands(df)
            
            # 生成信号
            df['signal'] = 0.0
            df['signal'][self.period:] = np.where(
                (df[self.price_column][self.period:] <= df['lower_band'][self.period:]) & 
                (df[self.price_column].shift(1)[self.period:] > df['lower_band'].shift(1)[self.period:]),
                1.0, 0.0
            )
            df['signal'][self.period:] = np.where(
                (df[self.price_column][self.period:] >= df['upper_band'][self.period:]) & 
                (df[self.price_column].shift(1)[self.period:] < df['upper_band'].shift(1)[self.period:]),
                -1.0, df['signal'][self.period:]
            )
            
            # 创建图形
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            # 绘制价格和布林带
            ax1.plot(df.index, df[self.price_column], label='价格')
            ax1.plot(df.index, df['middle_band'], label='中轨', color='blue')
            ax1.plot(df.index, df['upper_band'], label='上轨', color='red', linestyle='--')
            ax1.plot(df.index, df['lower_band'], label='下轨', color='green', linestyle='--')
            
            # 填充布林带区域
            ax1.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.1)
            
            # 标记买入信号
            buy_signals = df[df['signal'] == 1.0]
            ax1.scatter(buy_signals.index, buy_signals[self.price_column], 
                       marker='^', color='g', label='买入信号', alpha=1)
            
            # 标记卖出信号
            sell_signals = df[df['signal'] == -1.0]
            ax1.scatter(sell_signals.index, sell_signals[self.price_column], 
                       marker='v', color='r', label='卖出信号', alpha=1)
            
            # 设置图形属性
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.set_title(f'{symbol} - 布林带策略信号')
            ax1.legend()
            ax1.grid(True)
            
            # 自动调整日期标签
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存或显示图形
            if output_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                logger.info(f"策略信号图已保存到: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"绘制策略信号图时发生异常: {str(e)}")

class MultiIndicatorStrategy(Strategy):
    """多指标策略
    结合多个技术指标生成交易信号
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化多指标策略
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        # 设置默认配置
        default_config = {
            'indicators': ['sma', 'rsi', 'macd'],
            'weights': {'sma': 0.3, 'rsi': 0.3, 'macd': 0.4},
            'threshold': 0.5,
            'sma_config': {'short_window': 50, 'long_window': 200},
            'rsi_config': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd_config': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'price_column': 'close'
        }
        
        # 合并用户配置和默认配置
        if config:
            default_config.update(config)
        
        super().__init__(default_config, log_level)
        
        self.indicators = self.config['indicators']
        self.weights = self.config['weights']
        self.threshold = self.config['threshold']
        self.sma_config = self.config['sma_config']
        self.rsi_config = self.config['rsi_config']
        self.macd_config = self.config['macd_config']
        self.price_column = self.config['price_column']
        
        logger.info(f"多指标策略初始化完成，使用指标: {', '.join(self.indicators)}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标
        
        Args:
            df: 价格数据
        
        Returns:
            包含所有指标的数据
        """
        # 复制数据以避免修改原始数据
        result_df = df.copy()
        
        # 计算SMA指标
        if 'sma' in self.indicators:
            result_df['short_mavg'] = result_df[self.price_column].rolling(window=self.sma_config['short_window']).mean()
            result_df['long_mavg'] = result_df[self.price_column].rolling(window=self.sma_config['long_window']).mean()
            result_df['sma_signal'] = np.where(result_df['short_mavg'] > result_df['long_mavg'], 1.0, -1.0)
        
        # 计算RSI指标
        if 'rsi' in self.indicators:
            delta = result_df[self.price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_config['period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_config['period']).mean()
            rs = gain / loss
            result_df['rsi'] = 100 - (100 / (1 + rs))
            # 标准化RSI值为 -1 到 1
            result_df['rsi_signal'] = np.where(
                result_df['rsi'] > self.rsi_config['overbought'], -1.0, 
                np.where(result_df['rsi'] < self.rsi_config['oversold'], 1.0, 0.0)
            )
        
        # 计算MACD指标
        if 'macd' in self.indicators:
            result_df['ema_fast'] = result_df[self.price_column].ewm(span=self.macd_config['fast_period'], adjust=False).mean()
            result_df['ema_slow'] = result_df[self.price_column].ewm(span=self.macd_config['slow_period'], adjust=False).mean()
            result_df['macd_line'] = result_df['ema_fast'] - result_df['ema_slow']
            result_df['signal_line'] = result_df['macd_line'].ewm(span=self.macd_config['signal_period'], adjust=False).mean()
            result_df['macd_hist'] = result_df['macd_line'] - result_df['signal_line']
            result_df['macd_signal'] = np.where(result_df['macd_line'] > result_df['signal_line'], 1.0, -1.0)
        
        return result_df
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成交易信号
        
        Args:
            data: 包含各个符号数据的字典
        
        Returns:
            交易信号字典
        """
        signals = {}
        
        try:
            for symbol, df in data.items():
                if symbol not in self.symbols:
                    continue
                
                # 确保数据包含所需的价格列
                if self.price_column not in df.columns:
                    logger.warning(f"数据中缺少价格列: {self.price_column}")
                    continue
                
                # 计算所有指标
                df = self.calculate_indicators(df)
                
                # 初始化综合信号
                df['combined_signal'] = 0.0
                
                # 计算各指标的贡献
                for indicator in self.indicators:
                    if indicator in self.weights and f'{indicator}_signal' in df.columns:
                        df['combined_signal'] += df[f'{indicator}_signal'] * self.weights[indicator]
                
                # 初始化交易信号列
                df['signal'] = 0.0
                
                # 根据综合信号生成交易信号
                max_window = max(
                    self.sma_config.get('long_window', 0) if 'sma' in self.indicators else 0,
                    self.rsi_config.get('period', 0) if 'rsi' in self.indicators else 0,
                    self.macd_config.get('slow_period', 0) if 'macd' in self.indicators else 0
                )
                
                df['signal'][max_window:] = np.where(
                    df['combined_signal'][max_window:] > self.threshold, 1.0, 
                    np.where(df['combined_signal'][max_window:] < -self.threshold, -1.0, 0.0)
                )
                
                # 收集交易信号
                symbol_signals = []
                for timestamp, row in df.iterrows():
                    if row['signal'] == 1.0:
                        # 买入信号
                        signal_info = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'strength': row['combined_signal'],
                            'price': row[self.price_column]
                        }
                        # 添加各指标的详细信息
                        if 'sma' in self.indicators:
                            signal_info['sma_short'] = row['short_mavg']
                            signal_info['sma_long'] = row['long_mavg']
                        if 'rsi' in self.indicators:
                            signal_info['rsi'] = row['rsi']
                        if 'macd' in self.indicators:
                            signal_info['macd_line'] = row['macd_line']
                            signal_info['signal_line'] = row['signal_line']
                        
                        symbol_signals.append(signal_info)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, 1.0)
                    elif row['signal'] == -1.0:
                        # 卖出信号
                        signal_info = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'strength': abs(row['combined_signal']),
                            'price': row[self.price_column]
                        }
                        # 添加各指标的详细信息
                        if 'sma' in self.indicators:
                            signal_info['sma_short'] = row['short_mavg']
                            signal_info['sma_long'] = row['long_mavg']
                        if 'rsi' in self.indicators:
                            signal_info['rsi'] = row['rsi']
                        if 'macd' in self.indicators:
                            signal_info['macd_line'] = row['macd_line']
                            signal_info['signal_line'] = row['signal_line']
                        
                        symbol_signals.append(signal_info)
                        # 更新策略信号
                        self.update_signals(symbol, timestamp, -1.0)
                
                signals[symbol] = symbol_signals
                
                logger.info(f"为 {symbol} 生成了 {len(symbol_signals)} 个交易信号")
            
            return signals
        except Exception as e:
            logger.error(f"生成信号时发生异常: {str(e)}")
            return signals

# 模块版本
__version__ = '0.1.0'