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

class Backtester:
    """回测引擎
    提供交易策略的历史回测功能，支持多种回测模式和性能评估指标
    """
    # 回测模式
    MODE_SINGLE_ASSET = 'single_asset'
    MODE_PORTFOLIO = 'portfolio'
    
    # 交易方向
    DIRECTION_LONG = 1
    DIRECTION_SHORT = -1
    DIRECTION_FLAT = 0
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 initial_capital: float = 100000.0,
                 mode: str = MODE_SINGLE_ASSET,
                 transaction_cost_model: Optional[Any] = None,
                 slippage_model: Optional[Any] = None,
                 risk_manager: Optional[Any] = None,
                 log_level: int = logging.INFO):
        """初始化回测引擎
        
        Args:
            config: 配置字典
            initial_capital: 初始资金
            mode: 回测模式 ('single_asset' 或 'portfolio')
            transaction_cost_model: 交易成本模型
            slippage_model: 滑点模型
            risk_manager: 风险管理模块
            log_level: 日志级别
        """
        self.config = config or {}
        self.initial_capital = initial_capital
        self.mode = mode
        self.transaction_cost_model = transaction_cost_model
        self.slippage_model = slippage_model
        self.risk_manager = risk_manager
        
        # 回测结果
        self.equity_curve = None
        self.trades = []
        self.positions = pd.DataFrame()
        self.performance_metrics = {}
        
        # 回测状态
        self.current_date = None
        self.current_capital = initial_capital
        self.current_positions = {}
        self.current_nav = initial_capital
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info(f"Backtester 初始化完成，模式: {mode}，初始资金: {initial_capital}")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"backtester_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def load_data(self, 
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 data_type: str = 'price') -> bool:
        """加载回测数据
        
        Args:
            data: 数据（单个资产为DataFrame，多资产为Dict）
            data_type: 数据类型 ('price', 'returns', 'ohlcv')
        
        Returns:
            是否加载成功
        """
        try:
            # 验证数据格式
            if self.mode == self.MODE_SINGLE_ASSET:
                if not isinstance(data, pd.DataFrame):
                    logger.error("单资产模式下，数据必须是DataFrame类型")
                    return False
                
                # 检查数据是否有日期索引
                if not isinstance(data.index, pd.DatetimeIndex):
                    logger.warning("数据索引不是日期类型，尝试转换")
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        logger.error(f"转换索引为日期类型失败: {str(e)}")
                        return False
                
                self.data = data
            else:  # MODE_PORTFOLIO
                if not isinstance(data, dict):
                    logger.error("投资组合模式下，数据必须是字典类型")
                    return False
                
                # 检查每个资产的数据
                for asset, asset_data in data.items():
                    if not isinstance(asset_data, pd.DataFrame):
                        logger.error(f"资产 {asset} 的数据必须是DataFrame类型")
                        return False
                    
                    # 检查数据是否有日期索引
                    if not isinstance(asset_data.index, pd.DatetimeIndex):
                        logger.warning(f"资产 {asset} 的数据索引不是日期类型，尝试转换")
                        try:
                            data[asset].index = pd.to_datetime(data[asset].index)
                        except Exception as e:
                            logger.error(f"转换资产 {asset} 的索引为日期类型失败: {str(e)}")
                            return False
                
                self.data = data
            
            # 验证数据类型
            if data_type == 'ohlcv':
                # 检查是否包含OHLCV列
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if self.mode == self.MODE_SINGLE_ASSET:
                    for col in required_columns:
                        if col not in self.data.columns:
                            logger.error(f"OHLCV数据缺少必要列: {col}")
                            return False
                else:
                    for asset, asset_data in self.data.items():
                        for col in required_columns:
                            if col not in asset_data.columns:
                                logger.error(f"资产 {asset} 的OHLCV数据缺少必要列: {col}")
                                return False
            
            logger.info(f"数据加载成功，模式: {self.mode}，数据类型: {data_type}")
            return True
        except Exception as e:
            logger.error(f"加载数据时发生异常: {str(e)}")
            return False
    
    def set_strategy(self, 
                    strategy_func: Callable,
                    strategy_params: Optional[Dict[str, Any]] = None) -> bool:
        """设置交易策略
        
        Args:
            strategy_func: 策略函数，接收数据和当前状态，返回交易信号
            strategy_params: 策略参数
        
        Returns:
            是否设置成功
        """
        try:
            # 验证策略函数
            if not callable(strategy_func):
                logger.error("策略必须是可调用的函数")
                return False
            
            self.strategy_func = strategy_func
            self.strategy_params = strategy_params or {}
            
            logger.info("交易策略设置成功")
            return True
        except Exception as e:
            logger.error(f"设置交易策略时发生异常: {str(e)}")
            return False
    
    def _calculate_position_size(self, 
                                signal: Dict[str, int],
                                prices: Dict[str, float],
                                risk_per_trade: float = 0.02) -> Dict[str, float]:
        """计算头寸大小
        
        Args:
            signal: 交易信号
            prices: 当前价格
            risk_per_trade: 每笔交易风险百分比
        
        Returns:
            头寸大小字典
        """
        positions = {}
        
        # 如果有风险管理模块，使用它来计算头寸大小
        if self.risk_manager is not None:
            try:
                # 假设risk_manager有calculate_position_size方法
                positions = self.risk_manager.calculate_position_size(signal, prices, self.current_capital)
            except Exception as e:
                logger.warning(f"使用风险管理模块计算头寸大小时发生异常: {str(e)}，使用默认方法")
                # 回退到默认方法
                pass
        
        # 默认头寸大小计算
        if not positions:
            n_signals = len([s for s in signal.values() if s != 0])
            if n_signals > 0:
                # 等资金分配到有信号的资产
                capital_per_asset = self.current_capital / n_signals
                
                for asset, direction in signal.items():
                    if direction != 0 and prices.get(asset, 0) > 0:
                        # 计算头寸大小（股数或合约数）
                        position_size = (capital_per_asset * risk_per_trade) / prices[asset]
                        # 取整
                        position_size = int(position_size)
                        # 应用交易方向
                        positions[asset] = position_size * direction
        
        return positions
    
    def _calculate_transaction_cost(self, 
                                  asset: str,
                                  quantity: float,
                                  price: float,
                                  direction: int) -> float:
        """计算交易成本
        
        Args:
            asset: 资产名称
            quantity: 交易数量
            price: 交易价格
            direction: 交易方向
        
        Returns:
            交易成本
        """
        # 如果有交易成本模型，使用它来计算
        if self.transaction_cost_model is not None:
            try:
                # 假设transaction_cost_model有calculate_cost方法
                cost = self.transaction_cost_model.calculate_cost(asset, quantity, price, direction)
                return cost
            except Exception as e:
                logger.warning(f"使用交易成本模型计算时发生异常: {str(e)}，使用默认方法")
                # 回退到默认方法
                pass
        
        # 默认交易成本计算（0.1%的交易金额）
        default_cost_rate = 0.001
        cost = abs(quantity) * price * default_cost_rate
        
        return cost
    
    def _calculate_slippage(self, 
                           asset: str,
                           quantity: float,
                           price: float,
                           direction: int) -> float:
        """计算滑点
        
        Args:
            asset: 资产名称
            quantity: 交易数量
            price: 期望价格
            direction: 交易方向
        
        Returns:
            实际交易价格
        """
        # 如果有滑点模型，使用它来计算
        if self.slippage_model is not None:
            try:
                # 假设slippage_model有calculate_slippage方法
                actual_price = self.slippage_model.calculate_slippage(asset, quantity, price, direction)
                return actual_price
            except Exception as e:
                logger.warning(f"使用滑点模型计算时发生异常: {str(e)}，使用默认方法")
                # 回退到默认方法
                pass
        
        # 默认滑点计算（0.05%的价格）
        default_slippage_rate = 0.0005
        slippage = price * default_slippage_rate
        
        # 买入时价格上涨，卖出时价格下跌
        if direction == self.DIRECTION_LONG:
            actual_price = price + slippage
        elif direction == self.DIRECTION_SHORT:
            actual_price = price - slippage
        else:
            actual_price = price
        
        return actual_price
    
    def run_backtest(self, 
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    progress_bar: bool = True) -> bool:
        """运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            progress_bar: 是否显示进度条
        
        Returns:
            是否回测成功
        """
        try:
            logger.info("开始回测")
            
            # 重置回测状态
            self._reset_backtest_state()
            
            # 确定回测时间范围
            if self.mode == self.MODE_SINGLE_ASSET:
                all_dates = self.data.index
            else:
                # 对于投资组合模式，使用所有资产的共同日期
                all_dates = None
                for asset_data in self.data.values():
                    if all_dates is None:
                        all_dates = asset_data.index
                    else:
                        all_dates = all_dates.intersection(asset_data.index)
            
            # 过滤日期范围
            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                all_dates = all_dates[all_dates >= start_date]
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                all_dates = all_dates[all_dates <= end_date]
            
            # 确保有足够的日期进行回测
            if len(all_dates) < 2:
                logger.error("没有足够的日期数据进行回测")
                return False
            
            # 准备权益曲线和仓位记录
            self.equity_curve = pd.DataFrame(index=all_dates, columns=['equity', 'returns'])
            self.equity_curve['equity'].iloc[0] = self.initial_capital
            self.equity_curve['returns'].iloc[0] = 0.0
            
            # 初始化进度条
            if progress_bar:
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(all_dates), desc="回测进度")
                except ImportError:
                    logger.warning("无法导入tqdm库，不显示进度条")
                    pbar = None
            else:
                pbar = None
            
            # 遍历每个交易日进行回测
            for i, date in enumerate(all_dates):
                if i == 0:
                    # 第一个日期，初始化状态
                    self.current_date = date
                    if pbar:
                        pbar.update(1)
                    continue
                
                self.current_date = date
                
                # 获取当前市场数据
                current_data = self._get_current_data(date)
                
                # 获取当前价格
                current_prices = self._get_current_prices(date)
                
                # 计算前一日期的资产价值
                previous_date = all_dates[i-1]
                portfolio_value = self._calculate_portfolio_value(previous_date, current_prices)
                
                # 更新当前资金和净资产
                self.current_capital = portfolio_value
                self.current_nav = portfolio_value
                
                # 记录权益曲线
                self.equity_curve['equity'].iloc[i] = portfolio_value
                self.equity_curve['returns'].iloc[i] = (portfolio_value / self.equity_curve['equity'].iloc[i-1]) - 1
                
                # 执行策略，生成交易信号
                signal = self._generate_signal(current_data, i)
                
                # 如果有交易信号，执行交易
                if signal:
                    self._execute_trades(signal, current_prices)
                
                # 更新头寸记录
                self._update_position_record(date)
                
                # 更新进度条
                if pbar:
                    pbar.update(1)
            
            # 关闭进度条
            if pbar:
                pbar.close()
            
            # 计算回测性能指标
            self._calculate_performance_metrics()
            
            logger.info("回测完成")
            return True
        except Exception as e:
            logger.error(f"回测过程中发生异常: {str(e)}")
            return False
    
    def _reset_backtest_state(self):
        """重置回测状态"""
        self.equity_curve = None
        self.trades = []
        self.positions = pd.DataFrame()
        self.performance_metrics = {}
        
        self.current_date = None
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.current_nav = self.initial_capital
    
    def _get_current_data(self, date: datetime) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """获取当前日期的数据
        
        Args:
            date: 当前日期
        
        Returns:
            当前日期的数据
        """
        if self.mode == self.MODE_SINGLE_ASSET:
            # 对于单资产模式，返回当前日期的数据
            return self.data.loc[date]
        else:
            # 对于投资组合模式，返回每个资产的当前日期数据
            current_data = {}
            for asset, asset_data in self.data.items():
                if date in asset_data.index:
                    current_data[asset] = asset_data.loc[date]
            return current_data
    
    def _get_current_prices(self, date: datetime) -> Dict[str, float]:
        """获取当前日期的价格
        
        Args:
            date: 当前日期
        
        Returns:
            当前价格字典
        """
        prices = {}
        
        if self.mode == self.MODE_SINGLE_ASSET:
            # 单资产模式
            if 'close' in self.data.columns and date in self.data.index:
                prices['asset'] = self.data.loc[date, 'close']
            elif date in self.data.index:
                # 假设第一列是价格
                prices['asset'] = self.data.iloc[self.data.index.get_loc(date), 0]
        else:
            # 投资组合模式
            for asset, asset_data in self.data.items():
                if 'close' in asset_data.columns and date in asset_data.index:
                    prices[asset] = asset_data.loc[date, 'close']
                elif date in asset_data.index:
                    # 假设第一列是价格
                    prices[asset] = asset_data.iloc[asset_data.index.get_loc(date), 0]
        
        return prices
    
    def _calculate_portfolio_value(self, 
                                  date: datetime,
                                  current_prices: Dict[str, float]) -> float:
        """计算投资组合价值
        
        Args:
            date: 当前日期
            current_prices: 当前价格
        
        Returns:
            投资组合价值
        """
        # 计算头寸的市场价值
        positions_value = 0.0
        
        for asset, quantity in self.current_positions.items():
            if asset in current_prices and quantity != 0:
                positions_value += quantity * current_prices[asset]
        
        # 总组合价值 = 现金 + 头寸市场价值
        portfolio_value = self.current_capital + positions_value
        
        return portfolio_value
    
    def _generate_signal(self, 
                        current_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                        day_index: int) -> Dict[str, int]:
        """生成交易信号
        
        Args:
            current_data: 当前市场数据
            day_index: 当前日期在回测序列中的索引
        
        Returns:
            交易信号字典
        """
        try:
            # 调用策略函数生成信号
            signal = self.strategy_func(
                data=current_data,
                current_positions=self.current_positions,
                current_capital=self.current_capital,
                day_index=day_index,
                **self.strategy_params
            )
            
            # 验证信号格式
            if not isinstance(signal, dict):
                logger.warning("策略返回的信号格式不正确，应为字典类型")
                return {}
            
            # 标准化信号值
            for asset, value in signal.items():
                if value > 0:
                    signal[asset] = self.DIRECTION_LONG
                elif value < 0:
                    signal[asset] = self.DIRECTION_SHORT
                else:
                    signal[asset] = self.DIRECTION_FLAT
            
            return signal
        except Exception as e:
            logger.error(f"生成交易信号时发生异常: {str(e)}")
            return {}
    
    def _execute_trades(self, 
                       signal: Dict[str, int],
                       current_prices: Dict[str, float]):
        """执行交易
        
        Args:
            signal: 交易信号
            current_prices: 当前价格
        """
        # 计算需要的头寸大小
        position_sizes = self._calculate_position_size(signal, current_prices)
        
        # 执行每笔交易
        for asset, target_position in position_sizes.items():
            # 获取当前头寸
            current_position = self.current_positions.get(asset, 0)
            
            # 计算需要交易的数量
            trade_quantity = target_position - current_position
            
            # 如果需要交易
            if trade_quantity != 0 and asset in current_prices:
                # 确定交易方向
                direction = self.DIRECTION_LONG if trade_quantity > 0 else self.DIRECTION_SHORT
                
                # 计算滑点后的实际价格
                price = current_prices[asset]
                actual_price = self._calculate_slippage(asset, abs(trade_quantity), price, direction)
                
                # 计算交易成本
                transaction_cost = self._calculate_transaction_cost(asset, abs(trade_quantity), actual_price, direction)
                
                # 计算交易金额
                trade_amount = abs(trade_quantity) * actual_price + transaction_cost
                
                # 检查资金是否足够
                if trade_amount > self.current_capital:
                    logger.warning(f"资金不足，无法执行交易: {asset}, 数量: {trade_quantity}")
                    continue
                
                # 更新资金
                if direction == self.DIRECTION_LONG:
                    self.current_capital -= trade_amount
                else:
                    # 卖空交易，假设可以完全使用卖空所得资金
                    self.current_capital += trade_quantity * actual_price - transaction_cost
                
                # 更新头寸
                self.current_positions[asset] = target_position
                
                # 记录交易
                trade_record = {
                    'date': self.current_date,
                    'asset': asset,
                    'quantity': trade_quantity,
                    'price': actual_price,
                    'cost': transaction_cost,
                    'amount': trade_amount,
                    'direction': direction,
                    'total_position': target_position
                }
                self.trades.append(trade_record)
                
                logger.debug(f"执行交易: {trade_record}")
    
    def _update_position_record(self, date: datetime):
        """更新头寸记录
        
        Args:
            date: 当前日期
        """
        # 创建当前日期的头寸记录
        position_record = {'date': date}
        position_record.update(self.current_positions)
        
        # 添加到头寸记录DataFrame
        if self.positions.empty:
            self.positions = pd.DataFrame([position_record])
        else:
            self.positions = pd.concat([self.positions, pd.DataFrame([position_record])], ignore_index=True)
        
        # 设置日期索引
        self.positions.set_index('date', inplace=True)
    
    def _calculate_performance_metrics(self):
        """计算回测性能指标"""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            logger.warning("没有足够的权益曲线数据计算性能指标")
            return
        
        # 计算基本指标
        total_return = (self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]) - 1
        
        # 计算年化收益率
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (252 / days) - 1
        else:
            annualized_return = 0.0
        
        # 计算波动率
        returns = self.equity_curve['returns'].dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # 计算最大回撤
        cumulative_max = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] / cumulative_max) - 1
        max_drawdown = drawdown.min()
        
        # 计算胜率
        winning_trades = len([t for t in self.trades if t['direction'] == self.DIRECTION_LONG and t['price'] < self._get_price_at_exit(t)]) + \
                         len([t for t in self.trades if t['direction'] == self.DIRECTION_SHORT and t['price'] > self._get_price_at_exit(t)])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        if total_trades > 0:
            profit = sum([t['quantity'] * (self._get_price_at_exit(t) - t['price']) for t in self.trades if (t['direction'] == self.DIRECTION_LONG and self._get_price_at_exit(t) > t['price']) or \
                          (t['direction'] == self.DIRECTION_SHORT and self._get_price_at_exit(t) < t['price'])])
            loss = sum([abs(t['quantity'] * (self._get_price_at_exit(t) - t['price'])) for t in self.trades if (t['direction'] == self.DIRECTION_LONG and self._get_price_at_exit(t) < t['price']) or \
                        (t['direction'] == self.DIRECTION_SHORT and self._get_price_at_exit(t) > t['price'])])
            profit_factor = profit / loss if loss > 0 else 0
        else:
            profit_factor = 0
        
        # 构建性能指标字典
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'start_date': self.equity_curve.index[0],
            'end_date': self.equity_curve.index[-1],
            'duration_days': days
        }
    
    def _get_price_at_exit(self, trade: Dict[str, Any]) -> float:
        """获取交易退出时的价格
        
        Args:
            trade: 交易记录
        
        Returns:
            退出价格
        """
        # 查找该资产的下一次交易或回测结束时的价格
        asset = trade['asset']
        trade_index = self.trades.index(trade)
        
        # 查找下一次交易
        for next_trade in self.trades[trade_index + 1:]:
            if next_trade['asset'] == asset:
                return next_trade['price']
        
        # 如果没有下一次交易，使用回测结束时的价格
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            end_date = self.equity_curve.index[-1]
            end_prices = self._get_current_prices(end_date)
            if asset in end_prices:
                return end_prices[asset]
        
        # 如果都找不到，返回交易时的价格
        return trade['price']
    
    def get_results(self) -> Dict[str, Any]:
        """获取回测结果
        
        Returns:
            回测结果字典
        """
        results = {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'positions': self.positions,
            'performance_metrics': self.performance_metrics,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_nav if self.current_nav else self.initial_capital,
            'backtest_mode': self.mode
        }
        
        return results
    
    def generate_report(self, 
                       output_file: Optional[str] = None,
                       detailed: bool = True) -> Dict[str, Any]:
        """生成回测报告
        
        Args:
            output_file: 输出文件路径
            detailed: 是否生成详细报告
        
        Returns:
            格式化的回测报告数据
        """
        # 构建报告
        report = {
            'report_date': datetime.now(),
            'summary': {
                'backtest_mode': self.mode,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_nav if self.current_nav else self.initial_capital,
                'total_return_pct': self.performance_metrics.get('total_return', 0) * 100,
                'annualized_return_pct': self.performance_metrics.get('annualized_return', 0) * 100,
                'max_drawdown_pct': self.performance_metrics.get('max_drawdown', 0) * 100,
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'total_trades': self.performance_metrics.get('total_trades', 0),
                'win_rate_pct': self.performance_metrics.get('win_rate', 0) * 100,
                'profit_factor': self.performance_metrics.get('profit_factor', 0),
                'backtest_duration_days': self.performance_metrics.get('duration_days', 0)
            }
        }
        
        # 如果需要详细报告
        if detailed:
            report['detailed_metrics'] = self.performance_metrics
            report['trades'] = self.trades
            report['equity_curve'] = self.equity_curve.to_dict() if self.equity_curve is not None else {}
            
            # 如果有多个资产，添加资产表现
            if self.mode == self.MODE_PORTFOLIO:
                asset_performance = {}
                # 计算每个资产的表现（简化版）
                report['asset_performance'] = asset_performance
        
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
                
                logger.info(f"回测报告已保存到: {output_file}")
            except Exception as e:
                logger.error(f"保存回测报告时发生异常: {str(e)}")
        
        logger.info("回测报告生成完成")
        
        return report
    
    def save_results(self, 
                    output_dir: Optional[str] = None,
                    prefix: str = 'backtest') -> Dict[str, str]:
        """保存回测结果
        
        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        
        Returns:
            保存的文件路径字典
        """
        try:
            logger.info("保存回测结果")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存目录
            if output_dir is None:
                output_dir = os.path.join(self.config.get('results_dir', './results'), 'backtesting')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 准备保存的文件路径
            saved_files = {}
            
            # 保存权益曲线
            if self.equity_curve is not None:
                equity_file = os.path.join(output_dir, f"{prefix}_{timestamp}_equity.csv")
                self.equity_curve.to_csv(equity_file)
                saved_files['equity_curve'] = equity_file
            
            # 保存交易记录
            if self.trades:
                trades_file = os.path.join(output_dir, f"{prefix}_{timestamp}_trades.csv")
                pd.DataFrame(self.trades).to_csv(trades_file, index=False)
                saved_files['trades'] = trades_file
            
            # 保存头寸记录
            if not self.positions.empty:
                positions_file = os.path.join(output_dir, f"{prefix}_{timestamp}_positions.csv")
                self.positions.to_csv(positions_file)
                saved_files['positions'] = positions_file
            
            # 保存性能指标
            if self.performance_metrics:
                metrics_file = os.path.join(output_dir, f"{prefix}_{timestamp}_metrics.json")
                # 将datetime对象转换为字符串
                metrics_serializable = self.performance_metrics.copy()
                for key, value in metrics_serializable.items():
                    if isinstance(value, datetime):
                        metrics_serializable[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
                
                saved_files['metrics'] = metrics_file
            
            logger.info(f"回测结果已保存到: {output_dir}")
            
            return saved_files
        except Exception as e:
            logger.error(f"保存回测结果时发生异常: {str(e)}")
            raise
    
    def plot_equity_curve(self, 
                         ax: Optional[Any] = None,
                         show: bool = True,
                         save_path: Optional[str] = None) -> Any:
        """绘制权益曲线
        
        Args:
            ax: Matplotlib轴对象
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if self.equity_curve is None:
                logger.warning("没有权益曲线数据可绘制")
                return None
            
            # 创建图表
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制权益曲线
            ax.plot(self.equity_curve.index, self.equity_curve['equity'], label='Equity Curve')
            
            # 设置图表属性
            ax.set_title('Backtest Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置日期格式化
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"权益曲线图已保存到: {save_path}")
            
            # 显示图表
            if show and ax is None:
                plt.show()
            
            return ax
        except Exception as e:
            logger.error(f"绘制权益曲线时发生异常: {str(e)}")
            return None
    
    def plot_drawdown(self, 
                     ax: Optional[Any] = None,
                     show: bool = True,
                     save_path: Optional[str] = None) -> Any:
        """绘制回撤曲线
        
        Args:
            ax: Matplotlib轴对象
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if self.equity_curve is None:
                logger.warning("没有权益曲线数据可绘制回撤")
                return None
            
            # 计算回撤
            cumulative_max = self.equity_curve['equity'].cummax()
            drawdown = (self.equity_curve['equity'] / cumulative_max) - 1
            
            # 创建图表
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制回撤曲线
            ax.fill_between(drawdown.index, drawdown * 100, 0, where=drawdown < 0, 
                          facecolor='red', alpha=0.3, label='Drawdown (%)')
            
            # 设置图表属性
            ax.set_title('Backtest Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置日期格式化
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回撤曲线图已保存到: {save_path}")
            
            # 显示图表
            if show and ax is None:
                plt.show()
            
            return ax
        except Exception as e:
            logger.error(f"绘制回撤曲线时发生异常: {str(e)}")
            return None
    
    def plot_monthly_returns(self, 
                           ax: Optional[Any] = None,
                           show: bool = True,
                           save_path: Optional[str] = None) -> Any:
        """绘制月度收益率热力图
        
        Args:
            ax: Matplotlib轴对象
            show: 是否显示图表
            save_path: 保存路径
        
        Returns:
            Matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.equity_curve is None:
                logger.warning("没有权益曲线数据可绘制月度收益率")
                return None
            
            # 计算月度收益率
            monthly_returns = self.equity_curve['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_pct = monthly_returns * 100
            
            # 创建透视表
            monthly_returns_pct.index = pd.to_datetime(monthly_returns_pct.index)
            monthly_data = pd.DataFrame({
                'Year': monthly_returns_pct.index.year,
                'Month': monthly_returns_pct.index.month_name(),
                'Return': monthly_returns_pct.values
            })
            
            pivot_table = monthly_data.pivot('Year', 'Month', 'Return')
            
            # 确保月份顺序正确
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            pivot_table = pivot_table.reindex(columns=month_order)
            
            # 创建图表
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
            
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
            if show and ax is None:
                plt.show()
            
            return ax
        except Exception as e:
            logger.error(f"绘制月度收益率热力图时发生异常: {str(e)}")
            return None

# 模块版本
__version__ = '0.1.0'