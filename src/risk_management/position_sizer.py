import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import json

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PositionSizer:
    """头寸大小管理器
    负责计算和管理每个交易的最佳头寸大小，实现多种头寸计算策略
    """
    # 支持的头寸计算策略
    STRATEGY_FIXED_AMOUNT = 'fixed_amount'
    STRATEGY_FIXED_PERCENTAGE = 'fixed_percentage'
    STRATEGY_RISK_PER_TRADE = 'risk_per_trade'
    STRATEGY_VOLATILITY_ADJUSTED = 'volatility_adjusted'
    STRATEGY_OPTIMAL_F = 'optimal_f'
    STRATEGY_KELLY_CRITERION = 'kelly_criterion'
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 initial_capital: float = 1000000.0,
                 strategy: str = STRATEGY_RISK_PER_TRADE,
                 fixed_amount: float = 10000.0,
                 fixed_percentage: float = 0.02,
                 risk_per_trade: float = 0.02,
                 max_position_percentage: float = 0.1,
                 volatility_lookback: int = 20,
                 optimal_f_multiplier: float = 0.5,
                 kelly_criterion_multiplier: float = 0.5,
                 min_position_size: int = 1,
                 max_position_size: int = None):
        """初始化头寸大小管理器
        
        Args:
            config: 配置字典
            initial_capital: 初始资金
            strategy: 头寸计算策略
            fixed_amount: 固定金额策略的金额
            fixed_percentage: 固定百分比策略的百分比
            risk_per_trade: 单笔交易风险比例
            max_position_percentage: 单个持仓最大百分比
            volatility_lookback: 波动率计算的回看期
            optimal_f_multiplier: Optimal F策略的乘数
            kelly_criterion_multiplier: Kelly准则的乘数
            min_position_size: 最小头寸大小
            max_position_size: 最大头寸大小
        """
        self.config = config or {}
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy = strategy
        self.fixed_amount = fixed_amount
        self.fixed_percentage = fixed_percentage
        self.risk_per_trade = risk_per_trade
        self.max_position_percentage = max_position_percentage
        self.volatility_lookback = volatility_lookback
        self.optimal_f_multiplier = optimal_f_multiplier
        self.kelly_criterion_multiplier = kelly_criterion_multiplier
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        
        # 交易历史记录，用于计算Optimal F和Kelly准则
        self.trade_history = []
        
        # 波动率数据缓存
        self.volatility_cache = {}
        
        # 初始化日志
        self._init_logger()
        
        logger.info(f"PositionSizer 初始化完成，使用策略: {strategy}")
    
    def _init_logger(self):
        """初始化日志记录器"""
        import os
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"position_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def update_capital(self, new_capital: float) -> None:
        """更新当前资金
        
        Args:
            new_capital: 新的资金金额
        """
        self.current_capital = new_capital
        logger.info(f"更新当前资金: {new_capital:.2f}")
    
    def update_trade_history(self, 
                            ticker: str,
                            entry_price: float,
                            exit_price: float,
                            shares: int,
                            entry_date: datetime,
                            exit_date: datetime) -> None:
        """更新交易历史
        
        Args:
            ticker: 股票代码
            entry_price: 入场价格
            exit_price: 出场价格
            shares: 交易数量
            entry_date: 入场日期
            exit_date: 出场日期
        """
        # 计算交易结果
        profit_loss = (exit_price - entry_price) * shares
        return_percent = (exit_price - entry_price) / entry_price
        
        # 添加到交易历史
        trade = {
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'profit_loss': profit_loss,
            'return_percent': return_percent,
            'position_value': entry_price * shares
        }
        
        self.trade_history.append(trade)
        logger.info(f"添加交易历史: {ticker}, 盈利: {profit_loss:.2f}, 收益率: {return_percent:.2%}")
    
    def calculate_position_size(self, 
                               ticker: str,
                               current_price: float,
                               stop_loss_price: Optional[float] = None,
                               volatility: Optional[float] = None,
                               returns_data: Optional[pd.DataFrame] = None,
                               additional_params: Optional[Dict[str, Any]] = None) -> int:
        """计算头寸大小
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
            stop_loss_price: 止损价格
            volatility: 波动率
            returns_data: 收益率数据
            additional_params: 其他参数
        
        Returns:
            建议的持仓数量
        """
        # 确保价格有效
        if current_price <= 0:
            logger.error(f"无效的价格: {current_price}")
            return 0
        
        # 根据策略计算头寸大小
        if self.strategy == self.STRATEGY_FIXED_AMOUNT:
            shares = self._calculate_fixed_amount(current_price)
        elif self.strategy == self.STRATEGY_FIXED_PERCENTAGE:
            shares = self._calculate_fixed_percentage(current_price)
        elif self.strategy == self.STRATEGY_RISK_PER_TRADE:
            shares = self._calculate_risk_per_trade(current_price, stop_loss_price, volatility)
        elif self.strategy == self.STRATEGY_VOLATILITY_ADJUSTED:
            shares = self._calculate_volatility_adjusted(ticker, current_price, volatility, returns_data)
        elif self.strategy == self.STRATEGY_OPTIMAL_F:
            shares = self._calculate_optimal_f(current_price, additional_params)
        elif self.strategy == self.STRATEGY_KELLY_CRITERION:
            shares = self._calculate_kelly_criterion(current_price, additional_params)
        else:
            logger.warning(f"未知的策略: {self.strategy}，使用风险百分比策略")
            shares = self._calculate_risk_per_trade(current_price, stop_loss_price, volatility)
        
        # 应用最小头寸大小限制
        shares = max(self.min_position_size, shares)
        
        # 应用最大头寸大小限制
        if self.max_position_size is not None:
            shares = min(self.max_position_size, shares)
        
        # 应用最大持仓百分比限制
        max_position_value = self.current_capital * self.max_position_percentage
        max_shares = int(max_position_value / current_price)
        shares = min(max_shares, shares)
        
        logger.info(f"为 {ticker} 计算的头寸大小: {shares} 股，价格: {current_price:.2f}，价值: {shares * current_price:.2f}")
        
        return shares
    
    def _calculate_fixed_amount(self, current_price: float) -> int:
        """固定金额策略
        
        Args:
            current_price: 当前价格
        
        Returns:
            建议的持仓数量
        """
        # 计算可以购买的数量
        shares = int(self.fixed_amount / current_price)
        return shares
    
    def _calculate_fixed_percentage(self, current_price: float) -> int:
        """固定百分比策略
        
        Args:
            current_price: 当前价格
        
        Returns:
            建议的持仓数量
        """
        # 计算投资金额
        investment_amount = self.current_capital * self.fixed_percentage
        
        # 计算可以购买的数量
        shares = int(investment_amount / current_price)
        return shares
    
    def _calculate_risk_per_trade(self, 
                                 current_price: float,
                                 stop_loss_price: Optional[float] = None,
                                 volatility: Optional[float] = None) -> int:
        """单笔交易风险策略
        
        Args:
            current_price: 当前价格
            stop_loss_price: 止损价格
            volatility: 波动率
        
        Returns:
            建议的持仓数量
        """
        # 计算单笔交易的风险金额
        risk_amount = self.current_capital * self.risk_per_trade
        
        # 计算每单位风险
        if stop_loss_price and stop_loss_price < current_price:
            # 使用止损价格计算风险
            risk_per_share = current_price - stop_loss_price
        elif volatility is not None:
            # 使用波动率计算风险
            risk_per_share = current_price * volatility
        else:
            # 默认使用10%的价格作为风险估计
            risk_per_share = current_price * 0.10
        
        # 确保风险不为零
        if risk_per_share <= 0:
            risk_per_share = current_price * 0.10  # 默认使用10%的价格作为风险
        
        # 计算可以购买的数量
        shares = int(risk_amount / risk_per_share)
        return shares
    
    def _calculate_volatility_adjusted(self, 
                                      ticker: str,
                                      current_price: float,
                                      volatility: Optional[float] = None,
                                      returns_data: Optional[pd.DataFrame] = None) -> int:
        """波动率调整策略
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
            volatility: 波动率
            returns_data: 收益率数据
        
        Returns:
            建议的持仓数量
        """
        # 如果没有提供波动率，尝试计算
        if volatility is None:
            volatility = self._calculate_volatility(ticker, returns_data)
        
        # 如果波动率仍然为None，使用默认值
        if volatility is None:
            logger.warning(f"无法获取 {ticker} 的波动率数据，使用默认值 0.10")
            volatility = 0.10
        
        # 目标风险金额
        target_risk = self.current_capital * self.risk_per_trade
        
        # 计算头寸大小
        # 波动率调整的头寸大小 = 目标风险 / (价格 * 波动率)
        shares = int(target_risk / (current_price * volatility))
        
        return shares
    
    def _calculate_optimal_f(self, 
                            current_price: float,
                            additional_params: Optional[Dict[str, Any]] = None) -> int:
        """Optimal F 策略
        
        Args:
            current_price: 当前价格
            additional_params: 其他参数
        
        Returns:
            建议的持仓数量
        """
        # 检查是否有足够的交易历史
        if len(self.trade_history) < 20:
            logger.warning("交易历史不足，无法计算Optimal F，使用风险百分比策略")
            return self._calculate_risk_per_trade(current_price)
        
        # 计算Optimal F
        optimal_f = self._compute_optimal_f()
        
        # 应用乘数
        optimal_f = optimal_f * self.optimal_f_multiplier
        
        # 计算可以投入的资金
        investment_amount = self.current_capital * optimal_f
        
        # 计算头寸大小
        shares = int(investment_amount / current_price)
        
        logger.info(f"使用Optimal F策略，计算得到: f={optimal_f:.4f}，投资金额={investment_amount:.2f}，头寸大小={shares}")
        
        return shares
    
    def _calculate_kelly_criterion(self, 
                                  current_price: float,
                                  additional_params: Optional[Dict[str, Any]] = None) -> int:
        """Kelly准则策略
        
        Args:
            current_price: 当前价格
            additional_params: 其他参数
        
        Returns:
            建议的持仓数量
        """
        # 检查是否有足够的交易历史
        if len(self.trade_history) < 20:
            logger.warning("交易历史不足，无法计算Kelly准则，使用风险百分比策略")
            return self._calculate_risk_per_trade(current_price)
        
        # 计算赢面和平均盈亏比
        win_rate, avg_win_loss_ratio = self._compute_win_rate_and_ratio()
        
        # 计算Kelly比例
        if avg_win_loss_ratio > 0:
            kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
        else:
            kelly_fraction = 0.0
        
        # 确保Kelly比例不为负
        kelly_fraction = max(0.0, kelly_fraction)
        
        # 应用乘数
        kelly_fraction = kelly_fraction * self.kelly_criterion_multiplier
        
        # 计算可以投入的资金
        investment_amount = self.current_capital * kelly_fraction
        
        # 计算头寸大小
        shares = int(investment_amount / current_price)
        
        logger.info(f"使用Kelly准则策略，计算得到: f={kelly_fraction:.4f}，投资金额={investment_amount:.2f}，头寸大小={shares}")
        
        return shares
    
    def _calculate_volatility(self, 
                             ticker: str,
                             returns_data: Optional[pd.DataFrame] = None) -> Optional[float]:
        """计算波动率
        
        Args:
            ticker: 股票代码
            returns_data: 收益率数据
        
        Returns:
            波动率
        """
        # 检查缓存
        if ticker in self.volatility_cache:
            cached_vol, cache_time = self.volatility_cache[ticker]
            # 如果缓存未满5分钟，返回缓存值
            if (datetime.now() - cache_time).total_seconds() < 300:
                return cached_vol
        
        # 如果提供了收益率数据，计算波动率
        if returns_data is not None and ticker in returns_data.columns:
            # 获取最近N天的数据
            if len(returns_data) >= self.volatility_lookback:
                recent_returns = returns_data[ticker].tail(self.volatility_lookback)
                # 计算日波动率
                daily_vol = recent_returns.std()
                # 转换为年化波动率
                annualized_vol = daily_vol * np.sqrt(252)
                
                # 更新缓存
                self.volatility_cache[ticker] = (annualized_vol, datetime.now())
                
                return annualized_vol
        
        # 如果无法计算，返回None
        return None
    
    def _compute_optimal_f(self) -> float:
        """计算Optimal F值
        
        Returns:
            Optimal F值
        """
        try:
            # 提取交易结果的比率
            trade_ratios = []
            for trade in self.trade_history:
                # 计算交易的回报率（相对于总资金）
                position_value = trade['position_value']
                profit_loss = trade['profit_loss']
                # 计算风险回报率 (盈利/亏损除以投入资金的比例)
                ratio = profit_loss / position_value
                trade_ratios.append(ratio)
            
            # 计算各f值对应的复合收益率
            f_values = np.arange(0.01, 1.01, 0.01)
            best_f = 0.0
            best_returns = -np.inf
            
            for f in f_values:
                # 计算每个f值的复合收益率
                equity = 1.0  # 初始资金设为1
                for ratio in trade_ratios:
                    # 应用f值计算新的资金
                    equity *= (1 + f * ratio)
                    # 如果资金归零，终止计算
                    if equity <= 0:
                        break
                
                # 更新最佳f值
                if equity > best_returns:
                    best_returns = equity
                    best_f = f
            
            return best_f
        except Exception as e:
            logger.error(f"计算Optimal F时发生异常: {str(e)}")
            return 0.02  # 返回默认值
    
    def _compute_win_rate_and_ratio(self) -> Tuple[float, float]:
        """计算胜率和平均盈亏比
        
        Returns:
            (胜率, 平均盈亏比)
        """
        # 分离盈利和亏损交易
        winning_trades = [t for t in self.trade_history if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trade_history if t['profit_loss'] < 0]
        
        # 计算胜率
        total_trades = len(winning_trades) + len(losing_trades)
        if total_trades == 0:
            return 0.5, 1.0  # 返回默认值
        
        win_rate = len(winning_trades) / total_trades
        
        # 计算平均盈利和平均亏损
        avg_win = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['profit_loss']) for t in losing_trades]) if losing_trades else 0
        
        # 计算平均盈亏比
        if avg_loss > 0:
            avg_win_loss_ratio = avg_win / avg_loss
        else:
            avg_win_loss_ratio = 1.0  # 如果没有亏损交易，返回1.0
        
        return win_rate, avg_win_loss_ratio
    
    def calculate_portfolio_position_sizes(self, 
                                         signals: Dict[str, float],
                                         prices: Dict[str, float],
                                         stop_loss_prices: Optional[Dict[str, float]] = None,
                                         portfolio_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """计算整个投资组合的头寸大小
        
        Args:
            signals: 信号字典 {ticker: signal_strength}
            prices: 价格字典 {ticker: price}
            stop_loss_prices: 止损价格字典 {ticker: stop_loss_price}
            portfolio_constraints: 组合约束
        
        Returns:
            头寸大小字典 {ticker: shares}
        """
        if not signals or not prices:
            logger.warning("没有信号或价格数据，无法计算组合头寸大小")
            return {}
        
        # 确保stop_loss_prices不为None
        stop_loss_prices = stop_loss_prices or {}
        
        # 标准化信号强度
        signal_strengths = list(signals.values())
        if signal_strengths:
            max_strength = max(abs(s) for s in signal_strengths)
            if max_strength > 0:
                normalized_signals = {t: s / max_strength for t, s in signals.items()}
            else:
                normalized_signals = signals
        else:
            normalized_signals = signals
        
        # 计算每个资产的头寸大小
        position_sizes = {}
        total_weight = 0.0
        
        # 第一遍：计算每个资产的权重
        weights = {}
        for ticker, signal_strength in normalized_signals.items():
            if ticker not in prices:
                continue
            
            price = prices[ticker]
            stop_loss = stop_loss_prices.get(ticker)
            
            # 计算基础头寸大小
            base_shares = self.calculate_position_size(
                ticker=ticker,
                current_price=price,
                stop_loss_price=stop_loss
            )
            
            # 基于信号强度调整权重
            weight = base_shares * price * abs(signal_strength)
            weights[ticker] = weight
            total_weight += weight
        
        # 如果总权重为零，返回空字典
        if total_weight == 0:
            return {}
        
        # 第二遍：根据权重分配资金
        available_capital = self.current_capital * self.max_position_percentage
        
        for ticker, weight in weights.items():
            if ticker not in prices:
                continue
            
            price = prices[ticker]
            signal_strength = normalized_signals[ticker]
            
            # 计算分配的资金
            allocated_capital = (weight / total_weight) * available_capital
            
            # 计算最终头寸大小
            shares = int(allocated_capital / price)
            
            # 确保至少交易1股
            shares = max(1, shares)
            
            # 添加到结果
            position_sizes[ticker] = shares
        
        # 应用组合约束
        if portfolio_constraints:
            position_sizes = self._apply_portfolio_constraints(position_sizes, prices, portfolio_constraints)
        
        logger.info(f"计算的组合头寸大小: {position_sizes}")
        
        return position_sizes
    
    def _apply_portfolio_constraints(self, 
                                    position_sizes: Dict[str, int],
                                    prices: Dict[str, float],
                                    constraints: Dict[str, Any]) -> Dict[str, int]:
        """应用组合约束
        
        Args:
            position_sizes: 头寸大小字典
            prices: 价格字典
            constraints: 组合约束
        
        Returns:
            调整后的头寸大小字典
        """
        # 计算当前组合价值
        portfolio_value = sum(shares * prices[ticker] for ticker, shares in position_sizes.items())
        
        # 应用总价值约束
        if 'max_total_value' in constraints:
            max_total_value = constraints['max_total_value']
            if portfolio_value > max_total_value:
                # 按比例缩减所有头寸
                scale_factor = max_total_value / portfolio_value
                position_sizes = {t: max(self.min_position_size, int(s * scale_factor)) for t, s in position_sizes.items()}
                logger.info(f"应用总价值约束，缩放因子: {scale_factor:.4f}")
        
        # 应用单个资产最大权重约束
        if 'max_weight_per_asset' in constraints:
            max_weight_per_asset = constraints['max_weight_per_asset']
            for ticker, shares in position_sizes.items():
                if ticker in prices:
                    position_value = shares * prices[ticker]
                    weight = position_value / portfolio_value
                    if weight > max_weight_per_asset:
                        # 调整该资产的头寸大小
                        max_position_value = portfolio_value * max_weight_per_asset
                        new_shares = int(max_position_value / prices[ticker])
                        position_sizes[ticker] = max(self.min_position_size, new_shares)
                        logger.info(f"应用单个资产最大权重约束，调整 {ticker} 头寸从 {shares} 到 {new_shares}")
        
        # 应用行业约束
        if 'sector_limits' in constraints and 'sector_map' in constraints:
            sector_limits = constraints['sector_limits']
            sector_map = constraints['sector_map']
            
            # 计算各行业的当前暴露
            sector_exposures = {}
            for ticker, shares in position_sizes.items():
                if ticker in prices and ticker in sector_map:
                    sector = sector_map[ticker]
                    if sector not in sector_exposures:
                        sector_exposures[sector] = 0
                    sector_exposures[sector] += shares * prices[ticker]
            
            # 检查并调整行业暴露
            for sector, exposure in sector_exposures.items():
                if sector in sector_limits:
                    max_sector_exposure = sector_limits[sector]
                    if exposure > max_sector_exposure:
                        # 按比例缩减该行业内的所有资产
                        scale_factor = max_sector_exposure / exposure
                        for ticker, shares in position_sizes.items():
                            if ticker in sector_map and sector_map[ticker] == sector and ticker in prices:
                                new_shares = max(self.min_position_size, int(shares * scale_factor))
                                position_sizes[ticker] = new_shares
                        logger.info(f"应用行业暴露约束，缩放 {sector} 行业头寸，缩放因子: {scale_factor:.4f}")
        
        return position_sizes
    
    def backtest_position_sizing(self, 
                               signals: pd.DataFrame,
                               prices: pd.DataFrame,
                               stop_loss_prices: Optional[pd.DataFrame] = None,
                               initial_capital: Optional[float] = None) -> pd.DataFrame:
        """回测头寸大小策略
        
        Args:
            signals: 信号DataFrame，索引为日期，列为股票代码
            prices: 价格DataFrame，索引为日期，列为股票代码
            stop_loss_prices: 止损价格DataFrame
            initial_capital: 初始资金
        
        Returns:
            回测结果DataFrame
        """
        if signals.empty or prices.empty:
            logger.warning("信号或价格数据为空，无法进行回测")
            return pd.DataFrame()
        
        # 设置初始资金
        backtest_capital = initial_capital if initial_capital is not None else self.initial_capital
        
        # 确保索引一致
        common_dates = signals.index.intersection(prices.index)
        if common_dates.empty:
            logger.warning("没有共同的日期索引，无法进行回测")
            return pd.DataFrame()
        
        signals = signals.loc[common_dates]
        prices = prices.loc[common_dates]
        
        # 准备结果DataFrame
        results = pd.DataFrame(index=common_dates)
        results['equity'] = backtest_capital
        results['daily_return'] = 0.0
        
        # 跟踪每日持仓
        daily_positions = {}
        
        # 回测
        for date in common_dates:
            # 获取当日数据
            daily_signals = signals.loc[date].dropna()
            daily_prices = prices.loc[date].dropna()
            
            # 获取当日有信号和价格的股票
            valid_tickers = daily_signals.index.intersection(daily_prices.index)
            
            if not valid_tickers.empty:
                # 构建信号和价格字典
                signal_dict = daily_signals[valid_tickers].to_dict()
                price_dict = daily_prices[valid_tickers].to_dict()
                
                # 获取止损价格（如果有）
                stop_loss_dict = {}
                if stop_loss_prices is not None and date in stop_loss_prices.index:
                    daily_stop_loss = stop_loss_prices.loc[date]
                    for ticker in valid_tickers:
                        if ticker in daily_stop_loss and pd.notna(daily_stop_loss[ticker]):
                            stop_loss_dict[ticker] = daily_stop_loss[ticker]
                
                # 计算当日头寸大小
                daily_positions[date] = self.calculate_portfolio_position_sizes(
                    signals=signal_dict,
                    prices=price_dict,
                    stop_loss_prices=stop_loss_dict
                )
            else:
                daily_positions[date] = {}
            
            # 计算当日收益（简化版，实际回测需要更复杂的处理）
            if date > common_dates[0]:
                prev_date = common_dates[common_dates.get_loc(date) - 1]
                prev_positions = daily_positions.get(prev_date, {})
                
                # 计算持仓收益
                daily_profit = 0.0
                for ticker, shares in prev_positions.items():
                    if ticker in daily_prices:
                        prev_price = prices.loc[prev_date, ticker]
                        curr_price = daily_prices[ticker]
                        daily_profit += shares * (curr_price - prev_price)
                
                # 更新权益
                prev_equity = results.loc[prev_date, 'equity']
                curr_equity = prev_equity + daily_profit
                results.loc[date, 'equity'] = curr_equity
                
                # 计算日收益率
                if prev_equity > 0:
                    results.loc[date, 'daily_return'] = daily_profit / prev_equity
        
        # 计算累积收益率
        results['cumulative_return'] = (1 + results['daily_return']).cumprod() - 1
        
        logger.info(f"头寸大小策略回测完成，最终权益: {results['equity'].iloc[-1]:.2f}，总收益率: {results['cumulative_return'].iloc[-1]:.2%}")
        
        return results
    
    def save_state(self, file_path: Optional[str] = None) -> str:
        """保存头寸大小管理器状态
        
        Args:
            file_path: 保存路径
        
        Returns:
            实际保存路径
        """
        try:
            import os
            logger.info("保存头寸大小管理器状态")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if file_path is None:
                states_dir = os.path.join(self.config.get('results_dir', './results'), 'position_sizer_states')
                if not os.path.exists(states_dir):
                    os.makedirs(states_dir)
                
                file_path = os.path.join(states_dir, f"position_sizer_state_{timestamp}.json")
            
            # 准备状态数据
            state_data = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'strategy': self.strategy,
                'fixed_amount': self.fixed_amount,
                'fixed_percentage': self.fixed_percentage,
                'risk_per_trade': self.risk_per_trade,
                'max_position_percentage': self.max_position_percentage,
                'volatility_lookback': self.volatility_lookback,
                'optimal_f_multiplier': self.optimal_f_multiplier,
                'kelly_criterion_multiplier': self.kelly_criterion_multiplier,
                'min_position_size': self.min_position_size,
                'max_position_size': self.max_position_size,
                'timestamp': timestamp,
                'config': self.config
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"头寸大小管理器状态已保存到: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"保存头寸大小管理器状态时发生异常: {str(e)}")
            raise
    
    def load_state(self, file_path: str) -> bool:
        """加载头寸大小管理器状态
        
        Args:
            file_path: 状态文件路径
        
        Returns:
            是否加载成功
        """
        try:
            import os
            logger.info(f"加载头寸大小管理器状态: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"状态文件不存在: {file_path}")
                return False
            
            # 读取状态数据
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 恢复状态
            self.initial_capital = state_data.get('initial_capital', self.initial_capital)
            self.current_capital = state_data.get('current_capital', self.current_capital)
            self.strategy = state_data.get('strategy', self.strategy)
            self.fixed_amount = state_data.get('fixed_amount', self.fixed_amount)
            self.fixed_percentage = state_data.get('fixed_percentage', self.fixed_percentage)
            self.risk_per_trade = state_data.get('risk_per_trade', self.risk_per_trade)
            self.max_position_percentage = state_data.get('max_position_percentage', self.max_position_percentage)
            self.volatility_lookback = state_data.get('volatility_lookback', self.volatility_lookback)
            self.optimal_f_multiplier = state_data.get('optimal_f_multiplier', self.optimal_f_multiplier)
            self.kelly_criterion_multiplier = state_data.get('kelly_criterion_multiplier', self.kelly_criterion_multiplier)
            self.min_position_size = state_data.get('min_position_size', self.min_position_size)
            self.max_position_size = state_data.get('max_position_size', self.max_position_size)
            
            # 更新配置
            if 'config' in state_data:
                self.config.update(state_data['config'])
                # 更新日志
                self.logger = self._init_logger()
            
            logger.info("头寸大小管理器状态加载完成")
            
            return True
        except Exception as e:
            logger.error(f"加载头寸大小管理器状态时发生异常: {str(e)}")
            return False

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['PositionSizer']