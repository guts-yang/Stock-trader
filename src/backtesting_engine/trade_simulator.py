import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import json
import os
import math

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradeSimulator:
    """交易模拟器
    用于在回测环境中模拟交易执行，处理订单、成交和滑点等
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 initial_balance: float = 100000.0,
                 slippage_model: Optional[Callable] = None,
                 transaction_cost_model: Optional[Callable] = None,
                 liquidity_model: Optional[Callable] = None,
                 log_level: int = logging.INFO):
        """初始化交易模拟器
        
        Args:
            config: 配置字典
            initial_balance: 初始资金
            slippage_model: 滑点模型函数
            transaction_cost_model: 交易成本模型函数
            liquidity_model: 流动性模型函数
            log_level: 日志级别
        """
        self.config = config or {}
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.slippage_model = slippage_model or self._default_slippage_model
        self.transaction_cost_model = transaction_cost_model or self._default_transaction_cost_model
        self.liquidity_model = liquidity_model or self._default_liquidity_model
        
        # 交易状态
        self.positions = {}
        self.order_book = {}
        self.trade_history = []
        self.order_history = []
        self.current_time = None
        
        # 配置参数
        self.max_order_size = self.config.get('max_order_size', 0.1)  # 最大订单规模占比
        self.min_order_size = self.config.get('min_order_size', 0.001)  # 最小订单规模
        self.slippage_rate = self.config.get('slippage_rate', 0.001)  # 默认滑点率
        self.commission_rate = self.config.get('commission_rate', 0.0005)  # 默认佣金率
        self.min_commission = self.config.get('min_commission', 1.0)  # 最小佣金
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("TradeSimulator 初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"trade_simulator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def _default_slippage_model(self, 
                               price: float,
                               quantity: float,
                               direction: str,
                               asset: str,
                               current_time: datetime,
                               **kwargs) -> float:
        """默认滑点模型
        
        Args:
            price: 参考价格
            quantity: 交易数量
            direction: 交易方向（buy/sell）
            asset: 资产名称
            current_time: 当前时间
            **kwargs: 其他参数
        
        Returns:
            调整后的实际成交价格
        """
        # 简单滑点模型：基于价格和交易数量的线性模型
        # 从配置获取滑点率，默认0.1%
        slippage_rate = kwargs.get('slippage_rate', self.slippage_rate)
        
        # 计算滑点
        slippage = price * slippage_rate * math.sqrt(abs(quantity) / 1000)  # 假设滑点与交易量平方根成正比
        
        # 根据交易方向调整价格
        if direction.lower() == 'buy':
            actual_price = price + slippage
        elif direction.lower() == 'sell':
            actual_price = price - slippage
        else:
            actual_price = price
        
        return actual_price
    
    def _default_transaction_cost_model(self, 
                                       price: float,
                                       quantity: float,
                                       asset: str,
                                       current_time: datetime,
                                       **kwargs) -> float:
        """默认交易成本模型
        
        Args:
            price: 成交价格
            quantity: 交易数量
            asset: 资产名称
            current_time: 当前时间
            **kwargs: 其他参数
        
        Returns:
            交易成本
        """
        # 计算交易金额
        trade_value = price * abs(quantity)
        
        # 计算佣金
        commission_rate = kwargs.get('commission_rate', self.commission_rate)
        min_commission = kwargs.get('min_commission', self.min_commission)
        
        commission = max(trade_value * commission_rate, min_commission)
        
        return commission
    
    def _default_liquidity_model(self, 
                                asset: str,
                                quantity: float,
                                current_time: datetime,
                                **kwargs) -> float:
        """默认流动性模型
        
        Args:
            asset: 资产名称
            quantity: 交易数量
            current_time: 当前时间
            **kwargs: 其他参数
        
        Returns:
            可执行的数量比例 (0-1)
        """
        # 简单流动性模型：总是返回1，表示可以完全执行
        return 1.0
    
    def set_time(self, current_time: datetime):
        """设置当前时间
        
        Args:
            current_time: 当前时间
        """
        self.current_time = current_time
        
    def get_balance(self) -> float:
        """获取当前资金余额
        
        Returns:
            当前资金余额
        """
        return self.current_balance
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取当前所有头寸
        
        Returns:
            头寸字典
        """
        return self.positions.copy()
    
    def get_position(self, asset: str) -> Optional[Dict[str, Any]]:
        """获取特定资产的头寸
        
        Args:
            asset: 资产名称
        
        Returns:
            头寸信息字典，如果没有该资产的头寸则返回None
        """
        return self.positions.get(asset, None)
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """计算总资产净值
        
        Args:
            prices: 资产价格字典
        
        Returns:
            总资产净值
        """
        # 计算资金余额加上所有头寸的市值
        equity = self.current_balance
        
        for asset, pos_info in self.positions.items():
            if asset in prices:
                equity += pos_info['quantity'] * prices[asset]
        
        return equity
    
    def get_exposure(self, prices: Dict[str, float]) -> Dict[str, float]:
        """计算暴露度
        
        Args:
            prices: 资产价格字典
        
        Returns:
            暴露度字典
        """
        # 计算总资产净值
        equity = self.get_equity(prices)
        
        exposure = {}
        
        # 计算每个资产的暴露度
        for asset, pos_info in self.positions.items():
            if asset in prices:
                pos_value = pos_info['quantity'] * prices[asset]
                exposure[asset] = pos_value / equity if equity > 0 else 0
        
        return exposure
    
    def place_order(self, 
                   asset: str,
                   quantity: float,
                   order_type: str = 'market',
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'day',
                   **kwargs) -> Dict[str, Any]:
        """下单
        
        Args:
            asset: 资产名称
            quantity: 交易数量（正数为买入，负数为卖出）
            order_type: 订单类型（market, limit, stop, stop_limit）
            price: 价格（对限价单和止损限价单有效）
            stop_price: 止损价格（对止损单和止损限价单有效）
            time_in_force: 订单有效期
            **kwargs: 其他参数
        
        Returns:
            订单信息字典
        """
        try:
            # 验证参数
            if not asset or quantity == 0:
                logger.error(f"无效的订单参数: asset={asset}, quantity={quantity}")
                return {'status': 'rejected', 'reason': '无效的订单参数'}
            
            # 检查订单规模
            equity = self.get_equity(kwargs.get('prices', {}))
            if equity > 0 and abs(quantity * (price or 0)) > equity * self.max_order_size:
                logger.warning(f"订单规模超过最大限制: {asset}, quantity={quantity}")
                return {'status': 'rejected', 'reason': '订单规模超过最大限制'}
            
            if abs(quantity) < self.min_order_size:
                logger.warning(f"订单规模小于最小限制: {asset}, quantity={quantity}")
                return {'status': 'rejected', 'reason': '订单规模小于最小限制'}
            
            # 创建订单ID
            order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{asset}"
            
            # 构建订单信息
            order = {
                'order_id': order_id,
                'asset': asset,
                'quantity': quantity,
                'remaining_quantity': quantity,
                'order_type': order_type.lower(),
                'price': price,
                'stop_price': stop_price,
                'time_in_force': time_in_force.lower(),
                'status': 'pending',
                'placed_time': self.current_time or datetime.now(),
                'filled_time': None,
                'filled_price': None,
                'transaction_cost': 0,
                'slippage': 0,
                'reason': None,
                **kwargs
            }
            
            # 添加到订单簿
            if asset not in self.order_book:
                self.order_book[asset] = []
            self.order_book[asset].append(order)
            
            # 添加到订单历史
            self.order_history.append(order.copy())
            
            logger.info(f"下单成功: {asset}, {quantity}, {order_type}")
            
            # 对于市价单，立即尝试执行
            if order_type.lower() == 'market':
                return self.execute_order(order_id, **kwargs)
            
            return order
        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            return {'status': 'rejected', 'reason': str(e)}
    
    def execute_order(self, 
                     order_id: str,
                     **kwargs) -> Dict[str, Any]:
        """执行订单
        
        Args:
            order_id: 订单ID
            **kwargs: 其他参数，包括当前价格等
        
        Returns:
            执行后的订单信息
        """
        try:
            # 查找订单
            order = None
            for asset, orders in self.order_book.items():
                for i, o in enumerate(orders):
                    if o['order_id'] == order_id:
                        order = o
                        order_index = i
                        break
                if order:
                    break
            
            if not order:
                logger.warning(f"找不到订单: {order_id}")
                return {'status': 'rejected', 'reason': '找不到订单'}
            
            # 检查订单状态
            if order['status'] != 'pending':
                logger.warning(f"订单已处理: {order_id}, 当前状态: {order['status']}")
                return order
            
            asset = order['asset']
            quantity = order['remaining_quantity']
            
            # 获取当前价格
            prices = kwargs.get('prices', {})
            current_price = prices.get(asset)
            
            if current_price is None:
                logger.warning(f"无法获取资产价格: {asset}")
                return {'status': 'rejected', 'reason': '无法获取资产价格'}
            
            # 计算方向
            direction = 'buy' if quantity > 0 else 'sell'
            
            # 应用流动性模型
            liquidity_factor = self.liquidity_model(asset=asset, 
                                                  quantity=quantity, 
                                                  current_time=self.current_time,
                                                  **kwargs)
            
            # 计算实际可执行数量
            executable_quantity = quantity * liquidity_factor
            
            # 应用滑点模型
            actual_price = self.slippage_model(price=current_price, 
                                             quantity=executable_quantity,
                                             direction=direction,
                                             asset=asset,
                                             current_time=self.current_time,
                                             **kwargs)
            
            # 应用交易成本模型
            transaction_cost = self.transaction_cost_model(price=actual_price,
                                                          quantity=executable_quantity,
                                                          asset=asset,
                                                          current_time=self.current_time,
                                                          **kwargs)
            
            # 计算总成本
            total_cost = abs(executable_quantity) * actual_price + transaction_cost
            
            # 检查资金是否足够
            if direction == 'buy' and total_cost > self.current_balance:
                logger.warning(f"资金不足: 需要 {total_cost}, 可用 {self.current_balance}")
                # 按比例减少交易数量
                scale_factor = self.current_balance / total_cost
                executable_quantity *= scale_factor
                transaction_cost *= scale_factor
                total_cost = self.current_balance
            
            # 更新资金余额
            if direction == 'buy':
                self.current_balance -= total_cost
            else:
                self.current_balance += abs(executable_quantity) * actual_price - transaction_cost
            
            # 更新头寸
            if asset not in self.positions:
                self.positions[asset] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'entry_time': self.current_time or datetime.now(),
                    'total_cost': 0
                }
            
            # 计算新的平均价格
            current_pos = self.positions[asset]
            new_quantity = current_pos['quantity'] + executable_quantity
            
            if new_quantity != 0:
                # 如果未平仓，更新平均价格
                current_pos['avg_price'] = ((current_pos['quantity'] * current_pos['avg_price'] + 
                                           executable_quantity * actual_price) / 
                                          new_quantity) if new_quantity != 0 else 0
                current_pos['total_cost'] += abs(executable_quantity) * actual_price
            else:
                # 如果平仓，重置头寸
                current_pos['avg_price'] = 0
                current_pos['total_cost'] = 0
            
            current_pos['quantity'] = new_quantity
            
            # 如果是新开仓，更新入场时间
            if current_pos['quantity'] != 0 and current_pos['quantity'] - executable_quantity == 0:
                current_pos['entry_time'] = self.current_time or datetime.now()
            
            # 更新订单状态
            order['remaining_quantity'] = quantity - executable_quantity
            
            if order['remaining_quantity'] == 0:
                order['status'] = 'filled'
                # 从订单簿中移除
                self.order_book[asset].pop(order_index)
            else:
                order['status'] = 'partially_filled'
                # 更新订单簿中的订单
                self.order_book[asset][order_index] = order
            
            order['filled_time'] = self.current_time or datetime.now()
            order['filled_price'] = actual_price
            order['transaction_cost'] = transaction_cost
            order['slippage'] = actual_price - current_price if direction == 'buy' else current_price - actual_price
            
            # 记录交易
            if executable_quantity != 0:
                trade = {
                    'trade_id': f"trade_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{asset}",
                    'order_id': order_id,
                    'asset': asset,
                    'quantity': executable_quantity,
                    'price': actual_price,
                    'direction': direction,
                    'transaction_cost': transaction_cost,
                    'trade_time': self.current_time or datetime.now(),
                    'slippage': order['slippage']
                }
                
                self.trade_history.append(trade)
            
            logger.info(f"订单执行成功: {order_id}, 执行数量: {executable_quantity}")
            
            return order
        except Exception as e:
            logger.error(f"执行订单失败: {str(e)}")
            # 更新订单状态为失败
            for asset, orders in self.order_book.items():
                for i, o in enumerate(orders):
                    if o['order_id'] == order_id:
                        o['status'] = 'failed'
                        o['reason'] = str(e)
                        self.order_book[asset][i] = o
                        break
            return {'status': 'failed', 'reason': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """取消订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            取消后的订单信息
        """
        try:
            # 查找订单
            order = None
            for asset, orders in self.order_book.items():
                for i, o in enumerate(orders):
                    if o['order_id'] == order_id:
                        order = o
                        order_index = i
                        break
                if order:
                    break
            
            if not order:
                logger.warning(f"找不到订单: {order_id}")
                return {'status': 'rejected', 'reason': '找不到订单'}
            
            # 检查订单状态
            if order['status'] not in ['pending', 'partially_filled']:
                logger.warning(f"订单无法取消: {order_id}, 当前状态: {order['status']}")
                return order
            
            # 更新订单状态
            order['status'] = 'cancelled'
            order['cancelled_time'] = self.current_time or datetime.now()
            
            # 从订单簿中移除
            self.order_book[order['asset']].pop(order_index)
            
            logger.info(f"订单取消成功: {order_id}")
            
            return order
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            return {'status': 'failed', 'reason': str(e)}
    
    def update_orders(self, prices: Dict[str, float], **kwargs) -> List[Dict[str, Any]]:
        """更新所有未执行的订单
        
        Args:
            prices: 当前资产价格字典
            **kwargs: 其他参数
        
        Returns:
            执行的订单列表
        """
        executed_orders = []
        
        # 创建订单簿的副本以避免在迭代中修改
        order_book_copy = {asset: orders.copy() for asset, orders in self.order_book.items()}
        
        for asset, orders in order_book_copy.items():
            for order in orders:
                # 检查订单是否应该执行
                if self._should_execute_order(order, prices.get(asset), **kwargs):
                    executed_order = self.execute_order(order['order_id'], prices=prices, **kwargs)
                    if executed_order.get('status') in ['filled', 'partially_filled']:
                        executed_orders.append(executed_order)
            
            # 检查过期订单
            self._check_expired_orders(asset, **kwargs)
            
        return executed_orders
    
    def _should_execute_order(self, 
                            order: Dict[str, Any],
                            current_price: Optional[float],
                            **kwargs) -> bool:
        """判断订单是否应该执行
        
        Args:
            order: 订单信息
            current_price: 当前价格
            **kwargs: 其他参数
        
        Returns:
            是否应该执行
        """
        if current_price is None:
            return False
        
        order_type = order['order_type']
        
        if order_type == 'limit':
            # 限价单：买入单价格大于等于限价，卖出单价格小于等于限价
            if order['quantity'] > 0:  # 买入
                return current_price <= order['price']
            else:  # 卖出
                return current_price >= order['price']
        elif order_type == 'stop':
            # 止损单：买入单价格大于等于止损价，卖出单价格小于等于止损价
            if order['quantity'] > 0:  # 买入
                return current_price >= order['stop_price']
            else:  # 卖出
                return current_price <= order['stop_price']
        elif order_type == 'stop_limit':
            # 止损限价单：先达到止损价，然后按照限价执行
            if order['stop_price'] is None or order['price'] is None:
                return False
            
            # 检查是否达到止损条件
            stop_condition = False
            if order['quantity'] > 0:  # 买入
                stop_condition = current_price >= order['stop_price']
            else:  # 卖出
                stop_condition = current_price <= order['stop_price']
            
            # 如果达到止损条件，按照限价单执行
            if stop_condition:
                if order['quantity'] > 0:  # 买入
                    return current_price <= order['price']
                else:  # 卖出
                    return current_price >= order['price']
            
            return False
        
        return False
    
    def _check_expired_orders(self, asset: str, **kwargs):
        """检查并处理过期订单
        
        Args:
            asset: 资产名称
            **kwargs: 其他参数
        """
        if asset not in self.order_book:
            return
        
        current_time = self.current_time or datetime.now()
        valid_orders = []
        
        for order in self.order_book[asset]:
            # 检查订单是否过期
            if self._is_order_expired(order, current_time, **kwargs):
                order['status'] = 'expired'
                order['expired_time'] = current_time
                self.order_history.append(order.copy())
                logger.info(f"订单过期: {order['order_id']}")
            else:
                valid_orders.append(order)
        
        # 更新订单簿
        self.order_book[asset] = valid_orders
    
    def _is_order_expired(self, 
                        order: Dict[str, Any],
                        current_time: datetime,
                        **kwargs) -> bool:
        """判断订单是否过期
        
        Args:
            order: 订单信息
            current_time: 当前时间
            **kwargs: 其他参数
        
        Returns:
            是否过期
        """
        time_in_force = order['time_in_force']
        placed_time = order['placed_time']
        
        if time_in_force == 'day':
            # 当日有效：如果当前时间已过下单日收盘时间，则过期
            if placed_time.date() < current_time.date():
                return True
        elif time_in_force == 'gtc':  # Good Till Cancelled
            # 除非手动取消，否则一直有效
            return False
        elif time_in_force == 'ioc':  # Immediate or Cancel
            # 立即执行或取消，这里简化处理为过期
            return True
        elif time_in_force == 'fok':  # Fill or Kill
            # 立即全部执行或取消，这里简化处理为过期
            return True
        
        return False
    
    def close_position(self, asset: str, **kwargs) -> Dict[str, Any]:
        """平仓
        
        Args:
            asset: 资产名称
            **kwargs: 其他参数
        
        Returns:
            平仓结果
        """
        # 检查是否有该资产的头寸
        if asset not in self.positions or self.positions[asset]['quantity'] == 0:
            logger.warning(f"没有该资产的头寸: {asset}")
            return {'status': 'rejected', 'reason': '没有该资产的头寸'}
        
        # 计算平仓数量
        quantity = -self.positions[asset]['quantity']
        
        # 下单平仓
        return self.place_order(asset=asset, 
                               quantity=quantity, 
                               order_type='market',
                               **kwargs)
    
    def close_all_positions(self, **kwargs) -> List[Dict[str, Any]]:
        """平掉所有头寸
        
        Args:
            **kwargs: 其他参数
        
        Returns:
            平仓结果列表
        """
        results = []
        
        # 创建头寸列表的副本以避免在迭代中修改
        positions_copy = list(self.positions.keys())
        
        for asset in positions_copy:
            result = self.close_position(asset=asset, **kwargs)
            results.append(result)
        
        return results
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """获取交易历史
        
        Returns:
            交易历史列表
        """
        return self.trade_history.copy()
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """获取订单历史
        
        Returns:
            订单历史列表
        """
        return self.order_history.copy()
    
    def get_open_orders(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取未执行的订单
        
        Returns:
            未执行订单字典
        """
        return self.order_book.copy()
    
    def calculate_pnl(self, prices: Dict[str, float]) -> Dict[str, float]:
        """计算盈亏
        
        Args:
            prices: 资产价格字典
        
        Returns:
            盈亏字典
        """
        # 计算总盈亏
        total_pnl = 0
        
        # 计算每个资产的盈亏
        pnl_by_asset = {}
        
        for asset, pos_info in self.positions.items():
            if asset in prices:
                current_value = pos_info['quantity'] * prices[asset]
                cost_basis = pos_info['total_cost']
                pnl = current_value - cost_basis
                
                pnl_by_asset[asset] = pnl
                total_pnl += pnl
        
        return {
            'total_pnl': total_pnl,
            'pnl_by_asset': pnl_by_asset
        }
    
    def calculate_commission_paid(self) -> float:
        """计算已支付的佣金总额
        
        Returns:
            佣金总额
        """
        return sum(trade.get('transaction_cost', 0) for trade in self.trade_history)
    
    def calculate_slippage_cost(self) -> float:
        """计算滑点成本总额
        
        Returns:
            滑点成本总额
        """
        total_slippage = 0
        
        for trade in self.trade_history:
            # 计算滑点成本：滑点 * 数量
            slippage = trade.get('slippage', 0)
            quantity = trade.get('quantity', 0)
            total_slippage += slippage * abs(quantity)
        
        return total_slippage
    
    def reset(self):
        """重置交易模拟器状态"""
        self.current_balance = self.initial_balance
        self.positions = {}
        self.order_book = {}
        self.trade_history = []
        self.order_history = []
        self.current_time = None
        
        logger.info("交易模拟器已重置")
    
    def save_state(self, file_path: str) -> bool:
        """保存交易模拟器状态
        
        Args:
            file_path: 保存文件路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 准备要保存的数据
            state = {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'positions': self.positions,
                'order_book': self.order_book,
                'trade_history': self.trade_history,
                'order_history': self.order_history,
                'current_time': self.current_time.isoformat() if self.current_time else None,
                'config': self.config
            }
            
            # 保存为JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"交易模拟器状态已保存到: {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"保存交易模拟器状态失败: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """加载交易模拟器状态
        
        Args:
            file_path: 加载文件路径
        
        Returns:
            是否加载成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 恢复状态
            self.initial_balance = state.get('initial_balance', self.initial_balance)
            self.current_balance = state.get('current_balance', self.initial_balance)
            self.positions = state.get('positions', {})
            self.order_book = state.get('order_book', {})
            self.trade_history = state.get('trade_history', [])
            self.order_history = state.get('order_history', [])
            
            # 恢复时间
            current_time_str = state.get('current_time')
            if current_time_str:
                self.current_time = datetime.fromisoformat(current_time_str)
            else:
                self.current_time = None
            
            # 恢复配置
            if 'config' in state:
                self.config.update(state['config'])
            
            logger.info(f"交易模拟器状态已从: {file_path} 加载")
            
            return True
        except Exception as e:
            logger.error(f"加载交易模拟器状态失败: {str(e)}")
            return False
    
    def generate_trading_report(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成交易报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径
        
        Returns:
            交易报告数据
        """
        try:
            # 过滤交易历史
            filtered_trades = self.trade_history
            
            if start_date or end_date:
                filtered_trades = []
                for trade in self.trade_history:
                    trade_time = trade['trade_time']
                    if isinstance(trade_time, str):
                        trade_time = datetime.fromisoformat(trade_time)
                    
                    if ((not start_date or trade_time >= start_date) and 
                        (not end_date or trade_time <= end_date)):
                        filtered_trades.append(trade)
            
            # 计算交易统计信息
            total_trades = len(filtered_trades)
            buy_trades = sum(1 for trade in filtered_trades if trade['direction'] == 'buy')
            sell_trades = sum(1 for trade in filtered_trades if trade['direction'] == 'sell')
            
            # 计算总交易量和总交易金额
            total_volume = sum(abs(trade['quantity']) for trade in filtered_trades)
            total_value = sum(abs(trade['quantity']) * trade['price'] for trade in filtered_trades)
            
            # 计算总佣金
            total_commission = sum(trade.get('transaction_cost', 0) for trade in filtered_trades)
            
            # 计算平均每笔交易金额和佣金
            avg_trade_value = total_value / total_trades if total_trades > 0 else 0
            avg_commission_per_trade = total_commission / total_trades if total_trades > 0 else 0
            
            # 按资产分组统计
            trades_by_asset = {}
            for trade in filtered_trades:
                asset = trade['asset']
                if asset not in trades_by_asset:
                    trades_by_asset[asset] = []
                trades_by_asset[asset].append(trade)
            
            # 准备报告数据
            report = {
                'report_date': datetime.now(),
                'period': {
                    'start_date': start_date or (self.trade_history[0]['trade_time'] if self.trade_history else None),
                    'end_date': end_date or (self.trade_history[-1]['trade_time'] if self.trade_history else None)
                },
                'summary': {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'total_volume': total_volume,
                    'total_value': total_value,
                    'total_commission': total_commission,
                    'avg_trade_value': avg_trade_value,
                    'avg_commission_per_trade': avg_commission_per_trade,
                    'number_of_assets_traded': len(trades_by_asset)
                },
                'by_asset': {},
                'trades': filtered_trades
            }
            
            # 计算每个资产的统计信息
            for asset, trades in trades_by_asset.items():
                asset_total_trades = len(trades)
                asset_total_volume = sum(abs(trade['quantity']) for trade in trades)
                asset_total_value = sum(abs(trade['quantity']) * trade['price'] for trade in trades)
                asset_total_commission = sum(trade.get('transaction_cost', 0) for trade in trades)
                
                report['by_asset'][asset] = {
                    'total_trades': asset_total_trades,
                    'total_volume': asset_total_volume,
                    'total_value': asset_total_value,
                    'total_commission': asset_total_commission,
                    'avg_trade_value': asset_total_value / asset_total_trades if asset_total_trades > 0 else 0,
                    'avg_commission_per_trade': asset_total_commission / asset_total_trades if asset_total_trades > 0 else 0
                }
            
            # 如果指定了输出文件，保存报告
            if output_file:
                try:
                    # 确保目录存在
                    output_dir = os.path.dirname(output_file)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # 保存为JSON文件
                    with open(output_file, 'w', encoding='utf-8') as f:
                        # 将datetime对象转换为字符串
                        report_serializable = report.copy()
                        if 'report_date' in report_serializable and isinstance(report_serializable['report_date'], datetime):
                            report_serializable['report_date'] = report_serializable['report_date'].strftime('%Y-%m-%d %H:%M:%S')
                        if 'start_date' in report_serializable.get('period', {}) and isinstance(report_serializable['period']['start_date'], datetime):
                            report_serializable['period']['start_date'] = report_serializable['period']['start_date'].strftime('%Y-%m-%d')
                        if 'end_date' in report_serializable.get('period', {}) and isinstance(report_serializable['period']['end_date'], datetime):
                            report_serializable['period']['end_date'] = report_serializable['period']['end_date'].strftime('%Y-%m-%d')
                        
                        json.dump(report_serializable, f, indent=2, ensure_ascii=False, default=str)
                    
                    logger.info(f"交易报告已保存到: {output_file}")
                except Exception as e:
                    logger.error(f"保存交易报告时发生异常: {str(e)}")
            
            logger.info("交易报告生成完成")
            
            return report
        except Exception as e:
            logger.error(f"生成交易报告时发生异常: {str(e)}")
            raise

# 模块版本
__version__ = '0.1.0'