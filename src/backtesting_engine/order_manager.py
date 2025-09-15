import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any, Set
import os
import abc
import json
import uuid

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Order:
    """订单类
    表示一个交易订单
    """
    # 订单类型
    class Type:
        MARKET = 'MARKET'      # 市价单
        LIMIT = 'LIMIT'        # 限价单
        STOP = 'STOP'          # 止损单
        STOP_LIMIT = 'STOP_LIMIT'  # 止损限价单
        TRAILING_STOP = 'TRAILING_STOP'  # 跟踪止损单
    
    # 订单方向
    class Side:
        BUY = 'BUY'            # 买入
        SELL = 'SELL'          # 卖出
    
    # 订单状态
    class Status:
        PENDING = 'PENDING'    # 待处理
        FILLED = 'FILLED'      # 已成交
        PARTIALLY_FILLED = 'PARTIALLY_FILLED'  # 部分成交
        CANCELLED = 'CANCELLED'  # 已取消
        REJECTED = 'REJECTED'  # 已拒绝
        EXPIRED = 'EXPIRED'    # 已过期
    
    def __init__(self, 
                 symbol: str,
                 side: str,
                 order_type: str,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 trailing_amount: Optional[float] = None,
                 time_in_force: str = 'GTC',
                 order_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """初始化订单
        
        Args:
            symbol: 交易符号
            side: 订单方向 (BUY/SELL)
            order_type: 订单类型 (MARKET/LIMIT/STOP/STOP_LIMIT/TRAILING_STOP)
            quantity: 订单数量
            price: 价格 (用于限价单和止损限价单)
            stop_price: 止损价格 (用于止损单和止损限价单)
            trailing_amount: 跟踪金额 (用于跟踪止损单)
            time_in_force: 有效时间 (GTC/IOC/FOK/DAY/GTD)
            order_id: 订单ID (如果不提供，将自动生成)
            timestamp: 订单创建时间戳
            metadata: 订单元数据
        """
        # 验证输入参数
        if side not in [self.Side.BUY, self.Side.SELL]:
            raise ValueError(f"无效的订单方向: {side}")
        
        if order_type not in [self.Type.MARKET, self.Type.LIMIT, self.Type.STOP, self.Type.STOP_LIMIT, self.Type.TRAILING_STOP]:
            raise ValueError(f"无效的订单类型: {order_type}")
        
        if quantity <= 0:
            raise ValueError(f"订单数量必须大于0: {quantity}")
        
        # 订单基本信息
        self.order_id = order_id or str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.filled_quantity = 0.0
        self.price = price
        self.stop_price = stop_price
        self.trailing_amount = trailing_amount
        self.time_in_force = time_in_force
        self.status = self.Status.PENDING
        self.timestamp = timestamp or datetime.now()
        self.fill_timestamp = None
        self.last_updated = datetime.now()
        self.metadata = metadata or {}
        
        # 成交记录
        self.fill_records = []
        
        logger.debug(f"创建订单: {self.order_id}, {self.symbol}, {self.side}, {self.order_type}, {self.quantity}")
    
    def fill(self, quantity: float, price: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """部分或完全成交订单
        
        Args:
            quantity: 成交数量
            price: 成交价格
            timestamp: 成交时间戳
        
        Returns:
            成交记录
        """
        if quantity <= 0:
            raise ValueError("成交数量必须大于0")
        
        if quantity > (self.quantity - self.filled_quantity):
            raise ValueError("成交数量不能超过剩余未成交数量")
        
        # 创建成交记录
        fill_record = {
            'fill_id': str(uuid.uuid4()),
            'order_id': self.order_id,
            'timestamp': timestamp or datetime.now(),
            'quantity': quantity,
            'price': price,
            'commission': self.metadata.get('commission', 0.0) * quantity * price
        }
        
        # 更新订单状态
        self.filled_quantity += quantity
        self.last_updated = datetime.now()
        
        # 更新订单状态
        if self.filled_quantity == self.quantity:
            self.status = self.Status.FILLED
            self.fill_timestamp = fill_record['timestamp']
            logger.debug(f"订单完全成交: {self.order_id}, 成交数量: {quantity}, 成交价格: {price}")
        else:
            self.status = self.Status.PARTIALLY_FILLED
            logger.debug(f"订单部分成交: {self.order_id}, 成交数量: {quantity}, 成交价格: {price}, 累计成交: {self.filled_quantity}/{self.quantity}")
        
        # 添加成交记录
        self.fill_records.append(fill_record)
        
        return fill_record
    
    def cancel(self, reason: str = "User cancelled", timestamp: Optional[datetime] = None) -> bool:
        """取消订单
        
        Args:
            reason: 取消原因
            timestamp: 取消时间戳
        
        Returns:
            是否取消成功
        """
        if self.status in [self.Status.FILLED, self.Status.CANCELLED, self.Status.REJECTED, self.Status.EXPIRED]:
            logger.warning(f"无法取消订单: {self.order_id}, 当前状态: {self.status}")
            return False
        
        self.status = self.Status.CANCELLED
        self.last_updated = timestamp or datetime.now()
        self.metadata['cancel_reason'] = reason
        
        logger.debug(f"订单已取消: {self.order_id}, 原因: {reason}")
        return True
    
    def reject(self, reason: str = "Order rejected", timestamp: Optional[datetime] = None) -> bool:
        """拒绝订单
        
        Args:
            reason: 拒绝原因
            timestamp: 拒绝时间戳
        
        Returns:
            是否拒绝成功
        """
        if self.status in [self.Status.FILLED, self.Status.CANCELLED, self.Status.REJECTED, self.Status.EXPIRED]:
            logger.warning(f"无法拒绝订单: {self.order_id}, 当前状态: {self.status}")
            return False
        
        self.status = self.Status.REJECTED
        self.last_updated = timestamp or datetime.now()
        self.metadata['reject_reason'] = reason
        
        logger.debug(f"订单已拒绝: {self.order_id}, 原因: {reason}")
        return True
    
    def expire(self, timestamp: Optional[datetime] = None) -> bool:
        """订单过期
        
        Args:
            timestamp: 过期时间戳
        
        Returns:
            是否过期成功
        """
        if self.status in [self.Status.FILLED, self.Status.CANCELLED, self.Status.REJECTED, self.Status.EXPIRED]:
            logger.warning(f"无法将订单标记为过期: {self.order_id}, 当前状态: {self.status}")
            return False
        
        self.status = self.Status.EXPIRED
        self.last_updated = timestamp or datetime.now()
        
        logger.debug(f"订单已过期: {self.order_id}")
        return True
    
    def update_trailing_stop(self, current_price: float) -> None:
        """更新跟踪止损价格
        
        Args:
            current_price: 当前价格
        """
        if self.order_type != self.Type.TRAILING_STOP:
            logger.warning(f"无法更新非跟踪止损单: {self.order_id}")
            return
        
        if not self.trailing_amount:
            logger.warning(f"跟踪止损单没有设置跟踪金额: {self.order_id}")
            return
        
        # 根据方向更新止损价格
        if self.side == self.Side.BUY:
            # 买入方向的跟踪止损，止损价应该低于当前价格
            new_stop_price = current_price - self.trailing_amount
            # 不允许止损价低于初始设置的价格
            if self.stop_price is None or new_stop_price > self.stop_price:
                self.stop_price = new_stop_price
                self.last_updated = datetime.now()
        else:
            # 卖出方向的跟踪止损，止损价应该高于当前价格
            new_stop_price = current_price + self.trailing_amount
            # 不允许止损价高于初始设置的价格
            if self.stop_price is None or new_stop_price < self.stop_price:
                self.stop_price = new_stop_price
                self.last_updated = datetime.now()
        
        logger.debug(f"跟踪止损价格已更新: {self.order_id}, 新的止损价格: {self.stop_price}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将订单转换为字典
        
        Returns:
            订单字典
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'trailing_amount': self.trailing_amount,
            'time_in_force': self.time_in_force,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'fill_timestamp': self.fill_timestamp.isoformat() if self.fill_timestamp and isinstance(self.fill_timestamp, datetime) else self.fill_timestamp,
            'last_updated': self.last_updated.isoformat() if isinstance(self.last_updated, datetime) else self.last_updated,
            'metadata': self.metadata,
            'fill_records': self.fill_records
        }
    
    @classmethod
    def from_dict(cls, order_dict: Dict[str, Any]) -> 'Order':
        """从字典创建订单
        
        Args:
            order_dict: 订单字典
        
        Returns:
            订单对象
        """
        # 创建订单对象
        order = cls(
            symbol=order_dict['symbol'],
            side=order_dict['side'],
            order_type=order_dict['order_type'],
            quantity=order_dict['quantity'],
            price=order_dict.get('price'),
            stop_price=order_dict.get('stop_price'),
            trailing_amount=order_dict.get('trailing_amount'),
            time_in_force=order_dict.get('time_in_force', 'GTC'),
            order_id=order_dict.get('order_id'),
            timestamp=datetime.fromisoformat(order_dict['timestamp']) if isinstance(order_dict['timestamp'], str) else order_dict['timestamp'],
            metadata=order_dict.get('metadata', {})
        )
        
        # 恢复订单状态
        order.filled_quantity = order_dict.get('filled_quantity', 0.0)
        order.status = order_dict.get('status', cls.Status.PENDING)
        order.fill_timestamp = datetime.fromisoformat(order_dict['fill_timestamp']) if order_dict.get('fill_timestamp') and isinstance(order_dict['fill_timestamp'], str) else order_dict.get('fill_timestamp')
        order.last_updated = datetime.fromisoformat(order_dict['last_updated']) if isinstance(order_dict['last_updated'], str) else order_dict['last_updated']
        order.fill_records = order_dict.get('fill_records', [])
        
        return order
    
    def __str__(self) -> str:
        """订单的字符串表示
        
        Returns:
            订单字符串
        """
        return (f"Order(id={self.order_id}, symbol={self.symbol}, side={self.side}, type={self.order_type}, "
                f"quantity={self.quantity}, filled={self.filled_quantity}, price={self.price}, "
                f"status={self.status})")

class OrderManager:
    """订单管理器
    管理所有订单的生命周期
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化订单管理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        
        # 订单存储
        self.orders: Dict[str, Order] = {}
        
        # 按状态分类的订单
        self.orders_by_status: Dict[str, Dict[str, Order]] = {
            Order.Status.PENDING: {},
            Order.Status.FILLED: {},
            Order.Status.PARTIALLY_FILLED: {},
            Order.Status.CANCELLED: {},
            Order.Status.REJECTED: {},
            Order.Status.EXPIRED: {}
        }
        
        # 按符号分类的订单
        self.orders_by_symbol: Dict[str, Dict[str, Order]] = {}
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("订单管理器初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"order_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def create_order(self, 
                    symbol: str,
                    side: str,
                    order_type: str,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trailing_amount: Optional[float] = None,
                    time_in_force: str = 'GTC',
                    order_id: Optional[str] = None,
                    timestamp: Optional[datetime] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Order:
        """创建新订单
        
        Args:
            symbol: 交易符号
            side: 订单方向
            order_type: 订单类型
            quantity: 订单数量
            price: 价格
            stop_price: 止损价格
            trailing_amount: 跟踪金额
            time_in_force: 有效时间
            order_id: 订单ID
            timestamp: 时间戳
            metadata: 元数据
        
        Returns:
            创建的订单
        """
        try:
            # 创建订单
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                trailing_amount=trailing_amount,
                time_in_force=time_in_force,
                order_id=order_id,
                timestamp=timestamp,
                metadata=metadata
            )
            
            # 添加到订单管理器
            self.add_order(order)
            
            logger.info(f"已创建订单: {order.order_id}, {symbol}, {side}, {quantity}")
            
            return order
        except Exception as e:
            logger.error(f"创建订单时发生异常: {str(e)}")
            raise
    
    def add_order(self, order: Order) -> bool:
        """添加订单到管理器
        
        Args:
            order: 订单对象
        
        Returns:
            是否添加成功
        """
        try:
            # 检查订单ID是否已存在
            if order.order_id in self.orders:
                logger.warning(f"订单ID已存在: {order.order_id}")
                return False
            
            # 添加到主订单字典
            self.orders[order.order_id] = order
            
            # 添加到按状态分类的字典
            if order.status not in self.orders_by_status:
                self.orders_by_status[order.status] = {}
            self.orders_by_status[order.status][order.order_id] = order
            
            # 添加到按符号分类的字典
            if order.symbol not in self.orders_by_symbol:
                self.orders_by_symbol[order.symbol] = {}
            self.orders_by_symbol[order.symbol][order.order_id] = order
            
            return True
        except Exception as e:
            logger.error(f"添加订单时发生异常: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            订单对象，如果不存在返回None
        """
        return self.orders.get(order_id)
    
    def get_orders_by_status(self, status: str) -> List[Order]:
        """获取指定状态的所有订单
        
        Args:
            status: 订单状态
        
        Returns:
            订单列表
        """
        return list(self.orders_by_status.get(status, {}).values())
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """获取指定符号的所有订单
        
        Args:
            symbol: 交易符号
        
        Returns:
            订单列表
        """
        return list(self.orders_by_symbol.get(symbol, {}).values())
    
    def get_all_orders(self) -> List[Order]:
        """获取所有订单
        
        Returns:
            订单列表
        """
        return list(self.orders.values())
    
    def update_order_status(self, order_id: str, new_status: str, reason: Optional[str] = None) -> bool:
        """更新订单状态
        
        Args:
            order_id: 订单ID
            new_status: 新的状态
            reason: 状态变更原因
        
        Returns:
            是否更新成功
        """
        try:
            # 检查订单是否存在
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            old_status = order.status
            
            # 如果状态没有变化，不需要更新
            if old_status == new_status:
                logger.debug(f"订单状态没有变化: {order_id}, 当前状态: {old_status}")
                return True
            
            # 从旧状态分类中移除
            if old_status in self.orders_by_status and order_id in self.orders_by_status[old_status]:
                del self.orders_by_status[old_status][order_id]
            
            # 更新订单状态
            if new_status == Order.Status.CANCELLED:
                success = order.cancel(reason=reason)
            elif new_status == Order.Status.REJECTED:
                success = order.reject(reason=reason)
            elif new_status == Order.Status.EXPIRED:
                success = order.expire()
            else:
                logger.warning(f"不支持的订单状态更新: {new_status}")
                return False
            
            if success:
                # 添加到新状态分类中
                if order.status not in self.orders_by_status:
                    self.orders_by_status[order.status] = {}
                self.orders_by_status[order.status][order_id] = order
                
                logger.info(f"订单状态已更新: {order_id}, 从 {old_status} 到 {order.status}")
            
            return success
        except Exception as e:
            logger.error(f"更新订单状态时发生异常: {str(e)}")
            return False
    
    def fill_order(self, order_id: str, quantity: float, price: float, timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """成交订单
        
        Args:
            order_id: 订单ID
            quantity: 成交数量
            price: 成交价格
            timestamp: 成交时间戳
        
        Returns:
            成交记录，如果失败返回None
        """
        try:
            # 检查订单是否存在
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return None
            
            order = self.orders[order_id]
            old_status = order.status
            
            # 检查订单是否可以成交
            if order.status in [Order.Status.FILLED, Order.Status.CANCELLED, Order.Status.REJECTED, Order.Status.EXPIRED]:
                logger.warning(f"订单无法成交: {order_id}, 当前状态: {order.status}")
                return None
            
            # 执行成交
            fill_record = order.fill(quantity, price, timestamp)
            
            # 更新订单分类
            if old_status in self.orders_by_status and order_id in self.orders_by_status[old_status]:
                del self.orders_by_status[old_status][order_id]
            
            # 添加到新状态分类中
            if order.status not in self.orders_by_status:
                self.orders_by_status[order.status] = {}
            self.orders_by_status[order.status][order_id] = order
            
            logger.info(f"订单成交: {order_id}, 数量: {quantity}, 价格: {price}")
            
            return fill_record
        except Exception as e:
            logger.error(f"成交订单时发生异常: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            reason: 取消原因
        
        Returns:
            是否取消成功
        """
        return self.update_order_status(order_id, Order.Status.CANCELLED, reason)
    
    def reject_order(self, order_id: str, reason: str = "Order rejected") -> bool:
        """拒绝订单
        
        Args:
            order_id: 订单ID
            reason: 拒绝原因
        
        Returns:
            是否拒绝成功
        """
        return self.update_order_status(order_id, Order.Status.REJECTED, reason)
    
    def expire_order(self, order_id: str) -> bool:
        """使订单过期
        
        Args:
            order_id: 订单ID
        
        Returns:
            是否过期成功
        """
        return self.update_order_status(order_id, Order.Status.EXPIRED)
    
    def update_trailing_stop_orders(self, symbol: str, current_price: float) -> List[Order]:
        """更新指定符号的所有跟踪止损订单
        
        Args:
            symbol: 交易符号
            current_price: 当前价格
        
        Returns:
            更新的订单列表
        """
        updated_orders = []
        
        try:
            # 获取指定符号的所有订单
            symbol_orders = self.get_orders_by_symbol(symbol)
            
            for order in symbol_orders:
                # 只处理跟踪止损订单
                if order.order_type == Order.Type.TRAILING_STOP and order.status in [Order.Status.PENDING, Order.Status.PARTIALLY_FILLED]:
                    # 更新跟踪止损价格
                    order.update_trailing_stop(current_price)
                    updated_orders.append(order)
            
            logger.debug(f"已更新 {len(updated_orders)} 个跟踪止损订单: {symbol}")
            
            return updated_orders
        except Exception as e:
            logger.error(f"更新跟踪止损订单时发生异常: {str(e)}")
            return updated_orders
    
    def check_order_expiration(self, current_time: Optional[datetime] = None) -> List[Order]:
        """检查并处理过期订单
        
        Args:
            current_time: 当前时间
        
        Returns:
            过期的订单列表
        """
        expired_orders = []
        
        try:
            current_time = current_time or datetime.now()
            
            # 检查所有待处理和部分成交的订单
            pending_orders = self.get_orders_by_status(Order.Status.PENDING) + \
                            self.get_orders_by_status(Order.Status.PARTIALLY_FILLED)
            
            for order in pending_orders:
                # 检查订单是否过期
                if self._is_order_expired(order, current_time):
                    # 使订单过期
                    if self.expire_order(order.order_id):
                        expired_orders.append(order)
            
            logger.debug(f"已处理 {len(expired_orders)} 个过期订单")
            
            return expired_orders
        except Exception as e:
            logger.error(f"检查订单过期时发生异常: {str(e)}")
            return expired_orders
    
    def _is_order_expired(self, order: Order, current_time: datetime) -> bool:
        """检查订单是否过期
        
        Args:
            order: 订单对象
            current_time: 当前时间
        
        Returns:
            是否过期
        """
        # 根据订单的有效时间策略判断是否过期
        if order.time_in_force == 'GTC':  # Good Till Cancelled
            # 除非被取消，否则一直有效
            return False
        elif order.time_in_force == 'DAY':  # Day Order
            # 当天结束前有效
            if order.timestamp.date() < current_time.date():
                return True
        elif order.time_in_force == 'GTD':  # Good Till Date
            # 直到指定日期前有效
            if 'expiry_date' in order.metadata:
                expiry_date = order.metadata['expiry_date']
                if isinstance(expiry_date, str):
                    expiry_date = datetime.fromisoformat(expiry_date)
                if current_time > expiry_date:
                    return True
        # IOC (Immediate or Cancel) 和 FOK (Fill or Kill) 类型的订单通常在创建后立即处理，这里不做特殊处理
        
        return False
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """获取所有订单的历史记录
        
        Returns:
            订单历史记录列表
        """
        return [order.to_dict() for order in self.orders.values()]
    
    def save_order_history(self, file_path: str) -> bool:
        """保存订单历史记录到文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否保存成功
        """
        try:
            # 获取订单历史记录
            order_history = self.get_order_history()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存到文件
            with open(file_path, 'w') as f:
                json.dump(order_history, f, indent=4, default=str)
            
            logger.info(f"订单历史记录已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存订单历史记录时发生异常: {str(e)}")
            return False
    
    def load_order_history(self, file_path: str) -> bool:
        """从文件加载订单历史记录
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"订单历史记录文件不存在: {file_path}")
                return False
            
            # 从文件加载
            with open(file_path, 'r') as f:
                order_history = json.load(f)
            
            # 清空现有订单
            self.orders.clear()
            for status in self.orders_by_status:
                self.orders_by_status[status].clear()
            self.orders_by_symbol.clear()
            
            # 重建订单
            for order_dict in order_history:
                order = Order.from_dict(order_dict)
                self.add_order(order)
            
            logger.info(f"订单历史记录已从: {file_path} 加载，共 {len(order_history)} 个订单")
            return True
        except Exception as e:
            logger.error(f"加载订单历史记录时发生异常: {str(e)}")
            return False
    
    def get_order_summary(self) -> Dict[str, Any]:
        """获取订单摘要统计
        
        Returns:
            订单摘要统计
        """
        summary = {
            'total_orders': len(self.orders),
            'orders_by_status': {},
            'orders_by_symbol': {},
            'filled_quantity': 0.0,
            'pending_quantity': 0.0
        }
        
        # 按状态统计订单
        for status, orders in self.orders_by_status.items():
            summary['orders_by_status'][status] = len(orders)
        
        # 按符号统计订单和数量
        for symbol, orders in self.orders_by_symbol.items():
            symbol_summary = {
                'count': len(orders),
                'total_quantity': 0.0,
                'filled_quantity': 0.0,
                'pending_quantity': 0.0
            }
            
            for order in orders.values():
                symbol_summary['total_quantity'] += order.quantity
                symbol_summary['filled_quantity'] += order.filled_quantity
                symbol_summary['pending_quantity'] += (order.quantity - order.filled_quantity)
            
            summary['orders_by_symbol'][symbol] = symbol_summary
            
            # 更新全局统计
            summary['filled_quantity'] += symbol_summary['filled_quantity']
            summary['pending_quantity'] += symbol_summary['pending_quantity']
        
        return summary

class PositionManager:
    """头寸管理器
    管理交易账户的所有头寸
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化头寸管理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        
        # 头寸存储
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("头寸管理器初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"position_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def update_position(self, 
                      symbol: str,
                      quantity: float,
                      price: float,
                      timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """更新头寸
        
        Args:
            symbol: 交易符号
            quantity: 交易数量（正数为买入，负数为卖出）
            price: 交易价格
            timestamp: 交易时间戳
        
        Returns:
            更新后的头寸
        """
        try:
            current_time = timestamp or datetime.now()
            
            # 如果符号不存在于头寸字典中，创建新头寸
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': 0.0,
                    'avg_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'total_cost': 0.0,
                    'last_update': current_time,
                    'trade_history': []
                }
            
            position = self.positions[symbol]
            prev_quantity = position['quantity']
            
            # 创建交易记录
            trade_record = {
                'timestamp': current_time,
                'quantity': quantity,
                'price': price,
                'cost': quantity * price
            }
            
            # 更新头寸
            if prev_quantity == 0:
                # 新开仓
                position['quantity'] = quantity
                position['avg_price'] = price
                position['total_cost'] = quantity * price
            elif (prev_quantity > 0 and quantity > 0) or (prev_quantity < 0 and quantity < 0):
                # 加仓
                new_quantity = prev_quantity + quantity
                position['avg_price'] = ((prev_quantity * position['avg_price']) + (quantity * price)) / new_quantity
                position['quantity'] = new_quantity
                position['total_cost'] += quantity * price
            else:
                # 减仓或平仓
                if abs(prev_quantity) > abs(quantity):
                    # 减仓
                    # 计算实现盈亏
                    pnl = abs(quantity) * (price - position['avg_price']) if prev_quantity > 0 else abs(quantity) * (position['avg_price'] - price)
                    position['realized_pnl'] += pnl
                    
                    # 更新头寸
                    position['quantity'] += quantity
                    position['total_cost'] += quantity * price
                else:
                    # 平仓并可能反手
                    # 计算实现盈亏
                    pnl = abs(prev_quantity) * (price - position['avg_price']) if prev_quantity > 0 else abs(prev_quantity) * (position['avg_price'] - price)
                    position['realized_pnl'] += pnl
                    
                    # 计算剩余头寸
                    remaining_quantity = prev_quantity + quantity
                    
                    if remaining_quantity == 0:
                        # 完全平仓
                        position['quantity'] = 0
                        position['avg_price'] = 0
                        position['total_cost'] = 0
                    else:
                        # 平仓后反手
                        position['quantity'] = remaining_quantity
                        position['avg_price'] = price
                        position['total_cost'] = remaining_quantity * price
            
            # 添加交易记录
            position['trade_history'].append(trade_record)
            position['last_update'] = current_time
            
            logger.info(f"头寸已更新: {symbol}, 数量: {quantity}, 价格: {price}, 当前持仓: {position['quantity']}")
            
            return position
        except Exception as e:
            logger.error(f"更新头寸时发生异常: {str(e)}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取指定符号的头寸
        
        Args:
            symbol: 交易符号
        
        Returns:
            头寸信息，如果不存在返回None
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有头寸
        
        Returns:
            所有头寸的字典
        """
        return self.positions
    
    def update_market_values(self, prices: Dict[str, float], timestamp: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """更新所有头寸的市场价值和浮动盈亏
        
        Args:
            prices: 各符号的当前市场价格
            timestamp: 更新时间戳
        
        Returns:
            更新后的所有头寸
        """
        try:
            current_time = timestamp or datetime.now()
            
            for symbol, position in self.positions.items():
                if symbol in prices and position['quantity'] != 0:
                    # 更新市场价值
                    position['market_value'] = position['quantity'] * prices[symbol]
                    
                    # 更新浮动盈亏
                    position['unrealized_pnl'] = position['quantity'] * (prices[symbol] - position['avg_price'])
                    
                    # 更新最后更新时间
                    position['last_update'] = current_time
            
            logger.debug(f"已更新 {len(prices)} 个符号的市场价值")
            
            return self.positions
        except Exception as e:
            logger.error(f"更新市场价值时发生异常: {str(e)}")
            return self.positions
    
    def close_position(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> bool:
        """平仓指定符号的所有头寸
        
        Args:
            symbol: 交易符号
            price: 平仓价格
            timestamp: 平仓时间戳
        
        Returns:
            是否平仓成功
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"没有找到 {symbol} 的头寸")
                return False
            
            position = self.positions[symbol]
            if position['quantity'] == 0:
                logger.warning(f"{symbol} 的头寸数量为0，无需平仓")
                return True
            
            # 计算平仓数量（与当前头寸数量相反）
            close_quantity = -position['quantity']
            
            # 更新头寸
            self.update_position(symbol, close_quantity, price, timestamp)
            
            logger.info(f"已平仓: {symbol}, 数量: {close_quantity}, 价格: {price}")
            
            return True
        except Exception as e:
            logger.error(f"平仓时发生异常: {str(e)}")
            return False
    
    def close_all_positions(self, prices: Dict[str, float], timestamp: Optional[datetime] = None) -> Dict[str, bool]:
        """平仓所有头寸
        
        Args:
            prices: 各符号的当前市场价格
            timestamp: 平仓时间戳
        
        Returns:
            各符号的平仓结果
        """
        results = {}
        
        try:
            # 获取所有有头寸的符号
            symbols = list(self.positions.keys())
            
            for symbol in symbols:
                if symbol in prices:
                    results[symbol] = self.close_position(symbol, prices[symbol], timestamp)
                else:
                    logger.warning(f"没有找到 {symbol} 的价格，无法平仓")
                    results[symbol] = False
            
            logger.info(f"已尝试平仓所有头寸，共 {len(symbols)} 个符号")
            
            return results
        except Exception as e:
            logger.error(f"平仓所有头寸时发生异常: {str(e)}")
            return results
    
    def get_position_summary(self) -> Dict[str, Any]:
        """获取头寸摘要统计
        
        Returns:
            头寸摘要统计
        """
        summary = {
            'total_symbols': len(self.positions),
            'long_positions': 0,
            'short_positions': 0,
            'total_quantity': 0.0,
            'total_market_value': 0.0,
            'total_unrealized_pnl': 0.0,
            'total_realized_pnl': 0.0,
            'positions': {}
        }
        
        for symbol, position in self.positions.items():
            # 复制头寸信息，不包含交易历史
            position_summary = position.copy()
            if 'trade_history' in position_summary:
                del position_summary['trade_history']
            
            summary['positions'][symbol] = position_summary
            
            # 更新统计信息
            if position['quantity'] > 0:
                summary['long_positions'] += 1
            elif position['quantity'] < 0:
                summary['short_positions'] += 1
            
            summary['total_quantity'] += abs(position['quantity'])
            summary['total_market_value'] += abs(position['market_value'])
            summary['total_unrealized_pnl'] += position['unrealized_pnl']
            summary['total_realized_pnl'] += position['realized_pnl']
        
        return summary
    
    def save_positions(self, file_path: str) -> bool:
        """保存头寸信息到文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否保存成功
        """
        try:
            # 获取头寸摘要
            positions_summary = self.get_position_summary()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存到文件
            with open(file_path, 'w') as f:
                json.dump(positions_summary, f, indent=4, default=str)
            
            logger.info(f"头寸信息已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存头寸信息时发生异常: {str(e)}")
            return False
    
    def load_positions(self, file_path: str) -> bool:
        """从文件加载头寸信息
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"头寸信息文件不存在: {file_path}")
                return False
            
            # 从文件加载
            with open(file_path, 'r') as f:
                positions_data = json.load(f)
            
            # 重建头寸
            if 'positions' in positions_data:
                self.positions = positions_data['positions']
                
                # 转换时间戳字符串为datetime对象
                for symbol, position in self.positions.items():
                    if 'last_update' in position and isinstance(position['last_update'], str):
                        try:
                            position['last_update'] = datetime.fromisoformat(position['last_update'])
                        except:
                            pass
            
            logger.info(f"头寸信息已从: {file_path} 加载，共 {len(self.positions)} 个符号")
            return True
        except Exception as e:
            logger.error(f"加载头寸信息时发生异常: {str(e)}")
            return False
    
    def reset(self) -> None:
        """重置头寸管理器"""
        self.positions.clear()
        logger.info("头寸管理器已重置")

# 模块版本
__version__ = '0.1.0'