import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import os
import pickle
from abc import ABC, abstractmethod
import json

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataHandler(ABC):
    """数据处理器抽象基类
    提供回测所需的数据加载、预处理和访问功能
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化数据处理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        self.data = {}
        self.symbol_info = {}
        self.data_loaded = False
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("DataHandler 初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"data_handler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    @abstractmethod
    def load_data(self, **kwargs) -> bool:
        """加载数据"""
        pass
    
    @abstractmethod
    def get_data(self, 
                symbol: str,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取数据"""
        pass
    
    @abstractmethod
    def get_latest_data(self, 
                       symbol: str,
                       n: int = 1,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取最新数据"""
        pass
    
    @abstractmethod
    def update_data(self, **kwargs) -> bool:
        """更新数据"""
        pass

class CSVDataHandler(DataHandler):
    """CSV文件数据处理器
    从CSV文件加载和处理数据
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化CSV数据处理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        super().__init__(config, log_level)
        
        # 设置默认配置
        self.data_dir = self.config.get('data_dir', './data')
        self.file_pattern = self.config.get('file_pattern', '{symbol}.csv')
        self.date_column = self.config.get('date_column', 'date')
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        self.price_columns = self.config.get('price_columns', ['open', 'high', 'low', 'close', 'volume'])
        self.auto_index = self.config.get('auto_index', True)
        
        logger.info("CSVDataHandler 初始化完成")
    
    def load_data(self, 
                 symbols: List[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 **kwargs) -> bool:
        """从CSV文件加载数据
        
        Args:
            symbols: 要加载的符号列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        
        Returns:
            是否加载成功
        """
        try:
            if symbols is None:
                # 如果没有指定符号，加载data_dir中的所有CSV文件
                symbols = []
                for file in os.listdir(self.data_dir):
                    if file.endswith('.csv'):
                        # 假设文件名就是符号名
                        symbol = file[:-4]  # 移除.csv后缀
                        symbols.append(symbol)
            
            if not symbols:
                logger.warning("没有要加载的符号")
                return False
            
            logger.info(f"开始加载数据，符号数量: {len(symbols)}")
            
            for symbol in symbols:
                # 构建文件路径
                file_path = os.path.join(self.data_dir, self.file_pattern.format(symbol=symbol))
                
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                
                # 读取CSV文件
                df = pd.read_csv(file_path, **kwargs)
                
                # 处理日期列
                if self.date_column in df.columns:
                    df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
                    
                    # 设置日期索引
                    if self.auto_index:
                        df.set_index(self.date_column, inplace=True)
                
                # 应用日期过滤
                if start_date and self.auto_index:
                    df = df[df.index >= start_date]
                if end_date and self.auto_index:
                    df = df[df.index <= end_date]
                
                # 保存数据
                self.data[symbol] = df
                
                # 保存符号信息
                self.symbol_info[symbol] = {
                    'start_date': df.index.min() if self.auto_index else None,
                    'end_date': df.index.max() if self.auto_index else None,
                    'columns': df.columns.tolist(),
                    'n_rows': len(df)
                }
                
                logger.info(f"已加载数据: {symbol}, 行数: {len(df)}")
            
            self.data_loaded = True
            
            logger.info("数据加载完成")
            return True
        except Exception as e:
            logger.error(f"加载数据时发生异常: {str(e)}")
            return False
    
    def get_data(self, 
                symbol: str,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取指定符号的数据
        
        Args:
            symbol: 符号
            start_date: 开始日期
            end_date: 结束日期
            columns: 列名列表
        
        Returns:
            数据DataFrame
        """
        try:
            # 检查符号是否存在
            if symbol not in self.data:
                logger.warning(f"没有找到符号的数据: {symbol}")
                # 尝试动态加载
                load_success = self.load_data(symbols=[symbol])
                if not load_success or symbol not in self.data:
                    return pd.DataFrame()
            
            df = self.data[symbol].copy()
            
            # 应用日期过滤
            if start_date and isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index >= start_date]
            if end_date and isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index <= end_date]
            
            # 应用列过滤
            if columns:
                # 只保留存在的列
                valid_columns = [col for col in columns if col in df.columns]
                df = df[valid_columns]
            
            return df
        except Exception as e:
            logger.error(f"获取数据时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data(self, 
                       symbol: str,
                       n: int = 1,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取指定符号的最新数据
        
        Args:
            symbol: 符号
            n: 数据条数
            columns: 列名列表
        
        Returns:
            最新数据DataFrame
        """
        try:
            # 获取完整数据
            df = self.get_data(symbol, columns=columns)
            
            if df.empty:
                return df
            
            # 返回最新的n行
            return df.tail(n)
        except Exception as e:
            logger.error(f"获取最新数据时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def update_data(self, 
                   symbol: str,
                   new_data: pd.DataFrame,
                   append: bool = True,
                   **kwargs) -> bool:
        """更新数据
        
        Args:
            symbol: 符号
            new_data: 新数据
            append: 是否追加到现有数据
            **kwargs: 其他参数
        
        Returns:
            是否更新成功
        """
        try:
            if symbol not in self.data or not append:
                # 如果符号不存在或不追加，则直接替换
                self.data[symbol] = new_data
            else:
                # 追加数据
                # 确保索引兼容
                if isinstance(self.data[symbol].index, pd.DatetimeIndex) and isinstance(new_data.index, pd.DatetimeIndex):
                    # 移除重复索引
                    combined = pd.concat([self.data[symbol], new_data]).drop_duplicates()
                    # 按索引排序
                    combined.sort_index(inplace=True)
                    self.data[symbol] = combined
                else:
                    # 简单追加
                    self.data[symbol] = pd.concat([self.data[symbol], new_data])
            
            # 更新符号信息
            self.symbol_info[symbol] = {
                'start_date': self.data[symbol].index.min() if isinstance(self.data[symbol].index, pd.DatetimeIndex) else None,
                'end_date': self.data[symbol].index.max() if isinstance(self.data[symbol].index, pd.DatetimeIndex) else None,
                'columns': self.data[symbol].columns.tolist(),
                'n_rows': len(self.data[symbol])
            }
            
            logger.info(f"数据已更新: {symbol}, 新行数: {len(self.data[symbol])}")
            
            return True
        except Exception as e:
            logger.error(f"更新数据时发生异常: {str(e)}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """获取所有可用的符号
        
        Returns:
            符号列表
        """
        return list(self.data.keys())
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取符号信息
        
        Args:
            symbol: 符号
        
        Returns:
            符号信息字典
        """
        return self.symbol_info.get(symbol)
    
    def save_data(self, 
                 symbols: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 file_format: str = 'csv',
                 **kwargs) -> Dict[str, str]:
        """保存数据
        
        Args:
            symbols: 要保存的符号列表
            output_dir: 输出目录
            file_format: 文件格式
            **kwargs: 其他参数
        
        Returns:
            保存的文件路径字典
        """
        try:
            if symbols is None:
                symbols = self.get_available_symbols()
            
            if output_dir is None:
                output_dir = self.data_dir
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            saved_files = {}
            
            for symbol in symbols:
                if symbol not in self.data:
                    logger.warning(f"没有找到符号的数据: {symbol}")
                    continue
                
                # 构建文件路径
                file_path = os.path.join(output_dir, f"{symbol}.{file_format}")
                
                # 根据文件格式保存
                if file_format.lower() == 'csv':
                    self.data[symbol].to_csv(file_path, **kwargs)
                elif file_format.lower() == 'pickle':
                    self.data[symbol].to_pickle(file_path, **kwargs)
                elif file_format.lower() == 'json':
                    self.data[symbol].to_json(file_path, **kwargs)
                else:
                    logger.warning(f"不支持的文件格式: {file_format}")
                    continue
                
                saved_files[symbol] = file_path
                
                logger.info(f"数据已保存: {file_path}")
            
            return saved_files
        except Exception as e:
            logger.error(f"保存数据时发生异常: {str(e)}")
            return {}
    
    def clear_data(self, symbols: Optional[List[str]] = None) -> bool:
        """清除数据
        
        Args:
            symbols: 要清除的符号列表
        
        Returns:
            是否清除成功
        """
        try:
            if symbols is None:
                # 清除所有数据
                self.data.clear()
                self.symbol_info.clear()
                self.data_loaded = False
            else:
                # 清除指定符号的数据
                for symbol in symbols:
                    if symbol in self.data:
                        del self.data[symbol]
                    if symbol in self.symbol_info:
                        del self.symbol_info[symbol]
                
                # 更新数据加载状态
                self.data_loaded = len(self.data) > 0
            
            logger.info("数据已清除")
            return True
        except Exception as e:
            logger.error(f"清除数据时发生异常: {str(e)}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要
        
        Returns:
            数据摘要字典
        """
        summary = {
            'total_symbols': len(self.data),
            'symbols': {},
            'start_date': None,
            'end_date': None
        }
        
        # 计算全局开始和结束日期
        all_start_dates = []
        all_end_dates = []
        
        for symbol, info in self.symbol_info.items():
            summary['symbols'][symbol] = {
                'n_rows': info['n_rows'],
                'n_columns': len(info['columns']),
                'columns': info['columns'],
                'start_date': info['start_date'],
                'end_date': info['end_date']
            }
            
            if info['start_date']:
                all_start_dates.append(info['start_date'])
            if info['end_date']:
                all_end_dates.append(info['end_date'])
        
        if all_start_dates:
            summary['start_date'] = min(all_start_dates)
        if all_end_dates:
            summary['end_date'] = max(all_end_dates)
        
        return summary

class DatabaseDataHandler(DataHandler):
    """数据库数据处理器
    从数据库加载和处理数据
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化数据库数据处理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        super().__init__(config, log_level)
        
        # 设置默认配置
        self.db_type = self.config.get('db_type', 'sqlite')
        self.connection_string = self.config.get('connection_string', 'sqlite:///data.db')
        self.table_pattern = self.config.get('table_pattern', 'ohlcv_{symbol}')
        self.date_column = self.config.get('date_column', 'date')
        
        # 数据库连接
        self.connection = None
        
        logger.info("DatabaseDataHandler 初始化完成")
    
    def _connect(self) -> bool:
        """连接到数据库
        
        Returns:
            是否连接成功
        """
        try:
            if self.db_type.lower() == 'sqlite':
                import sqlite3
                self.connection = sqlite3.connect(self.connection_string)
            elif self.db_type.lower() in ['postgres', 'postgresql']:
                import psycopg2
                self.connection = psycopg2.connect(self.connection_string)
            elif self.db_type.lower() == 'mysql':
                import pymysql
                self.connection = pymysql.connect(self.connection_string)
            elif self.db_type.lower() == 'mssql':
                import pyodbc
                self.connection = pyodbc.connect(self.connection_string)
            else:
                logger.error(f"不支持的数据库类型: {self.db_type}")
                return False
            
            logger.info(f"已连接到数据库: {self.db_type}")
            return True
        except Exception as e:
            logger.error(f"连接数据库时发生异常: {str(e)}")
            return False
    
    def _disconnect(self):
        """断开数据库连接"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("已断开数据库连接")
            except Exception as e:
                logger.error(f"断开数据库连接时发生异常: {str(e)}")
            finally:
                self.connection = None
    
    def load_data(self, 
                 symbols: List[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 **kwargs) -> bool:
        """从数据库加载数据
        
        Args:
            symbols: 要加载的符号列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        
        Returns:
            是否加载成功
        """
        try:
            # 连接数据库
            if not self._connect():
                return False
            
            if symbols is None:
                # 如果没有指定符号，尝试从数据库获取所有表
                symbols = []
                
                if self.db_type.lower() == 'sqlite':
                    query = "SELECT name FROM sqlite_master WHERE type='table';"
                    tables = pd.read_sql_query(query, self.connection)
                    for _, row in tables.iterrows():
                        table_name = row['name']
                        # 尝试提取符号名
                        if table_name.startswith('ohlcv_'):
                            symbol = table_name[6:]
                            symbols.append(symbol)
                else:
                    logger.warning("自动检测符号仅支持SQLite数据库")
                    return False
            
            if not symbols:
                logger.warning("没有要加载的符号")
                self._disconnect()
                return False
            
            logger.info(f"开始加载数据，符号数量: {len(symbols)}")
            
            for symbol in symbols:
                # 构建表名
                table_name = self.table_pattern.format(symbol=symbol)
                
                # 构建查询SQL
                query = f"SELECT * FROM {table_name}"
                
                # 添加日期过滤
                conditions = []
                if start_date:
                    conditions.append(f"{self.date_column} >= '{start_date.strftime('%Y-%m-%d')}'")
                if end_date:
                    conditions.append(f"{self.date_column} <= '{end_date.strftime('%Y-%m-%d')}'")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                # 添加排序
                query += f" ORDER BY {self.date_column}"
                
                try:
                    # 执行查询
                    df = pd.read_sql_query(query, self.connection)
                    
                    # 处理日期列
                    if self.date_column in df.columns:
                        df[self.date_column] = pd.to_datetime(df[self.date_column])
                        df.set_index(self.date_column, inplace=True)
                    
                    # 保存数据
                    self.data[symbol] = df
                    
                    # 保存符号信息
                    self.symbol_info[symbol] = {
                        'start_date': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                        'end_date': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
                        'columns': df.columns.tolist(),
                        'n_rows': len(df)
                    }
                    
                    logger.info(f"已加载数据: {symbol}, 行数: {len(df)}")
                except Exception as e:
                    logger.warning(f"加载符号 {symbol} 的数据时发生异常: {str(e)}")
                    continue
            
            self.data_loaded = True
            
            # 断开数据库连接
            self._disconnect()
            
            logger.info("数据加载完成")
            return True
        except Exception as e:
            logger.error(f"加载数据时发生异常: {str(e)}")
            # 确保断开连接
            self._disconnect()
            return False
    
    def get_data(self, 
                symbol: str,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取指定符号的数据
        
        Args:
            symbol: 符号
            start_date: 开始日期
            end_date: 结束日期
            columns: 列名列表
        
        Returns:
            数据DataFrame
        """
        try:
            # 检查符号是否已经加载
            if symbol not in self.data:
                logger.warning(f"符号 {symbol} 的数据未加载，尝试动态加载")
                # 动态加载该符号的数据
                load_success = self.load_data(symbols=[symbol], start_date=start_date, end_date=end_date)
                if not load_success or symbol not in self.data:
                    return pd.DataFrame()
            
            df = self.data[symbol].copy()
            
            # 应用日期过滤
            if start_date and isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index >= start_date]
            if end_date and isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index <= end_date]
            
            # 应用列过滤
            if columns:
                # 只保留存在的列
                valid_columns = [col for col in columns if col in df.columns]
                df = df[valid_columns]
            
            return df
        except Exception as e:
            logger.error(f"获取数据时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data(self, 
                       symbol: str,
                       n: int = 1,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取指定符号的最新数据
        
        Args:
            symbol: 符号
            n: 数据条数
            columns: 列名列表
        
        Returns:
            最新数据DataFrame
        """
        try:
            # 获取完整数据
            df = self.get_data(symbol, columns=columns)
            
            if df.empty:
                return df
            
            # 返回最新的n行
            return df.tail(n)
        except Exception as e:
            logger.error(f"获取最新数据时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def update_data(self, 
                   symbol: str,
                   new_data: pd.DataFrame,
                   append: bool = True,
                   **kwargs) -> bool:
        """更新数据到数据库
        
        Args:
            symbol: 符号
            new_data: 新数据
            append: 是否追加到现有数据
            **kwargs: 其他参数
        
        Returns:
            是否更新成功
        """
        try:
            # 连接数据库
            if not self._connect():
                return False
            
            # 构建表名
            table_name = self.table_pattern.format(symbol=symbol)
            
            # 如果需要追加，先获取现有数据的最新日期
            if append:
                # 检查表是否存在
                if self.db_type.lower() == 'sqlite':
                    check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
                    table_exists = not pd.read_sql_query(check_query, self.connection).empty
                else:
                    # 对于其他数据库类型，使用try-except判断表是否存在
                    try:
                        pd.read_sql_query(f"SELECT 1 FROM {table_name} LIMIT 1", self.connection)
                        table_exists = True
                    except:
                        table_exists = False
                
                if table_exists:
                    # 只插入新数据
                    if isinstance(new_data.index, pd.DatetimeIndex):
                        # 获取现有数据的最新日期
                        latest_date_query = f"SELECT MAX({self.date_column}) as latest_date FROM {table_name}"
                        latest_date_result = pd.read_sql_query(latest_date_query, self.connection)
                        if not latest_date_result.empty and pd.notna(latest_date_result['latest_date'].iloc[0]):
                            latest_date = pd.to_datetime(latest_date_result['latest_date'].iloc[0])
                            # 过滤出新数据
                            new_data = new_data[new_data.index > latest_date]
            
            # 准备数据插入
            if not new_data.empty:
                # 确保数据有日期列
                if isinstance(new_data.index, pd.DatetimeIndex):
                    df_to_insert = new_data.reset_index()
                else:
                    df_to_insert = new_data.copy()
                
                # 写入数据库
                df_to_insert.to_sql(table_name, self.connection, if_exists='append', index=False, **kwargs)
                
                logger.info(f"已将 {len(df_to_insert)} 行数据写入数据库表 {table_name}")
                
                # 更新本地缓存
                if symbol in self.data:
                    if append:
                        # 追加到本地缓存
                        if isinstance(self.data[symbol].index, pd.DatetimeIndex) and isinstance(new_data.index, pd.DatetimeIndex):
                            combined = pd.concat([self.data[symbol], new_data]).drop_duplicates()
                            combined.sort_index(inplace=True)
                            self.data[symbol] = combined
                        else:
                            self.data[symbol] = pd.concat([self.data[symbol], new_data])
                    else:
                        # 替换本地缓存
                        self.data[symbol] = new_data
                else:
                    # 添加到本地缓存
                    self.data[symbol] = new_data
                
                # 更新符号信息
                self.symbol_info[symbol] = {
                    'start_date': self.data[symbol].index.min() if isinstance(self.data[symbol].index, pd.DatetimeIndex) else None,
                    'end_date': self.data[symbol].index.max() if isinstance(self.data[symbol].index, pd.DatetimeIndex) else None,
                    'columns': self.data[symbol].columns.tolist(),
                    'n_rows': len(self.data[symbol])
                }
            else:
                logger.info(f"没有新数据需要写入数据库表 {table_name}")
            
            # 断开数据库连接
            self._disconnect()
            
            logger.info(f"数据已更新: {symbol}")
            
            return True
        except Exception as e:
            logger.error(f"更新数据时发生异常: {str(e)}")
            # 确保断开连接
            self._disconnect()
            return False

class DataProcessor:
    """数据处理器
    提供数据预处理和特征工程功能
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化数据处理器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("DataProcessor 初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"data_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def resample_data(self, 
                     df: pd.DataFrame,
                     freq: str = 'D',
                     **kwargs) -> pd.DataFrame:
        """重采样数据
        
        Args:
            df: 原始数据
            freq: 重采样频率
            **kwargs: 其他参数
        
        Returns:
            重采样后的数据
        """
        try:
            # 检查数据是否有日期索引
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("数据索引不是日期类型")
                return df
            
            # 定义重采样规则
            resample_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # 应用用户自定义的重采样规则
            if kwargs:
                resample_rules.update(kwargs)
            
            # 执行重采样
            resampled_df = df.resample(freq).agg(resample_rules)
            
            # 删除包含NaN的行
            resampled_df.dropna(inplace=True)
            
            logger.info(f"数据已重采样到 {freq} 频率")
            
            return resampled_df
        except Exception as e:
            logger.error(f"重采样数据时发生异常: {str(e)}")
            return df
    
    def calculate_returns(self, 
                         df: pd.DataFrame,
                         price_column: str = 'close',
                         periods: int = 1,
                         log_returns: bool = False) -> pd.DataFrame:
        """计算收益率
        
        Args:
            df: 原始数据
            price_column: 价格列名
            periods: 计算收益率的周期数
            log_returns: 是否计算对数收益率
        
        Returns:
            添加了收益率列的数据
        """
        try:
            # 检查价格列是否存在
            if price_column not in df.columns:
                logger.error(f"价格列 {price_column} 不存在")
                return df
            
            # 复制数据以避免修改原始数据
            result_df = df.copy()
            
            # 计算收益率
            if log_returns:
                # 对数收益率
                result_df[f'log_return_{periods}'] = np.log(result_df[price_column] / 
                                                         result_df[price_column].shift(periods))
            else:
                # 简单收益率
                result_df[f'return_{periods}'] = result_df[price_column].pct_change(periods)
            
            logger.info(f"已计算 {periods} 期收益率")
            
            return result_df
        except Exception as e:
            logger.error(f"计算收益率时发生异常: {str(e)}")
            return df
    
    def calculate_technical_indicators(self, 
                                     df: pd.DataFrame,
                                     indicators: List[str] = None,
                                     **kwargs) -> pd.DataFrame:
        """计算技术指标
        
        Args:
            df: 原始数据
            indicators: 要计算的技术指标列表
            **kwargs: 其他参数
        
        Returns:
            添加了技术指标列的数据
        """
        try:
            # 设置默认指标
            if indicators is None:
                indicators = ['sma', 'ema', 'rsi', 'macd', 'bb']
            
            # 复制数据以避免修改原始数据
            result_df = df.copy()
            
            # 计算每个指标
            for indicator in indicators:
                if indicator.lower() == 'sma':
                    # 简单移动平均线
                    periods = kwargs.get('sma_periods', [5, 10, 20, 50, 200])
                    for period in periods:
                        result_df[f'sma_{period}'] = result_df['close'].rolling(window=period).mean()
                elif indicator.lower() == 'ema':
                    # 指数移动平均线
                    periods = kwargs.get('ema_periods', [5, 10, 20, 50, 200])
                    for period in periods:
                        result_df[f'ema_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
                elif indicator.lower() == 'rsi':
                    # 相对强弱指标
                    period = kwargs.get('rsi_period', 14)
                    delta = result_df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    result_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                elif indicator.lower() == 'macd':
                    # MACD指标
                    fast_period = kwargs.get('macd_fast_period', 12)
                    slow_period = kwargs.get('macd_slow_period', 26)
                    signal_period = kwargs.get('macd_signal_period', 9)
                    
                    result_df['macd_line'] = result_df['close'].ewm(span=fast_period, adjust=False).mean() - \
                                          result_df['close'].ewm(span=slow_period, adjust=False).mean()
                    result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal_period, adjust=False).mean()
                    result_df['macd_hist'] = result_df['macd_line'] - result_df['macd_signal']
                elif indicator.lower() == 'bb':
                    # 布林带
                    period = kwargs.get('bb_period', 20)
                    std_dev = kwargs.get('bb_std_dev', 2)
                    
                    result_df[f'bb_middle_{period}'] = result_df['close'].rolling(window=period).mean()
                    result_df[f'bb_upper_{period}'] = result_df[f'bb_middle_{period}'] + \
                                                   (result_df['close'].rolling(window=period).std() * std_dev)
                    result_df[f'bb_lower_{period}'] = result_df[f'bb_middle_{period}'] - \
                                                   (result_df['close'].rolling(window=period).std() * std_dev)
                elif indicator.lower() == 'atr':
                    # 平均真实波动幅度
                    period = kwargs.get('atr_period', 14)
                    
                    # 计算真实波动幅度
                    result_df['tr'] = np.maximum(
                        result_df['high'] - result_df['low'],
                        np.maximum(
                            abs(result_df['high'] - result_df['close'].shift(1)),
                            abs(result_df['low'] - result_df['close'].shift(1))
                        )
                    )
                    result_df[f'atr_{period}'] = result_df['tr'].rolling(window=period).mean()
                    # 删除临时列
                    result_df.drop('tr', axis=1, inplace=True)
                elif indicator.lower() == 'obv':
                    # 能量潮指标
                    result_df['obv'] = (np.sign(result_df['close'].diff()) * result_df['volume']).fillna(0).cumsum()
                else:
                    logger.warning(f"不支持的技术指标: {indicator}")
            
            logger.info(f"已计算技术指标: {', '.join(indicators)}")
            
            return result_df
        except Exception as e:
            logger.error(f"计算技术指标时发生异常: {str(e)}")
            return df
    
    def normalize_data(self, 
                      df: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'minmax') -> pd.DataFrame:
        """标准化数据
        
        Args:
            df: 原始数据
            columns: 要标准化的列
            method: 标准化方法
        
        Returns:
            标准化后的数据
        """
        try:
            # 如果没有指定列，默认处理所有数值列
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 验证列是否存在
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                logger.warning("没有有效的数值列可标准化")
                return df
            
            # 复制数据以避免修改原始数据
            result_df = df.copy()
            
            # 执行标准化
            for col in valid_columns:
                if method.lower() == 'minmax':
                    # 最小-最大标准化
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    if max_val - min_val > 0:
                        result_df[f'{col}_norm'] = (result_df[col] - min_val) / (max_val - min_val)
                    else:
                        result_df[f'{col}_norm'] = 0
                elif method.lower() == 'zscore':
                    # Z-score标准化
                    mean_val = result_df[col].mean()
                    std_val = result_df[col].std()
                    if std_val > 0:
                        result_df[f'{col}_zscore'] = (result_df[col] - mean_val) / std_val
                    else:
                        result_df[f'{col}_zscore'] = 0
                elif method.lower() == 'robust':
                    # 稳健标准化（基于中位数和四分位数范围）
                    median_val = result_df[col].median()
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        result_df[f'{col}_robust'] = (result_df[col] - median_val) / iqr
                    else:
                        result_df[f'{col}_robust'] = 0
                else:
                    logger.warning(f"不支持的标准化方法: {method}")
            
            logger.info(f"已使用 {method} 方法标准化数据列: {', '.join(valid_columns)}")
            
            return result_df
        except Exception as e:
            logger.error(f"标准化数据时发生异常: {str(e)}")
            return df
    
    def fill_missing_values(self, 
                           df: pd.DataFrame,
                           columns: List[str] = None,
                           method: str = 'ffill',
                           **kwargs) -> pd.DataFrame:
        """填充缺失值
        
        Args:
            df: 原始数据
            columns: 要填充的列
            method: 填充方法
            **kwargs: 其他参数
        
        Returns:
            填充后的数据
        """
        try:
            # 如果没有指定列，默认处理所有数值列
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 验证列是否存在
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                logger.warning("没有有效的数值列可填充")
                return df
            
            # 复制数据以避免修改原始数据
            result_df = df.copy()
            
            # 执行填充
            for col in valid_columns:
                if method.lower() == 'ffill':
                    # 前向填充
                    result_df[col] = result_df[col].fillna(method='ffill')
                elif method.lower() == 'bfill':
                    # 后向填充
                    result_df[col] = result_df[col].fillna(method='bfill')
                elif method.lower() == 'mean':
                    # 均值填充
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif method.lower() == 'median':
                    # 中位数填充
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                elif method.lower() == 'interpolate':
                    # 插值填充
                    result_df[col] = result_df[col].interpolate(**kwargs)
                elif method.lower() == 'value':
                    # 固定值填充
                    fill_value = kwargs.get('fill_value', 0)
                    result_df[col] = result_df[col].fillna(fill_value)
                else:
                    logger.warning(f"不支持的填充方法: {method}")
            
            # 再次检查是否还有缺失值，如果有，使用默认方法填充
            for col in valid_columns:
                if result_df[col].isnull().any():
                    logger.warning(f"列 {col} 仍有缺失值，使用前向填充")
                    result_df[col] = result_df[col].fillna(method='ffill')
                    # 如果还有缺失值，使用后向填充
                    if result_df[col].isnull().any():
                        result_df[col] = result_df[col].fillna(method='bfill')
            
            logger.info(f"已填充数据缺失值，方法: {method}")
            
            return result_df
        except Exception as e:
            logger.error(f"填充缺失值时发生异常: {str(e)}")
            return df
    
    def detect_and_remove_outliers(self, 
                                 df: pd.DataFrame,
                                 columns: List[str] = None,
                                 method: str = 'zscore',
                                 threshold: float = 3.0) -> pd.DataFrame:
        """检测并移除异常值
        
        Args:
            df: 原始数据
            columns: 要检测的列
            method: 异常值检测方法
            threshold: 阈值
        
        Returns:
            移除异常值后的数据
        """
        try:
            # 如果没有指定列，默认处理所有数值列
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 验证列是否存在
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                logger.warning("没有有效的数值列可检测异常值")
                return df
            
            # 复制数据以避免修改原始数据
            result_df = df.copy()
            
            # 标记异常值
            outliers_mask = pd.Series([False] * len(result_df), index=result_df.index)
            
            for col in valid_columns:
                if method.lower() == 'zscore':
                    # Z-score方法
                    z_scores = (result_df[col] - result_df[col].mean()) / result_df[col].std()
                    col_outliers = abs(z_scores) > threshold
                elif method.lower() == 'iqr':
                    # IQR方法
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    col_outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                elif method.lower() == 'mad':
                    # MAD方法（基于中位数绝对偏差）
                    median = result_df[col].median()
                    mad = (result_df[col] - median).abs().median()
                    modified_z_scores = 0.6745 * (result_df[col] - median) / mad if mad > 0 else 0
                    col_outliers = abs(modified_z_scores) > threshold
                else:
                    logger.warning(f"不支持的异常值检测方法: {method}")
                    continue
                
                # 更新异常值掩码
                outliers_mask = outliers_mask | col_outliers
            
            # 计算异常值数量
            num_outliers = outliers_mask.sum()
            
            # 移除异常值
            if num_outliers > 0:
                result_df = result_df[~outliers_mask]
                logger.info(f"已移除 {num_outliers} 个异常值，占总数据的 {(num_outliers/len(df)*100):.2f}%")
            else:
                logger.info("没有检测到异常值")
            
            return result_df
        except Exception as e:
            logger.error(f"检测并移除异常值时发生异常: {str(e)}")
            return df

# 模块版本
__version__ = '0.1.0'