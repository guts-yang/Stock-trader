"""数据库管理器模块，负责数据的本地存储和读取"""

import pandas as pd
import sqlite3
import logging
import os
from typing import Dict, List, Optional, Union
import json

class DatabaseManager:
    """数据库管理器类，负责与SQLite数据库交互，包括数据的存储、读取和更新"""
    
    def __init__(self, config: Dict):
        """初始化数据库管理器
        
        Args:
            config: 配置字典，包含数据库相关配置
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.db_path = config['database'].get('path', '../data/stock_data.db')
        self.echo = config['database'].get('echo', False)
        
        # 确保数据库文件所在目录存在
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            self.logger.info(f"创建数据库目录: {db_dir}")
        
        # 连接数据库
        self.conn = None
        self.connect()
        
        # 初始化表结构
        self._init_tables()
    
    def connect(self):
        """连接到SQLite数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info(f"成功连接到数据库: {self.db_path}")
        except Exception as e:
            self.logger.error(f"连接数据库失败: {str(e)}")
            self.conn = None
    
    def disconnect(self):
        """断开数据库连接"""
        if self.conn is not None:
            self.conn.close()
            self.logger.info("数据库连接已断开")
            self.conn = None
    
    def _init_tables(self):
        """初始化数据库表结构"""
        if self.conn is None:
            self.logger.error("数据库未连接，无法初始化表结构")
            return
        
        try:
            cursor = self.conn.cursor()
            
            # 创建股票基础信息表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_basic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT NOT NULL,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                list_date TEXT,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ts_code)
            )
            ''')
            
            # 创建日线行情表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                volume INTEGER,
                amount REAL,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ts_code, trade_date)
            )
            ''')
            
            # 创建基本面数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT NOT NULL,
                ann_date TEXT,
                f_ann_date TEXT,
                end_date TEXT,
                eps REAL,
                dt_eps REAL,
                total_revenue_ps REAL,
                revenue_ps REAL,
                capital_rese_ps REAL,
                surplus_rese_ps REAL,
                undist_profit_ps REAL,
                extra_item REAL,
                profit_dedt REAL,
                gross_margin REAL,
                current_ratio REAL,
                quick_ratio REAL,
                cash_ratio REAL,
                invturn_days REAL,
                ar_turn REAL,
                ar_turn_days REAL,
                ol_turn REAL,
                fa_turn REAL,
                assets_turn REAL,
                roe REAL,
                roe_waa REAL,
                roa REAL,
                npta REAL,
                roe_dt REAL,
                op_income REAL,
                ebit REAL,
                ebitda REAL,
                interest_coverage REAL,
                debt_to_assets REAL,
                assets_non_current REAL,
                profit_to_gr REAL,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ts_code, end_date)
            )
            ''')
            
            # 创建指数数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                volume INTEGER,
                amount REAL,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ts_code, trade_date)
            )
            ''')
            
            # 创建处理后的数据表（用于特征工程）
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                features TEXT,
                label REAL,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ts_code, trade_date)
            )
            ''')
            
            self.conn.commit()
            self.logger.info("数据库表结构初始化完成")
        except Exception as e:
            self.logger.error(f"初始化数据库表结构失败: {str(e)}")
            self.conn.rollback()
    
    def save_stock_basic(self, df: pd.DataFrame):
        """保存股票基础信息到数据库
        
        Args:
            df: 包含股票基础信息的DataFrame
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法保存股票基础信息")
            return
        
        try:
            # 确保列存在
            required_columns = ['ts_code']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据中缺少必要列: {col}")
                    return
            
            # 将DataFrame写入数据库，使用replace策略
            df.to_sql('stock_basic', self.conn, if_exists='replace', index=False)
            self.conn.commit()
            self.logger.info(f"成功保存{len(df)}条股票基础信息到数据库")
        except Exception as e:
            self.logger.error(f"保存股票基础信息失败: {str(e)}")
            self.conn.rollback()
    
    def save_daily_data(self, df: pd.DataFrame, if_exists: str = 'append'):
        """保存日线数据到数据库
        
        Args:
            df: 包含日线数据的DataFrame
            if_exists: 数据存在时的处理方式，可选 'replace', 'append', 'fail'
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法保存日线数据")
            return
        
        try:
            # 确保必要的列存在
            required_columns = ['ts_code', 'trade_date']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据中缺少必要列: {col}")
                    return
            
            # 将DataFrame写入数据库
            df.to_sql('daily_data', self.conn, if_exists=if_exists, index=False)
            self.conn.commit()
            self.logger.info(f"成功保存{len(df)}条日线数据到数据库")
        except Exception as e:
            self.logger.error(f"保存日线数据失败: {str(e)}")
            self.conn.rollback()
    
    def save_fundamental_data(self, df: pd.DataFrame, if_exists: str = 'append'):
        """保存基本面数据到数据库
        
        Args:
            df: 包含基本面数据的DataFrame
            if_exists: 数据存在时的处理方式
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法保存基本面数据")
            return
        
        try:
            # 确保必要的列存在
            required_columns = ['ts_code']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据中缺少必要列: {col}")
                    return
            
            # 将DataFrame写入数据库
            df.to_sql('fundamental_data', self.conn, if_exists=if_exists, index=False)
            self.conn.commit()
            self.logger.info(f"成功保存{len(df)}条基本面数据到数据库")
        except Exception as e:
            self.logger.error(f"保存基本面数据失败: {str(e)}")
            self.conn.rollback()
    
    def save_index_data(self, df: pd.DataFrame, if_exists: str = 'append'):
        """保存指数数据到数据库
        
        Args:
            df: 包含指数数据的DataFrame
            if_exists: 数据存在时的处理方式
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法保存指数数据")
            return
        
        try:
            # 确保必要的列存在
            required_columns = ['ts_code', 'trade_date']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据中缺少必要列: {col}")
                    return
            
            # 将DataFrame写入数据库
            df.to_sql('index_data', self.conn, if_exists=if_exists, index=False)
            self.conn.commit()
            self.logger.info(f"成功保存{len(df)}条指数数据到数据库")
        except Exception as e:
            self.logger.error(f"保存指数数据失败: {str(e)}")
            self.conn.rollback()
    
    def save_processed_data(self, df: pd.DataFrame, if_exists: str = 'append'):
        """保存处理后的数据（特征工程结果）到数据库
        
        Args:
            df: 包含处理后数据的DataFrame
            if_exists: 数据存在时的处理方式
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法保存处理后的数据")
            return
        
        try:
            # 确保必要的列存在
            required_columns = ['ts_code', 'trade_date']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据中缺少必要列: {col}")
                    return
            
            # 将DataFrame写入数据库
            df.to_sql('processed_data', self.conn, if_exists=if_exists, index=False)
            self.conn.commit()
            self.logger.info(f"成功保存{len(df)}条处理后的数据到数据库")
        except Exception as e:
            self.logger.error(f"保存处理后的数据失败: {str(e)}")
            self.conn.rollback()
    
    def load_stock_basic(self, ts_codes: List[str] = None) -> Optional[pd.DataFrame]:
        """从数据库加载股票基础信息
        
        Args:
            ts_codes: 股票代码列表，如果为None则加载所有
        
        Returns:
            股票基础信息的DataFrame，如果加载失败则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法加载股票基础信息")
            return None
        
        try:
            if ts_codes is None:
                query = "SELECT * FROM stock_basic"
                df = pd.read_sql_query(query, self.conn)
            else:
                # 使用参数化查询
                placeholders = ','.join(['?' for _ in ts_codes])
                query = f"SELECT * FROM stock_basic WHERE ts_code IN ({placeholders})"
                df = pd.read_sql_query(query, self.conn, params=ts_codes)
            
            self.logger.info(f"成功从数据库加载{len(df)}条股票基础信息")
            return df
        except Exception as e:
            self.logger.error(f"加载股票基础信息失败: {str(e)}")
            return None
    
    def load_daily_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """从数据库加载日线数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            日线数据的DataFrame，如果加载失败则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法加载日线数据")
            return None
        
        try:
            # 构建查询语句
            query_parts = ["SELECT * FROM daily_data WHERE ts_code = ?"]
            params = [ts_code]
            
            if start_date is not None:
                query_parts.append("AND trade_date >= ?")
                params.append(start_date)
            
            if end_date is not None:
                query_parts.append("AND trade_date <= ?")
                params.append(end_date)
            
            query_parts.append("ORDER BY trade_date")
            query = " ".join(query_parts)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            self.logger.info(f"成功从数据库加载{ts_code}的{len(df)}条日线数据")
            return df
        except Exception as e:
            self.logger.error(f"加载{ts_code}的日线数据失败: {str(e)}")
            return None
    
    def load_fundamental_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """从数据库加载基本面数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            基本面数据的DataFrame，如果加载失败则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法加载基本面数据")
            return None
        
        try:
            # 构建查询语句
            query_parts = ["SELECT * FROM fundamental_data WHERE ts_code = ?"]
            params = [ts_code]
            
            if start_date is not None:
                query_parts.append("AND end_date >= ?")
                params.append(start_date)
            
            if end_date is not None:
                query_parts.append("AND end_date <= ?")
                params.append(end_date)
            
            query = " ".join(query_parts)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            self.logger.info(f"成功从数据库加载{ts_code}的{len(df)}条基本面数据")
            return df
        except Exception as e:
            self.logger.error(f"加载{ts_code}的基本面数据失败: {str(e)}")
            return None
    
    def load_index_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """从数据库加载指数数据
        
        Args:
            ts_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            指数数据的DataFrame，如果加载失败则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法加载指数数据")
            return None
        
        try:
            # 构建查询语句
            query_parts = ["SELECT * FROM index_data WHERE ts_code = ?"]
            params = [ts_code]
            
            if start_date is not None:
                query_parts.append("AND trade_date >= ?")
                params.append(start_date)
            
            if end_date is not None:
                query_parts.append("AND trade_date <= ?")
                params.append(end_date)
            
            query_parts.append("ORDER BY trade_date")
            query = " ".join(query_parts)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            self.logger.info(f"成功从数据库加载{ts_code}的{len(df)}条指数数据")
            return df
        except Exception as e:
            self.logger.error(f"加载{ts_code}的指数数据失败: {str(e)}")
            return None
    
    def load_processed_data(self, ts_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """从数据库加载处理后的数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            处理后数据的DataFrame，如果加载失败则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法加载处理后的数据")
            return None
        
        try:
            # 构建查询语句
            query_parts = ["SELECT * FROM processed_data WHERE ts_code = ?"]
            params = [ts_code]
            
            if start_date is not None:
                query_parts.append("AND trade_date >= ?")
                params.append(start_date)
            
            if end_date is not None:
                query_parts.append("AND trade_date <= ?")
                params.append(end_date)
            
            query_parts.append("ORDER BY trade_date")
            query = " ".join(query_parts)
            
            df = pd.read_sql_query(query, self.conn, params=params)
            self.logger.info(f"成功从数据库加载{ts_code}的{len(df)}条处理后的数据")
            return df
        except Exception as e:
            self.logger.error(f"加载{ts_code}的处理后数据失败: {str(e)}")
            return None
    
    def update_data(self, table_name: str, data: Dict, condition: Dict):
        """更新数据库中的数据
        
        Args:
            table_name: 表名
            data: 要更新的数据字典，键为列名，值为新值
            condition: 更新条件字典，键为列名，值为条件值
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法更新数据")
            return
        
        try:
            cursor = self.conn.cursor()
            
            # 构建SET部分
            set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
            # 构建WHERE部分
            where_clause = " AND ".join([f"{key} = ?" for key in condition.keys()])
            
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            
            # 参数列表
            params = list(data.values()) + list(condition.values())
            
            cursor.execute(query, params)
            self.conn.commit()
            self.logger.info(f"成功更新{cursor.rowcount}条数据")
        except Exception as e:
            self.logger.error(f"更新数据失败: {str(e)}")
            self.conn.rollback()
    
    def delete_data(self, table_name: str, condition: Dict = None):
        """删除数据库中的数据
        
        Args:
            table_name: 表名
            condition: 删除条件字典，如果为None则删除表中所有数据
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法删除数据")
            return
        
        try:
            cursor = self.conn.cursor()
            
            if condition is None:
                query = f"DELETE FROM {table_name}"
                params = []
            else:
                where_clause = " AND ".join([f"{key} = ?" for key in condition.keys()])
                query = f"DELETE FROM {table_name} WHERE {where_clause}"
                params = list(condition.values())
            
            cursor.execute(query, params)
            self.conn.commit()
            self.logger.info(f"成功删除{cursor.rowcount}条数据")
        except Exception as e:
            self.logger.error(f"删除数据失败: {str(e)}")
            self.conn.rollback()
    
    def get_last_update_date(self, table_name: str, ts_code: str) -> Optional[str]:
        """获取指定表和股票的最后更新日期
        
        Args:
            table_name: 表名
            ts_code: 股票代码
        
        Returns:
            最后更新日期字符串，如果不存在则返回None
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法获取最后更新日期")
            return None
        
        try:
            query = f"SELECT MAX(trade_date) as last_date FROM {table_name} WHERE ts_code = ?"
            df = pd.read_sql_query(query, self.conn, params=[ts_code])
            
            if not df.empty and pd.notna(df['last_date'].iloc[0]):
                return str(df['last_date'].iloc[0])
            else:
                self.logger.warning(f"表{table_name}中不存在{ts_code}的数据")
                return None
        except Exception as e:
            self.logger.error(f"获取最后更新日期失败: {str(e)}")
            return None
    
    def check_table_exists(self, table_name: str) -> bool:
        """检查指定的表是否存在
        
        Args:
            table_name: 表名
        
        Returns:
            如果表存在则返回True，否则返回False
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法检查表是否存在")
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table_name])
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            self.logger.error(f"检查表{table_name}是否存在失败: {str(e)}")
            return False
    
    def get_table_size(self, table_name: str) -> int:
        """获取表的记录数量
        
        Args:
            table_name: 表名
        
        Returns:
            表的记录数量
        """
        if self.conn is None:
            self.logger.error("数据库未连接，无法获取表大小")
            return 0
        
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            df = pd.read_sql_query(query, self.conn)
            return df['count'].iloc[0]
        except Exception as e:
            self.logger.error(f"获取表{table_name}大小失败: {str(e)}")
            return 0