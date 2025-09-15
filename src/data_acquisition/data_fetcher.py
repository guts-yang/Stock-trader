"""数据获取器模块，负责从Tushare和AkShare获取股票数据"""

import pandas as pd
import tushare as ts
import akshare as ak
import logging
from typing import Dict, List, Optional, Union
import time

class DataFetcher:
    """数据获取器类，负责从各种数据源获取股票相关数据"""
    
    def __init__(self, config: Dict):
        """初始化数据获取器
        
        Args:
            config: 配置字典，包含数据源配置信息
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 初始化Tushare
        # 先检查配置中是否有data_sources.tushare
        tushare_config = None
        if 'data_sources' in config and 'tushare' in config['data_sources']:
            tushare_config = config['data_sources']['tushare']
        elif 'tushare' in config:
            # 向后兼容：直接在顶层查找tushare配置
            tushare_config = config['tushare']
        
        if tushare_config and 'api_key' in tushare_config and tushare_config['api_key']:
            try:
                ts.set_token(tushare_config['api_key'])
                self.ts_pro = ts.pro_api()
                self.logger.info("Tushare API 初始化成功")
            except Exception as e:
                self.logger.error(f"Tushare API 初始化失败: {str(e)}")
                self.ts_pro = None
        else:
            self.logger.warning("Tushare API 配置缺失或API Key为空")
            self.ts_pro = None
        
        # AkShare不需要API Key，直接可用
        self.logger.info("AkShare 模块加载成功")
        
        # 重试设置
        self.retry_count = tushare_config.get('retry_count', 3) if tushare_config else 3
        self.timeout = tushare_config.get('timeout', 30) if tushare_config else 30
    
    def get_stock_basic(self, market: str = 'all') -> Optional[pd.DataFrame]:
        """获取股票基础信息
        
        Args:
            market: 市场类型，可选 'all', 'sh', 'sz', 'cyb', 'zxb'
        
        Returns:
            股票基础信息的DataFrame，如果获取失败则返回None
        """
        try:
            # 优先使用Tushare
            if self.ts_pro:
                for i in range(self.retry_count):
                    try:
                        if market == 'all':
                            df = self.ts_pro.stock_basic(exchange='', list_status='L', 
                                                        fields='ts_code,symbol,name,area,industry,list_date')
                        else:
                            df = self.ts_pro.stock_basic(exchange=market, list_status='L', 
                                                        fields='ts_code,symbol,name,area,industry,list_date')
                        self.logger.info(f"通过Tushare获取股票基础信息成功，共{len(df)}只股票")
                        return df
                    except Exception as e:
                        self.logger.warning(f"Tushare获取股票基础信息失败(尝试{i+1}/{self.retry_count}): {str(e)}")
                        time.sleep(2)
            
            # Tushare失败时使用AkShare
            self.logger.info("尝试使用AkShare获取股票基础信息")
            df = ak.stock_board_industry_name_ths()
            if not df.empty:
                self.logger.info(f"通过AkShare获取股票基础信息成功，共{len(df)}条记录")
                return df
            else:
                self.logger.error("所有数据源获取股票基础信息均失败")
                return None
        except Exception as e:
            self.logger.error(f"获取股票基础信息异常: {str(e)}")
            return None
    
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票日线行情数据
        
        Args:
            ts_code: 股票代码，如 '000001.SZ'
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
        
        Returns:
            日线行情数据的DataFrame，如果获取失败则返回None
        """
        try:
            # 优先使用Tushare
            if self.ts_pro:
                for i in range(self.retry_count):
                    try:
                        df = self.ts_pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                        if not df.empty:
                            # 按日期排序
                            df = df.sort_values('trade_date')
                            # 转换日期格式
                            df['trade_date'] = pd.to_datetime(df['trade_date'])
                            self.logger.info(f"通过Tushare获取{ts_code}日线数据成功，共{len(df)}条记录")
                            return df
                    except Exception as e:
                        self.logger.warning(f"Tushare获取{ts_code}日线数据失败(尝试{i+1}/{self.retry_count}): {str(e)}")
                        time.sleep(2)
            
            # Tushare失败时使用AkShare
            self.logger.info(f"尝试使用AkShare获取{ts_code}日线数据")
            # AkShare使用的是'YYYY-MM-DD'格式
            start_date_ak = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end_date_ak = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 提取股票代码和市场
            symbol = ts_code.split('.')[0]
            market = ts_code.split('.')[1].lower()
            
            try:
                if market == 'sz':
                    df = ak.stock_zh_a_daily(symbol=symbol, period="daily", start_date=start_date_ak, end_date=end_date_ak)
                elif market == 'sh':
                    df = ak.stock_zh_a_daily(symbol=symbol, period="daily", start_date=start_date_ak, end_date=end_date_ak)
                else:
                    self.logger.error(f"不支持的市场类型: {market}")
                    return None
                
                if not df.empty:
                    # 添加股票代码列
                    df['ts_code'] = ts_code
                    self.logger.info(f"通过AkShare获取{ts_code}日线数据成功，共{len(df)}条记录")
                    return df
            except Exception as e:
                self.logger.error(f"AkShare获取{ts_code}日线数据失败: {str(e)}")
            
            self.logger.error(f"所有数据源获取{ts_code}日线数据均失败")
            return None
        except Exception as e:
            self.logger.error(f"获取{ts_code}日线数据异常: {str(e)}")
            return None
    
    def get_fundamental_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票基本面数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
        
        Returns:
            基本面数据的DataFrame，如果获取失败则返回None
        """
        try:
            # 优先使用Tushare
            if self.ts_pro:
                for i in range(self.retry_count):
                    try:
                        # 获取财务指标数据
                        df_fina_indicator = self.ts_pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
                        
                        if not df_fina_indicator.empty:
                            self.logger.info(f"通过Tushare获取{ts_code}基本面数据成功")
                            return df_fina_indicator
                    except Exception as e:
                        self.logger.warning(f"Tushare获取{ts_code}基本面数据失败(尝试{i+1}/{self.retry_count}): {str(e)}")
                        time.sleep(2)
            
            # Tushare失败时使用AkShare
            self.logger.info(f"尝试使用AkShare获取{ts_code}基本面数据")
            try:
                # 提取股票代码
                symbol = ts_code.split('.')[0]
                # 获取市盈率数据
                df_pe = ak.stock_a_pe(symbol=symbol)
                if not df_pe.empty:
                    self.logger.info(f"通过AkShare获取{ts_code}市盈率数据成功")
                    return df_pe
            except Exception as e:
                self.logger.error(f"AkShare获取{ts_code}基本面数据失败: {str(e)}")
            
            self.logger.error(f"所有数据源获取{ts_code}基本面数据均失败")
            return None
        except Exception as e:
            self.logger.error(f"获取{ts_code}基本面数据异常: {str(e)}")
            return None
    
    def get_index_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取指数数据
        
        Args:
            ts_code: 指数代码，如 '000300.SH'
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
        
        Returns:
            指数数据的DataFrame，如果获取失败则返回None
        """
        try:
            # 优先使用Tushare
            if self.ts_pro:
                for i in range(self.retry_count):
                    try:
                        df = self.ts_pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                        if not df.empty:
                            df = df.sort_values('trade_date')
                            df['trade_date'] = pd.to_datetime(df['trade_date'])
                            self.logger.info(f"通过Tushare获取{ts_code}指数数据成功")
                            return df
                    except Exception as e:
                        self.logger.warning(f"Tushare获取{ts_code}指数数据失败(尝试{i+1}/{self.retry_count}): {str(e)}")
                        time.sleep(2)
            
            # Tushare失败时使用AkShare
            self.logger.info(f"尝试使用AkShare获取{ts_code}指数数据")
            start_date_ak = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end_date_ak = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            try:
                if ts_code == '000300.SH':
                    df = ak.index_zh_a_daily(symbol="000300", period="daily", start_date=start_date_ak, end_date=end_date_ak)
                else:
                    # 尝试通用方法
                    symbol = ts_code.split('.')[0]
                    df = ak.index_zh_a_daily(symbol=symbol, period="daily", start_date=start_date_ak, end_date=end_date_ak)
                
                if not df.empty:
                    df['ts_code'] = ts_code
                    self.logger.info(f"通过AkShare获取{ts_code}指数数据成功")
                    return df
            except Exception as e:
                self.logger.error(f"AkShare获取{ts_code}指数数据失败: {str(e)}")
            
            self.logger.error(f"所有数据源获取{ts_code}指数数据均失败")
            return None
        except Exception as e:
            self.logger.error(f"获取{ts_code}指数数据异常: {str(e)}")
            return None
    
    def get_batch_daily_data(self, ts_codes: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        """批量获取多只股票的日线数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            以股票代码为键，日线数据DataFrame为值的字典
        """
        result = {}
        success_count = 0
        
        for ts_code in ts_codes:
            self.logger.info(f"开始获取{ts_code}的日线数据")
            df = self.get_daily_data(ts_code, start_date, end_date)
            result[ts_code] = df
            if df is not None:
                success_count += 1
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        self.logger.info(f"批量获取日线数据完成，成功{success_count}/{len(ts_codes)}只股票")
        return result