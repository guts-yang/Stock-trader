"""数据处理管道示例脚本，展示如何使用数据获取与预处理模块"""

import yaml
import logging
import logging.config
import os
from datetime import datetime, timedelta
import pandas as pd

# 导入数据获取与预处理模块
from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .database_manager import DatabaseManager

class DataPipeline:
    """数据处理管道类，整合数据获取、处理和存储功能"""
    
    def __init__(self, config_path: str = '../../config/config.yaml', 
                 logging_config_path: str = '../../config/logging.yaml'):
        """初始化数据处理管道
        
        Args:
            config_path: 配置文件路径
            logging_config_path: 日志配置文件路径
        """
        # 加载日志配置
        with open(logging_config_path, 'r', encoding='utf-8') as f:
            logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)
        self.logger = logging.getLogger(__name__)
        
        # 加载系统配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化各个模块
        self.fetcher = DataFetcher(self.config['data_sources'])
        self.processor = DataProcessor(self.config)
        self.db_manager = DatabaseManager(self.config)
        
        self.logger.info("数据处理管道初始化完成")
    
    def fetch_and_store_stock_basic(self):
        """获取并存储股票基础信息"""
        try:
            self.logger.info("开始获取股票基础信息")
            # 获取股票基础信息
            df_stock_basic = self.fetcher.get_stock_basic()
            
            if df_stock_basic is not None and not df_stock_basic.empty:
                # 保存到数据库
                self.db_manager.save_stock_basic(df_stock_basic)
                self.logger.info("股票基础信息获取和存储完成")
            else:
                self.logger.error("无法获取有效的股票基础信息")
        except Exception as e:
            self.logger.error(f"获取和存储股票基础信息时发生异常: {str(e)}")
    
    def fetch_and_store_daily_data(self, ts_codes: list = None, date_range: dict = None):
        """获取并存储股票日线数据
        
        Args:
            ts_codes: 股票代码列表，如果为None则从配置中获取
            date_range: 日期范围字典，包含start和end字段
        """
        try:
            # 如果未提供股票代码列表，则从配置中获取
            if ts_codes is None:
                ts_codes = self.config['data_processing'].get('stocks', [])
                
                # 如果配置中也没有，尝试从数据库加载
                if not ts_codes:
                    df_stock_basic = self.db_manager.load_stock_basic()
                    if df_stock_basic is not None and not df_stock_basic.empty:
                        # 只取前20只股票作为示例
                        ts_codes = df_stock_basic['ts_code'].head(20).tolist()
                        self.logger.info(f"从数据库加载了{len(ts_codes)}只股票代码")
                    else:
                        self.logger.error("无法获取股票代码列表，使用默认示例股票")
                        ts_codes = ['000001.SZ', '600000.SH']  # 默认示例股票
            
            # 如果未提供日期范围，则从配置中获取
            if date_range is None:
                date_range = self.config['data_processing'].get('date_range', {})
                
                # 如果配置中也没有，使用默认日期范围（最近一年）
                if not date_range or 'start' not in date_range or 'end' not in date_range:
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    date_range = {'start': start_date, 'end': end_date}
                    self.logger.info(f"使用默认日期范围: {start_date} 到 {end_date}")
            
            start_date = date_range['start']
            end_date = date_range['end']
            
            self.logger.info(f"开始获取{len(ts_codes)}只股票的日线数据，日期范围: {start_date} 到 {end_date}")
            
            # 批量获取日线数据
            daily_data_dict = self.fetcher.get_batch_daily_data(ts_codes, start_date, end_date)
            
            # 处理并存储每只股票的数据
            for ts_code, df_daily in daily_data_dict.items():
                if df_daily is not None and not df_daily.empty:
                    try:
                        # 清洗数据
                        cleaned_df = self.processor.clean_data(df_daily)
                        
                        # 如果启用了原始数据保存，则保存清洗后的数据
                        if self.config['data_processing'].get('save_raw_data', True):
                            self.db_manager.save_daily_data(cleaned_df)
                            self.logger.info(f"已保存{ts_code}的日线数据")
                    except Exception as e:
                        self.logger.error(f"处理和存储{ts_code}的日线数据时发生异常: {str(e)}")
            
            self.logger.info("日线数据获取和存储流程完成")
        except Exception as e:
            self.logger.error(f"获取和存储日线数据时发生异常: {str(e)}")
    
    def fetch_and_store_fundamental_data(self, ts_codes: list = None, date_range: dict = None):
        """获取并存储股票基本面数据
        
        Args:
            ts_codes: 股票代码列表
            date_range: 日期范围字典
        """
        try:
            # 如果未提供股票代码列表，则从配置中获取
            if ts_codes is None:
                ts_codes = self.config['data_processing'].get('stocks', [])
                if not ts_codes:
                    self.logger.error("无法获取股票代码列表")
                    return
            
            # 如果未提供日期范围，则从配置中获取
            if date_range is None:
                date_range = self.config['data_processing'].get('date_range', {})
                if not date_range or 'start' not in date_range or 'end' not in date_range:
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')  # 最近3年
                    date_range = {'start': start_date, 'end': end_date}
            
            start_date = date_range['start']
            end_date = date_range['end']
            
            self.logger.info(f"开始获取{len(ts_codes)}只股票的基本面数据，日期范围: {start_date} 到 {end_date}")
            
            # 获取并存储每只股票的基本面数据
            for ts_code in ts_codes:
                try:
                    df_fundamental = self.fetcher.get_fundamental_data(ts_code, start_date, end_date)
                    
                    if df_fundamental is not None and not df_fundamental.empty:
                        # 清洗数据
                        cleaned_df = self.processor.clean_data(df_fundamental)
                        
                        # 保存到数据库
                        self.db_manager.save_fundamental_data(cleaned_df)
                        self.logger.info(f"已保存{ts_code}的基本面数据")
                except Exception as e:
                    self.logger.error(f"处理和存储{ts_code}的基本面数据时发生异常: {str(e)}")
            
            self.logger.info("基本面数据获取和存储流程完成")
        except Exception as e:
            self.logger.error(f"获取和存储基本面数据时发生异常: {str(e)}")
    
    def fetch_and_store_index_data(self, index_codes: list = None, date_range: dict = None):
        """获取并存储指数数据
        
        Args:
            index_codes: 指数代码列表
            date_range: 日期范围字典
        """
        try:
            # 如果未提供指数代码列表，则使用默认值
            if index_codes is None:
                index_codes = [self.config['backtesting'].get('benchmark', '000300.SH')]
            
            # 如果未提供日期范围，则从配置中获取
            if date_range is None:
                date_range = self.config['data_processing'].get('date_range', {})
                if not date_range or 'start' not in date_range or 'end' not in date_range:
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    date_range = {'start': start_date, 'end': end_date}
            
            start_date = date_range['start']
            end_date = date_range['end']
            
            self.logger.info(f"开始获取{len(index_codes)}个指数的数据，日期范围: {start_date} 到 {end_date}")
            
            # 获取并存储每个指数的数据
            for index_code in index_codes:
                try:
                    df_index = self.fetcher.get_index_data(index_code, start_date, end_date)
                    
                    if df_index is not None and not df_index.empty:
                        # 清洗数据
                        cleaned_df = self.processor.clean_data(df_index)
                        
                        # 保存到数据库
                        self.db_manager.save_index_data(cleaned_df)
                        self.logger.info(f"已保存{index_code}的指数数据")
                except Exception as e:
                    self.logger.error(f"处理和存储{index_code}的指数数据时发生异常: {str(e)}")
            
            self.logger.info("指数数据获取和存储流程完成")
        except Exception as e:
            self.logger.error(f"获取和存储指数数据时发生异常: {str(e)}")
    
    def run_complete_pipeline(self):
        """运行完整的数据处理管道"""
        try:
            self.logger.info("开始运行完整的数据处理管道")
            
            # 1. 获取并存储股票基础信息
            self.fetch_and_store_stock_basic()
            
            # 2. 获取并存储指数数据（用于回测基准）
            self.fetch_and_store_index_data()
            
            # 3. 获取并存储日线数据
            self.fetch_and_store_daily_data()
            
            # 4. 获取并存储基本面数据
            self.fetch_and_store_fundamental_data()
            
            self.logger.info("完整的数据处理管道运行完成")
        except Exception as e:
            self.logger.error(f"运行完整数据处理管道时发生异常: {str(e)}")
    
    def update_existing_data(self):
        """更新数据库中已有的数据（增量更新）"""
        try:
            self.logger.info("开始更新数据库中的现有数据")
            
            # 1. 获取需要更新的股票列表
            df_stock_basic = self.db_manager.load_stock_basic()
            if df_stock_basic is None or df_stock_basic.empty:
                self.logger.warning("数据库中没有股票基础信息，将执行全量获取")
                self.run_complete_pipeline()
                return
            
            ts_codes = df_stock_basic['ts_code'].tolist()
            
            # 2. 计算更新的日期范围（从数据库中最后一条记录的日期到今天）
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 3. 对每只股票进行增量更新
            for ts_code in ts_codes:
                # 获取最后更新日期
                last_date = self.db_manager.get_last_update_date('daily_data', ts_code)
                
                if last_date is None:
                    # 如果没有数据，进行全量获取（使用过去一年的数据）
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                    self.logger.info(f"{ts_code}没有历史数据，将获取从{start_date}到{end_date}的全量数据")
                else:
                    # 如果有数据，进行增量更新
                    # 格式化last_date为标准格式
                    if len(last_date) == 8 and last_date.isdigit():
                        last_datetime = datetime.strptime(last_date, '%Y%m%d')
                    else:
                        # 尝试其他日期格式
                        try:
                            last_datetime = datetime.strptime(last_date, '%Y-%m-%d')
                        except:
                            self.logger.error(f"无法解析{ts_code}的最后更新日期: {last_date}")
                            continue
                    
                    # 设置开始日期为最后更新日期的下一天
                    start_date = (last_datetime + timedelta(days=1)).strftime('%Y%m%d')
                    
                    # 如果开始日期大于结束日期，说明数据已经是最新的
                    if start_date > end_date:
                        self.logger.info(f"{ts_code}的数据已经是最新的，不需要更新")
                        continue
                    
                    self.logger.info(f"开始更新{ts_code}的数据，日期范围: {start_date} 到 {end_date}")
                
                try:
                    # 获取增量数据
                    df_daily = self.fetcher.get_daily_data(ts_code, start_date, end_date)
                    
                    if df_daily is not None and not df_daily.empty:
                        # 清洗数据
                        cleaned_df = self.processor.clean_data(df_daily)
                        
                        # 保存到数据库
                        self.db_manager.save_daily_data(cleaned_df)
                        self.logger.info(f"已更新{ts_code}的日线数据")
                except Exception as e:
                    self.logger.error(f"更新{ts_code}的数据时发生异常: {str(e)}")
            
            self.logger.info("数据更新流程完成")
        except Exception as e:
            self.logger.error(f"更新数据时发生异常: {str(e)}")
    
    def close(self):
        """关闭数据处理管道，释放资源"""
        # 断开数据库连接
        self.db_manager.disconnect()
        self.logger.info("数据处理管道已关闭")

if __name__ == '__main__':
    """主函数，用于测试数据处理管道"""
    try:
        # 创建数据处理管道实例
        pipeline = DataPipeline()
        
        # 运行完整的数据处理管道
        pipeline.run_complete_pipeline()
        
        # 关闭管道
        pipeline.close()
    except Exception as e:
        print(f"运行数据处理管道时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()