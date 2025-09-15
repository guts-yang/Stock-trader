"""数据处理器模块，负责数据清洗、标准化等预处理工作"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

class DataProcessor:
    """数据处理器类，负责对获取的原始数据进行清洗、转换和标准化等预处理"""
    
    def __init__(self, config: Dict):
        """初始化数据处理器
        
        Args:
            config: 配置字典，包含数据处理相关配置
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.fill_method = config['data_processing'].get('fill_method', 'ffill')
        self.normalize_method = config['data_processing'].get('normalize_method', 'zscore')
        self.scalers = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据，处理缺失值、异常值等
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            清洗后的数据DataFrame
        """
        # 深拷贝避免修改原始数据
        cleaned_df = df.copy()
        
        # 检查并处理缺失值
        if cleaned_df.isnull().sum().sum() > 0:
            self.logger.info(f"发现{cleaned_df.isnull().sum().sum()}个缺失值，开始处理")
            
            # 对不同列采用不同的填充策略
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() > 0:
                    if self.fill_method == 'ffill':
                        cleaned_df[col] = cleaned_df[col].ffill()
                    elif self.fill_method == 'bfill':
                        cleaned_df[col] = cleaned_df[col].bfill()
                    elif self.fill_method == 'mean':
                        # 仅对数值型列使用均值填充
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    elif self.fill_method == 'median':
                        # 仅对数值型列使用中位数填充
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        # 默认使用向前填充
                        cleaned_df[col] = cleaned_df[col].ffill()
            
            # 对于仍然存在的缺失值（如序列开头的缺失值）
            cleaned_df = cleaned_df.dropna()
            self.logger.info(f"缺失值处理完成，剩余数据量: {len(cleaned_df)}")
        
        # 处理异常值（使用3σ法则）
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['ts_code', 'trade_date', 'symbol']:
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                # 找出异常值的掩码
                outliers_mask = (cleaned_df[col] < mean - 3 * std) | (cleaned_df[col] > mean + 3 * std)
                if outliers_mask.sum() > 0:
                    self.logger.info(f"列{col}中发现{outliers_mask.sum()}个异常值，使用上下限替换")
                    # 替换异常值为上下限
                    cleaned_df.loc[outliers_mask, col] = np.clip(cleaned_df[col], mean - 3 * std, mean + 3 * std)
        
        # 确保日期列格式正确
        if 'trade_date' in cleaned_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['trade_date']):
                try:
                    # 尝试多种日期格式
                    formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
                    for fmt in formats:
                        try:
                            cleaned_df['trade_date'] = pd.to_datetime(cleaned_df['trade_date'], format=fmt)
                            break
                        except:
                            continue
                except Exception as e:
                    self.logger.error(f"转换日期格式失败: {str(e)}")
            
            # 按日期排序
            cleaned_df = cleaned_df.sort_values('trade_date')
        
        return cleaned_df
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None, 
                       return_scaler: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """对数据进行标准化处理
        
        Args:
            df: 待标准化的数据DataFrame
            columns: 需要标准化的列，如果为None则对所有数值列进行标准化
            return_scaler: 是否返回用于标准化的scaler对象
        
        Returns:
            标准化后的数据DataFrame，如果return_scaler为True，则返回(DataFrame, scaler_dict)
        """
        # 深拷贝避免修改原始数据
        normalized_df = df.copy()
        
        # 如果未指定列，默认对所有数值列进行标准化
        if columns is None:
            columns = normalized_df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除非特征列
            exclude_cols = ['ts_code', 'symbol', 'trade_date', 'date']
            columns = [col for col in columns if col not in exclude_cols]
        
        # 创建scaler字典
        scaler_dict = {}
        
        for col in columns:
            if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
                try:
                    # 根据配置选择标准化方法
                    if self.normalize_method == 'minmax':
                        scaler = MinMaxScaler()
                    else:  # 默认zscore
                        scaler = StandardScaler()
                    
                    # 保存scaler
                    scaler_dict[col] = scaler
                    
                    # 进行标准化
                    normalized_df[col] = scaler.fit_transform(normalized_df[[col]])
                    
                    self.logger.info(f"列{col}标准化完成，方法: {self.normalize_method}")
                except Exception as e:
                    self.logger.error(f"列{col}标准化失败: {str(e)}")
        
        if return_scaler:
            return normalized_df, scaler_dict
        else:
            return normalized_df
    
    def apply_scaler(self, df: pd.DataFrame, scaler_dict: Dict) -> pd.DataFrame:
        """使用已有的scaler对新数据进行标准化
        
        Args:
            df: 待标准化的新数据DataFrame
            scaler_dict: 已训练好的scaler字典
        
        Returns:
            标准化后的数据DataFrame
        """
        scaled_df = df.copy()
        
        for col, scaler in scaler_dict.items():
            if col in scaled_df.columns and pd.api.types.is_numeric_dtype(scaled_df[col]):
                try:
                    scaled_df[col] = scaler.transform(scaled_df[[col]])
                    self.logger.info(f"使用已有scaler对列{col}进行标准化")
                except Exception as e:
                    self.logger.error(f"使用已有scaler对列{col}标准化失败: {str(e)}")
        
        return scaled_df
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """对时间序列数据进行重采样
        
        Args:
            df: 时间序列数据DataFrame
            freq: 重采样频率，如 'D'(日), 'W'(周), 'M'(月)
        
        Returns:
            重采样后的数据DataFrame
        """
        if 'trade_date' not in df.columns:
            self.logger.error("数据中不包含trade_date列，无法进行重采样")
            return df
        
        # 设置日期列为索引
        resampled_df = df.set_index('trade_date')
        
        # 对数值列进行重采样，根据频率选择合适的聚合方法
        numeric_cols = resampled_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = resampled_df.select_dtypes(exclude=[np.number]).columns
        
        # 对于不同的频率采用不同的聚合方法
        if freq == 'D':
            # 日线数据不需要重采样
            resampled_df = resampled_df.reset_index()
        elif freq == 'W':
            # 周线数据：开盘价用周一，收盘价用周五，最高价用一周最高，最低价用一周最低，成交量和成交额求和
            resampled_df = resampled_df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
            resampled_df = resampled_df.reset_index()
        elif freq == 'M':
            # 月线数据：类似周线
            resampled_df = resampled_df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
            resampled_df = resampled_df.reset_index()
        else:
            # 其他频率默认使用最后一个值
            resampled_df = resampled_df.resample(freq).last()
            resampled_df = resampled_df.reset_index()
        
        self.logger.info(f"数据重采样完成，频率: {freq}，原始数据量: {len(df)}，重采样后数据量: {len(resampled_df)}")
        return resampled_df
    
    def generate_technical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成基本的技术指标列
        
        Args:
            df: 包含开盘价、收盘价、最高价、最低价、成交量的DataFrame
        
        Returns:
            添加了技术指标列的数据DataFrame
        """
        tech_df = df.copy()
        
        # 计算收益率
        if 'close' in tech_df.columns:
            tech_df['return'] = tech_df['close'].pct_change()
            tech_df['log_return'] = np.log(tech_df['close'] / tech_df['close'].shift(1))
        
        # 计算涨跌停标记
        if 'open' in tech_df.columns and 'close' in tech_df.columns:
            # 简单的涨跌停判断（实际情况可能更复杂）
            tech_df['price_change'] = tech_df['close'] - tech_df['open']
            tech_df['pct_change'] = tech_df['price_change'] / tech_df['open'] * 100
            
            # 添加涨跌停标记
            tech_df['limit_up'] = (tech_df['pct_change'] >= 9.8).astype(int)
            tech_df['limit_down'] = (tech_df['pct_change'] <= -9.8).astype(int)
        
        # 计算成交量变化
        if 'volume' in tech_df.columns:
            tech_df['volume_change'] = tech_df['volume'].pct_change()
        
        self.logger.info(f"基本技术指标生成完成")
        return tech_df
    
    def format_date_columns(self, df: pd.DataFrame, date_columns: List[str] = None) -> pd.DataFrame:
        """格式化日期列
        
        Args:
            df: 待处理的数据DataFrame
            date_columns: 需要格式化的日期列列表
        
        Returns:
            格式化后的数据DataFrame
        """
        formatted_df = df.copy()
        
        if date_columns is None:
            # 自动检测日期列
            date_columns = []
            for col in formatted_df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_columns.append(col)
        
        for col in date_columns:
            if col in formatted_df.columns:
                try:
                    # 尝试多种日期格式
                    formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d%H%M%S', '%Y-%m-%d %H:%M:%S']
                    converted = False
                    
                    for fmt in formats:
                        try:
                            formatted_df[col] = pd.to_datetime(formatted_df[col], format=fmt)
                            converted = True
                            break
                        except:
                            continue
                    
                    if not converted:
                        # 尝试自动识别
                        formatted_df[col] = pd.to_datetime(formatted_df[col], errors='coerce')
                        
                    self.logger.info(f"列{col}日期格式转换完成")
                except Exception as e:
                    self.logger.error(f"列{col}日期格式转换失败: {str(e)}")
        
        return formatted_df
    
    def drop_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """去除重复行
        
        Args:
            df: 待处理的数据DataFrame
            subset: 用于判断重复的列，如果为None则考虑所有列
        
        Returns:
            去除重复行后的数据DataFrame
        """
        original_len = len(df)
        deduped_df = df.drop_duplicates(subset=subset)
        
        if len(deduped_df) < original_len:
            self.logger.info(f"去除重复行完成，原始数据量: {original_len}，去重后数据量: {len(deduped_df)}")
        
        return deduped_df
    
    def batch_process(self, data_dict: Dict[str, pd.DataFrame], 
                     process_funcs: List[str] = None) -> Dict[str, pd.DataFrame]:
        """批量处理多个DataFrame
        
        Args:
            data_dict: 以键为标识符，值为DataFrame的字典
            process_funcs: 要应用的处理函数名称列表
        
        Returns:
            处理后的DataFrame字典
        """
        if process_funcs is None:
            # 默认处理流程
            process_funcs = ['clean_data', 'normalize_data']
        
        result_dict = {}
        
        for key, df in data_dict.items():
            self.logger.info(f"开始批量处理{key}的数据")
            processed_df = df.copy()
            
            for func_name in process_funcs:
                if hasattr(self, func_name):
                    try:
                        func = getattr(self, func_name)
                        processed_df = func(processed_df)
                    except Exception as e:
                        self.logger.error(f"应用{func_name}处理{key}数据失败: {str(e)}")
                else:
                    self.logger.warning(f"处理函数{func_name}不存在")
            
            result_dict[key] = processed_df
        
        return result_dict