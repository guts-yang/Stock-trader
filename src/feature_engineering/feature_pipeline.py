"""特征工程流水线模块，整合所有特征工程功能"""

import pandas as pd
import numpy as np
import logging
import yaml
import os
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

# 导入特征工程相关模块
from .technical_indicators import TechnicalIndicators
from .fundamental_factors import FundamentalFactors
from .sentiment_analysis import SentimentAnalysis
from .factor_analysis import FactorAnalysis
from .feature_selection import FeatureSelection

class FeaturePipeline:
    """特征工程流水线类，整合所有特征工程功能"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化特征工程流水线
        
        Args:
            config: 特征工程配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'technical_indicators': {
                'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
                'rsi': {'timeperiod': 14},
                'bollinger': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
                'kdj': {'fastk_period': 9, 'slowk_period': 3, 'slowd_period': 3},
                'ma': {'timeperiods': [5, 10, 20, 60, 120, 250]},
                'volume_ma': {'timeperiods': [5, 10, 20]},
                'atr': {'timeperiod': 14},
                'obv': {},
                'cci': {'timeperiod': 14},
                'roc': {'timeperiod': 12},
                'willr': {'timeperiod': 14},
                'wr': {'timeperiod': 14},
                'bias': {'timeperiod': 6},
                'psar': {'acceleration': 0.02, 'maximum': 0.2},
                'dmi': {'timeperiod': 14}
            },
            'fundamental_factors': {
                'valuation': True,
                'profitability': True,
                'growth': True,
                'debt': True,
                'operations': True,
                'cash_flow': True,
                'turnover': True,
                'capital_structure': True
            },
            'sentiment_analysis': {
                'method': 'mixed',
                'window': 30,
                'news_source': 'eastmoney',
                'update_frequency': 'daily'
            },
            'factor_analysis': {
                'ic_method': 'spearman',
                'n_groups': 5,
                'max_lag': 5
            },
            'feature_selection': {
                'method': 'two_stage',
                'stage1_method': 'random_forest',
                'stage2_method': 'pca',
                'n_components': 128,
                'remove_low_var': True,
                'var_threshold': 0.01,
                'remove_high_corr': True,
                'corr_threshold': 0.9
            },
            'output': {
                'save_features': True,
                'output_dir': './features',
                'feature_suffix': '_features',
                'batch_size': 100
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'feature_engineering.log'
            }
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 初始化各个模块
        self.technical_indicators = TechnicalIndicators(self.config.get('technical_indicators', {}))
        self.fundamental_factors = FundamentalFactors(self.config.get('fundamental_factors', {}))
        self.sentiment_analysis = SentimentAnalysis(self.config.get('sentiment_analysis', {}))
        self.factor_analysis = FactorAnalysis()
        self.feature_selection = FeatureSelection()
        
        # 创建输出目录
        self._create_output_dir()
        
        self.logger.info("特征工程流水线初始化完成")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('FeaturePipeline')
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        logger.setLevel(log_level)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # 文件处理器
            if 'log_file' in log_config:
                file_handler = logging.FileHandler(log_config['log_file'])
                file_handler.setLevel(log_level)
            
            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            if 'log_file' in log_config:
                file_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(console_handler)
            if 'log_file' in log_config:
                logger.addHandler(file_handler)
        
        return logger
    
    def _create_output_dir(self) -> None:
        """创建输出目录"""
        output_config = self.config.get('output', {})
        output_dir = output_config.get('output_dir', './features')
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"创建输出目录: {output_dir}")
            except Exception as e:
                self.logger.error(f"创建输出目录时发生异常: {str(e)}")
    
    def load_config(self, config_path: str) -> bool:
        """从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载配置: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # 更新配置
            self.config.update(new_config)
            
            # 重新初始化各个模块
            self.technical_indicators = TechnicalIndicators(self.config.get('technical_indicators', {}))
            self.fundamental_factors = FundamentalFactors(self.config.get('fundamental_factors', {}))
            self.sentiment_analysis = SentimentAnalysis(self.config.get('sentiment_analysis', {}))
            
            self.logger.info("配置加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载配置时发生异常: {str(e)}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """保存配置到YAML文件
        
        Args:
            config_path: 配置文件保存路径
        
        Returns:
            是否保存成功
        """
        try:
            self.logger.info(f"保存配置到文件: {config_path}")
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info("配置保存完成")
            return True
        except Exception as e:
            self.logger.error(f"保存配置时发生异常: {str(e)}")
            return False
    
    def generate_technical_features(self, price_data: pd.DataFrame, 
                                  date_col: str = 'date', 
                                  asset_col: str = 'stock_code', 
                                  open_col: str = 'open', 
                                  high_col: str = 'high', 
                                  low_col: str = 'low', 
                                  close_col: str = 'close', 
                                  volume_col: str = 'volume') -> pd.DataFrame:
        """生成技术指标特征
        
        Args:
            price_data: 价格数据DataFrame
            date_col: 日期列名
            asset_col: 资产列名
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
        
        Returns:
            包含技术指标特征的DataFrame
        """
        try:
            self.logger.info("开始生成技术指标特征")
            
            # 按资产分组生成技术指标
            technical_features = []
            
            # 获取资产列表
            assets = price_data[asset_col].unique()
            
            with tqdm(total=len(assets), desc="生成技术指标") as pbar:
                for asset in assets:
                    # 筛选当前资产的数据
                    asset_data = price_data[price_data[asset_col] == asset].copy()
                    
                    if len(asset_data) < 30:  # 至少需要30天数据
                        self.logger.warning(f"资产{asset}的数据不足30天，跳过")
                        pbar.update(1)
                        continue
                    
                    # 排序数据
                    asset_data = asset_data.sort_values(date_col)
                    
                    # 计算所有技术指标
                    asset_features = self.technical_indicators.calculate_all_indicators(
                        asset_data, open_col, high_col, low_col, close_col, volume_col
                    )
                    
                    # 添加资产标识和日期
                    asset_features[asset_col] = asset
                    asset_features[date_col] = asset_data[date_col].values
                    
                    technical_features.append(asset_features)
                    
                    pbar.update(1)
            
            # 合并所有资产的特征
            if technical_features:
                result_df = pd.concat(technical_features, ignore_index=True)
            else:
                self.logger.warning("没有生成任何技术指标特征")
                return pd.DataFrame()
            
            # 重新排列列，将日期和资产列放在前面
            cols = [date_col, asset_col] + [col for col in result_df.columns if col not in [date_col, asset_col]]
            result_df = result_df[cols]
            
            self.logger.info(f"技术指标特征生成完成，共生成{result_df.shape[1] - 2}个特征")
            
            return result_df
        except Exception as e:
            self.logger.error(f"生成技术指标特征时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def generate_fundamental_features(self, financial_data: pd.DataFrame, 
                                    date_col: str = 'report_date', 
                                    asset_col: str = 'stock_code') -> pd.DataFrame:
        """生成基本面因子特征
        
        Args:
            financial_data: 财务数据DataFrame
            date_col: 日期列名
            asset_col: 资产列名
        
        Returns:
            包含基本面因子特征的DataFrame
        """
        try:
            self.logger.info("开始生成基本面因子特征")
            
            # 按资产分组生成基本面因子
            fundamental_features = []
            
            # 获取资产列表
            assets = financial_data[asset_col].unique()
            
            with tqdm(total=len(assets), desc="生成基本面因子") as pbar:
                for asset in assets:
                    # 筛选当前资产的数据
                    asset_data = financial_data[financial_data[asset_col] == asset].copy()
                    
                    if len(asset_data) == 0:
                        self.logger.warning(f"资产{asset}没有财务数据，跳过")
                        pbar.update(1)
                        continue
                    
                    # 排序数据
                    asset_data = asset_data.sort_values(date_col)
                    
                    # 计算所有基本面因子
                    asset_features = self.fundamental_factors.calculate_all_factors(asset_data)
                    
                    # 添加资产标识和日期
                    asset_features[asset_col] = asset
                    asset_features[date_col] = asset_data[date_col].values
                    
                    fundamental_features.append(asset_features)
                    
                    pbar.update(1)
            
            # 合并所有资产的特征
            if fundamental_features:
                result_df = pd.concat(fundamental_features, ignore_index=True)
            else:
                self.logger.warning("没有生成任何基本面因子特征")
                return pd.DataFrame()
            
            # 重新排列列，将日期和资产列放在前面
            cols = [date_col, asset_col] + [col for col in result_df.columns if col not in [date_col, asset_col]]
            result_df = result_df[cols]
            
            self.logger.info(f"基本面因子特征生成完成，共生成{result_df.shape[1] - 2}个特征")
            
            return result_df
        except Exception as e:
            self.logger.error(f"生成基本面因子特征时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def generate_sentiment_features(self, news_data: pd.DataFrame, 
                                  date_col: str = 'publish_date', 
                                  asset_col: str = 'stock_code', 
                                  content_col: str = 'content') -> pd.DataFrame:
        """生成情感分析特征
        
        Args:
            news_data: 新闻数据DataFrame
            date_col: 日期列名
            asset_col: 资产列名
            content_col: 内容列名
        
        Returns:
            包含情感分析特征的DataFrame
        """
        try:
            self.logger.info("开始生成情感分析特征")
            
            # 按资产分组生成情感分析特征
            sentiment_features = []
            
            # 获取资产列表
            assets = news_data[asset_col].unique()
            
            with tqdm(total=len(assets), desc="生成情感分析特征") as pbar:
                for asset in assets:
                    # 筛选当前资产的数据
                    asset_data = news_data[news_data[asset_col] == asset].copy()
                    
                    if len(asset_data) == 0:
                        self.logger.warning(f"资产{asset}没有新闻数据，跳过")
                        pbar.update(1)
                        continue
                    
                    # 排序数据
                    asset_data = asset_data.sort_values(date_col)
                    
                    # 批量分析新闻情感
                    asset_sentiment = self.sentiment_analysis.batch_news_sentiment_analysis(
                        asset_data[content_col].tolist(), 
                        asset_data[date_col].tolist()
                    )
                    
                    # 添加资产标识和日期
                    asset_sentiment[asset_col] = asset
                    asset_sentiment[date_col] = asset_data[date_col].values
                    
                    sentiment_features.append(asset_sentiment)
                    
                    pbar.update(1)
            
            # 合并所有资产的特征
            if sentiment_features:
                result_df = pd.concat(sentiment_features, ignore_index=True)
            else:
                self.logger.warning("没有生成任何情感分析特征")
                return pd.DataFrame()
            
            # 重新排列列，将日期和资产列放在前面
            cols = [date_col, asset_col] + [col for col in result_df.columns if col not in [date_col, asset_col]]
            result_df = result_df[cols]
            
            self.logger.info(f"情感分析特征生成完成，共生成{result_df.shape[1] - 2}个特征")
            
            return result_df
        except Exception as e:
            self.logger.error(f"生成情感分析特征时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def align_features_by_date(self, features_list: List[pd.DataFrame], 
                             date_col: str = 'date', 
                             asset_col: str = 'stock_code') -> pd.DataFrame:
        """按日期对齐多个特征DataFrame
        
        Args:
            features_list: 特征DataFrame列表
            date_col: 日期列名
            asset_col: 资产列名
        
        Returns:
            对齐后的特征DataFrame
        """
        try:
            self.logger.info("开始按日期对齐特征")
            
            if not features_list:
                self.logger.warning("特征列表为空，无法对齐")
                return pd.DataFrame()
            
            # 确保所有DataFrame都有日期列和资产列
            valid_features = []
            for i, df in enumerate(features_list):
                if date_col in df.columns and asset_col in df.columns:
                    valid_features.append(df)
                else:
                    self.logger.warning(f"第{i+1}个特征DataFrame缺少必要的列，跳过")
            
            if not valid_features:
                self.logger.warning("没有有效的特征DataFrame，无法对齐")
                return pd.DataFrame()
            
            # 从第一个DataFrame开始，依次合并其他DataFrame
            aligned_features = valid_features[0].copy()
            
            for i in range(1, len(valid_features)):
                # 使用日期和资产作为键进行合并
                aligned_features = pd.merge(
                    aligned_features,
                    valid_features[i],
                    on=[date_col, asset_col],
                    how='outer',
                    suffixes=(f'_feat{i-1}', f'_feat{i}')
                )
            
            # 排序
            aligned_features = aligned_features.sort_values([asset_col, date_col])
            
            self.logger.info(f"特征对齐完成，对齐后特征数: {aligned_features.shape[1]}")
            
            return aligned_features
        except Exception as e:
            self.logger.error(f"对齐特征时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_features(self, features: pd.DataFrame, 
                          date_col: str = 'date', 
                          asset_col: str = 'stock_code') -> pd.DataFrame:
        """预处理特征数据
        
        Args:
            features: 特征DataFrame
            date_col: 日期列名
            asset_col: 资产列名
        
        Returns:
            预处理后的特征DataFrame
        """
        try:
            self.logger.info("开始预处理特征数据")
            
            if features.empty:
                self.logger.warning("特征数据为空，无法预处理")
                return pd.DataFrame()
            
            # 复制数据，避免修改原始数据
            processed_features = features.copy()
            
            # 分离特征列和标识列
            id_cols = [date_col, asset_col]
            feature_cols = [col for col in processed_features.columns if col not in id_cols]
            
            # 处理缺失值
            self.logger.info("处理缺失值")
            
            # 按资产分组处理缺失值
            for asset in processed_features[asset_col].unique():
                # 筛选当前资产的数据
                asset_data = processed_features[processed_features[asset_col] == asset].copy()
                
                if len(asset_data) == 0:
                    continue
                
                # 按日期排序
                asset_data = asset_data.sort_values(date_col)
                
                # 对每个特征列，使用前向填充和后向填充处理缺失值
                for col in feature_cols:
                    if col in asset_data.columns:
                        # 前向填充
                        asset_data[col] = asset_data[col].ffill()
                        # 后向填充
                        asset_data[col] = asset_data[col].bfill()
                
                # 更新原始DataFrame
                processed_features.loc[processed_features[asset_col] == asset] = asset_data
            
            # 对于仍然有缺失值的列，使用列的均值填充
            self.logger.info("使用均值填充剩余的缺失值")
            for col in feature_cols:
                if col in processed_features.columns:
                    mean_val = processed_features[col].mean()
                    processed_features[col] = processed_features[col].fillna(mean_val)
            
            # 处理异常值
            self.logger.info("处理异常值")
            for col in feature_cols:
                if col in processed_features.columns:
                    # 使用IQR方法检测异常值
                    Q1 = processed_features[col].quantile(0.25)
                    Q3 = processed_features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # 替换异常值为边界值
                    processed_features[col] = processed_features[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.logger.info("特征预处理完成")
            
            return processed_features
        except Exception as e:
            self.logger.error(f"预处理特征时发生异常: {str(e)}")
            return features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                      date_col: str = 'date', 
                      asset_col: str = 'stock_code') -> pd.DataFrame:
        """特征选择
        
        Args:
            features: 特征DataFrame
            target: 目标变量
            date_col: 日期列名
            asset_col: 资产列名
        
        Returns:
            选择后的特征DataFrame
        """
        try:
            self.logger.info("开始进行特征选择")
            
            if features.empty or target.empty:
                self.logger.warning("特征数据或目标变量为空，无法进行特征选择")
                return pd.DataFrame()
            
            # 分离特征列和标识列
            id_cols = [date_col, asset_col]
            feature_cols = [col for col in features.columns if col not in id_cols]
            
            if len(feature_cols) == 0:
                self.logger.warning("没有可用的特征列，无法进行特征选择")
                return features
            
            # 提取特征矩阵
            X = features[feature_cols].copy()
            
            # 获取特征选择配置
            fs_config = self.config.get('feature_selection', {})
            n_components = fs_config.get('n_components', 128)
            method = fs_config.get('method', 'two_stage')
            
            # 进行特征选择
            selected_X = self.feature_selection.full_feature_selection_pipeline(
                X, target, 
                n_components=n_components, 
                method=method,
                **fs_config
            )
            
            # 将选择的特征与标识列合并
            selected_features = features[id_cols].copy()
            
            # 处理列名不匹配的情况
            common_cols = [col for col in selected_X.columns if col in features.columns]
            if common_cols:
                # 如果是原始特征，直接合并
                selected_features = pd.concat([selected_features, features[common_cols]], axis=1)
            else:
                # 如果是降维后的特征，添加到DataFrame
                for col in selected_X.columns:
                    selected_features[col] = selected_X[col].values
            
            self.logger.info(f"特征选择完成，选择的特征数: {selected_features.shape[1] - len(id_cols)}")
            
            return selected_features
        except Exception as e:
            self.logger.error(f"进行特征选择时发生异常: {str(e)}")
            return features
    
    def analyze_factors(self, features: pd.DataFrame, returns: pd.DataFrame, 
                      factor_cols: Optional[List[str]] = None, 
                      date_col: str = 'date', 
                      asset_col: str = 'stock_code', 
                      return_col: str = 'return') -> Dict[str, Dict]:
        """因子有效性分析
        
        Args:
            features: 特征DataFrame
            returns: 收益率DataFrame
            factor_cols: 要分析的因子列名列表，如果为None则分析所有特征列
            date_col: 日期列名
            asset_col: 资产列名
            return_col: 收益率列名
        
        Returns:
            包含各因子分析结果的字典
        """
        try:
            self.logger.info("开始进行因子有效性分析")
            
            if features.empty or returns.empty:
                self.logger.warning("特征数据或收益率数据为空，无法进行因子有效性分析")
                return {}
            
            # 合并特征数据和收益率数据
            merged_data = pd.merge(
                features, 
                returns[[date_col, asset_col, return_col]], 
                on=[date_col, asset_col], 
                how='inner'
            )
            
            if merged_data.empty:
                self.logger.warning("合并后的数据为空，无法进行因子有效性分析")
                return {}
            
            # 确定要分析的因子列
            if factor_cols is None:
                id_cols = [date_col, asset_col, return_col]
                factor_cols = [col for col in merged_data.columns if col not in id_cols]
            
            if not factor_cols:
                self.logger.warning("没有可用的因子列，无法进行因子有效性分析")
                return {}
            
            # 获取因子分析配置
            fa_config = self.config.get('factor_analysis', {})
            n_groups = fa_config.get('n_groups', 5)
            max_lag = fa_config.get('max_lag', 5)
            
            # 对每个因子进行分析
            analysis_results = {}
            
            with tqdm(total=len(factor_cols), desc="因子有效性分析") as pbar:
                for factor_col in factor_cols:
                    if factor_col not in merged_data.columns:
                        self.logger.warning(f"因子{factor_col}不在数据中，跳过")
                        pbar.update(1)
                        continue
                    
                    # 准备因子数据
                    factor_data = merged_data[[date_col, asset_col, factor_col]].copy()
                    return_data = merged_data[[date_col, asset_col, return_col]].copy()
                    
                    # 进行完整的因子分析
                    result = self.factor_analysis.full_factor_analysis(
                        factor_data, 
                        return_data, 
                        factor_col, 
                        return_col, 
                        date_col, 
                        asset_col, 
                        n_groups, 
                        max_lag
                    )
                    
                    if result:
                        analysis_results[factor_col] = result
                    
                    pbar.update(1)
            
            self.logger.info(f"因子有效性分析完成，分析的因子数: {len(analysis_results)}")
            
            return analysis_results
        except Exception as e:
            self.logger.error(f"进行因子有效性分析时发生异常: {str(e)}")
            return {}
    
    def generate_factor_reports(self, analysis_results: Dict[str, Dict], output_dir: str) -> None:
        """生成因子分析报告
        
        Args:
            analysis_results: 因子分析结果字典
            output_dir: 报告输出目录
        """
        try:
            self.logger.info(f"开始生成因子分析报告，输出目录: {output_dir}")
            
            # 创建输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 对每个因子生成报告
            for factor_name, result in analysis_results.items():
                # 生成报告文件路径
                report_path = os.path.join(output_dir, f"{factor_name}_analysis_report.md")
                
                # 生成报告
                report_content = self.factor_analysis.generate_factor_report(result)
                
                # 保存报告
                if report_content:
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    self.logger.info(f"因子{factor_name}的分析报告已保存到: {report_path}")
        except Exception as e:
            self.logger.error(f"生成因子分析报告时发生异常: {str(e)}")
    
    def run_full_pipeline(self, price_data: pd.DataFrame, financial_data: pd.DataFrame, 
                        news_data: pd.DataFrame, returns: pd.DataFrame, 
                        date_col: str = 'date', 
                        asset_col: str = 'stock_code', 
                        open_col: str = 'open', 
                        high_col: str = 'high', 
                        low_col: str = 'low', 
                        close_col: str = 'close', 
                        volume_col: str = 'volume', 
                        content_col: str = 'content', 
                        return_col: str = 'return') -> pd.DataFrame:
        """运行完整的特征工程流水线
        
        Args:
            price_data: 价格数据DataFrame
            financial_data: 财务数据DataFrame
            news_data: 新闻数据DataFrame
            returns: 收益率DataFrame
            date_col: 日期列名
            asset_col: 资产列名
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            content_col: 内容列名
            return_col: 收益率列名
        
        Returns:
            最终的特征矩阵DataFrame
        """
        try:
            self.logger.info("开始运行完整的特征工程流水线")
            
            # 阶段1: 生成各类特征
            self.logger.info("===== 阶段1: 生成各类特征 =====")
            
            # 生成技术指标特征
            technical_features = self.generate_technical_features(
                price_data, date_col, asset_col, open_col, high_col, low_col, close_col, volume_col
            )
            
            # 生成基本面因子特征
            fundamental_features = self.generate_fundamental_features(
                financial_data, date_col, asset_col
            )
            
            # 生成情感分析特征
            sentiment_features = self.generate_sentiment_features(
                news_data, date_col, asset_col, content_col
            )
            
            # 阶段2: 对齐和合并特征
            self.logger.info("===== 阶段2: 对齐和合并特征 =====")
            
            # 收集非空的特征DataFrame
            all_features = []
            if not technical_features.empty:
                all_features.append(technical_features)
            if not fundamental_features.empty:
                all_features.append(fundamental_features)
            if not sentiment_features.empty:
                all_features.append(sentiment_features)
            
            # 对齐特征
            aligned_features = self.align_features_by_date(all_features, date_col, asset_col)
            
            if aligned_features.empty:
                self.logger.error("特征对齐失败，流水线中断")
                return pd.DataFrame()
            
            # 阶段3: 特征预处理
            self.logger.info("===== 阶段3: 特征预处理 =====")
            
            # 预处理特征
            preprocessed_features = self.preprocess_features(aligned_features, date_col, asset_col)
            
            if preprocessed_features.empty:
                self.logger.error("特征预处理失败，流水线中断")
                return pd.DataFrame()
            
            # 阶段4: 特征选择
            self.logger.info("===== 阶段4: 特征选择 =====")
            
            # 合并目标变量
            features_with_target = pd.merge(
                preprocessed_features, 
                returns[[date_col, asset_col, return_col]], 
                on=[date_col, asset_col], 
                how='inner'
            )
            
            if features_with_target.empty:
                self.logger.error("合并特征和目标变量失败，流水线中断")
                return pd.DataFrame()
            
            # 特征选择
            selected_features = self.select_features(
                features_with_target, 
                features_with_target[return_col], 
                date_col, 
                asset_col
            )
            
            if selected_features.empty:
                self.logger.error("特征选择失败，流水线中断")
                return pd.DataFrame()
            
            # 阶段5: 因子有效性分析（可选）
            self.logger.info("===== 阶段5: 因子有效性分析 =====")
            
            # 确定要分析的因子列
            id_cols = [date_col, asset_col]
            factor_cols = [col for col in selected_features.columns if col not in id_cols]
            
            # 合并收益率数据进行分析
            analysis_data = pd.merge(
                selected_features, 
                returns[[date_col, asset_col, return_col]], 
                on=[date_col, asset_col], 
                how='inner'
            )
            
            # 进行因子有效性分析
            factor_analysis_results = self.analyze_factors(
                analysis_data, 
                analysis_data, 
                factor_cols, 
                date_col, 
                asset_col, 
                return_col
            )
            
            # 生成因子分析报告
            if factor_analysis_results:
                output_config = self.config.get('output', {})
                output_dir = output_config.get('output_dir', './features')
                report_dir = os.path.join(output_dir, 'factor_reports')
                self.generate_factor_reports(factor_analysis_results, report_dir)
            
            # 阶段6: 保存结果
            self.logger.info("===== 阶段6: 保存结果 =====")
            
            output_config = self.config.get('output', {})
            if output_config.get('save_features', True):
                output_dir = output_config.get('output_dir', './features')
                feature_suffix = output_config.get('feature_suffix', '_features')
                
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(output_dir, f"features_{timestamp}{feature_suffix}.csv")
                
                # 保存特征
                selected_features.to_csv(output_path, index=False)
                
                self.logger.info(f"特征数据已保存到: {output_path}")
            
            self.logger.info("完整的特征工程流水线运行完成")
            
            return selected_features
        except Exception as e:
            self.logger.error(f"运行特征工程流水线时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def run_batch_pipeline(self, price_data_dict: Dict[str, pd.DataFrame], 
                         financial_data_dict: Dict[str, pd.DataFrame], 
                         news_data_dict: Dict[str, pd.DataFrame], 
                         returns_dict: Dict[str, pd.DataFrame], 
                         date_col: str = 'date', 
                         asset_col: str = 'stock_code', 
                         open_col: str = 'open', 
                         high_col: str = 'high', 
                         low_col: str = 'low', 
                         close_col: str = 'close', 
                         volume_col: str = 'volume', 
                         content_col: str = 'content', 
                         return_col: str = 'return') -> Dict[str, pd.DataFrame]:
        """批量运行特征工程流水线
        
        Args:
            price_data_dict: 资产到价格数据的字典
            financial_data_dict: 资产到财务数据的字典
            news_data_dict: 资产到新闻数据的字典
            returns_dict: 资产到收益率数据的字典
            date_col: 日期列名
            asset_col: 资产列名
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            content_col: 内容列名
            return_col: 收益率列名
        
        Returns:
            资产到特征矩阵的字典
        """
        try:
            self.logger.info("开始批量运行特征工程流水线")
            
            output_config = self.config.get('output', {})
            batch_size = output_config.get('batch_size', 100)
            
            # 获取所有资产
            all_assets = set(price_data_dict.keys())
            all_assets.update(financial_data_dict.keys())
            all_assets.update(news_data_dict.keys())
            all_assets.update(returns_dict.keys())
            all_assets = list(all_assets)
            
            # 分批处理
            batch_results = {}
            total_batches = (len(all_assets) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(all_assets))
                batch_assets = all_assets[start_idx:end_idx]
                
                self.logger.info(f"处理批次 {batch_idx + 1}/{total_batches}，资产数量: {len(batch_assets)}")
                
                # 准备批次数据
                batch_price_data = []
                batch_financial_data = []
                batch_news_data = []
                batch_returns = []
                
                for asset in batch_assets:
                    # 添加价格数据
                    if asset in price_data_dict:
                        asset_price_data = price_data_dict[asset].copy()
                        asset_price_data[asset_col] = asset
                        batch_price_data.append(asset_price_data)
                    
                    # 添加财务数据
                    if asset in financial_data_dict:
                        asset_financial_data = financial_data_dict[asset].copy()
                        asset_financial_data[asset_col] = asset
                        batch_financial_data.append(asset_financial_data)
                    
                    # 添加新闻数据
                    if asset in news_data_dict:
                        asset_news_data = news_data_dict[asset].copy()
                        asset_news_data[asset_col] = asset
                        batch_news_data.append(asset_news_data)
                    
                    # 添加收益率数据
                    if asset in returns_dict:
                        asset_returns = returns_dict[asset].copy()
                        asset_returns[asset_col] = asset
                        batch_returns.append(asset_returns)
                
                # 合并批次数据
                if batch_price_data:
                    batch_price_df = pd.concat(batch_price_data, ignore_index=True)
                else:
                    batch_price_df = pd.DataFrame()
                
                if batch_financial_data:
                    batch_financial_df = pd.concat(batch_financial_data, ignore_index=True)
                else:
                    batch_financial_df = pd.DataFrame()
                
                if batch_news_data:
                    batch_news_df = pd.concat(batch_news_data, ignore_index=True)
                else:
                    batch_news_df = pd.DataFrame()
                
                if batch_returns:
                    batch_returns_df = pd.concat(batch_returns, ignore_index=True)
                else:
                    batch_returns_df = pd.DataFrame()
                
                # 运行流水线
                batch_features = self.run_full_pipeline(
                    batch_price_df, batch_financial_df, batch_news_df, batch_returns_df,
                    date_col, asset_col, open_col, high_col, low_col, close_col, volume_col,
                    content_col, return_col
                )
                
                # 按资产保存结果
                if not batch_features.empty:
                    for asset in batch_assets:
                        asset_features = batch_features[batch_features[asset_col] == asset].copy()
                        if not asset_features.empty:
                            batch_results[asset] = asset_features
            
            self.logger.info(f"批量运行特征工程流水线完成，成功处理{len(batch_results)}个资产")
            
            return batch_results
        except Exception as e:
            self.logger.error(f"批量运行特征工程流水线时发生异常: {str(e)}")
            return {}
    
    def update_features(self, existing_features: pd.DataFrame, new_data: Dict[str, pd.DataFrame], 
                      date_col: str = 'date', 
                      asset_col: str = 'stock_code') -> pd.DataFrame:
        """更新现有特征
        
        Args:
            existing_features: 现有特征DataFrame
            new_data: 包含新数据的字典，键为数据类型，值为新数据DataFrame
            date_col: 日期列名
            asset_col: 资产列名
        
        Returns:
            更新后的特征DataFrame
        """
        try:
            self.logger.info("开始更新特征数据")
            
            if existing_features.empty:
                self.logger.warning("现有特征数据为空，无法更新")
                return pd.DataFrame()
            
            # 获取现有特征的最新日期
            latest_date = existing_features[date_col].max()
            self.logger.info(f"现有特征的最新日期: {latest_date}")
            
            # 准备更新数据
            update_price_data = new_data.get('price', pd.DataFrame())
            update_financial_data = new_data.get('financial', pd.DataFrame())
            update_news_data = new_data.get('news', pd.DataFrame())
            update_returns = new_data.get('returns', pd.DataFrame())
            
            # 筛选新数据（日期大于最新日期）
            if not update_price_data.empty:
                update_price_data = update_price_data[update_price_data[date_col] > latest_date]
            
            if not update_financial_data.empty:
                update_financial_data = update_financial_data[update_financial_data[date_col] > latest_date]
            
            if not update_news_data.empty:
                update_news_data = update_news_data[update_news_data[date_col] > latest_date]
            
            if not update_returns.empty:
                update_returns = update_returns[update_returns[date_col] > latest_date]
            
            # 如果没有新数据，直接返回现有特征
            if (update_price_data.empty and update_financial_data.empty and 
                update_news_data.empty and update_returns.empty):
                self.logger.info("没有找到新数据，无需更新特征")
                return existing_features
            
            # 运行流水线生成新特征
            self.logger.info("生成新特征")
            new_features = self.run_full_pipeline(
                update_price_data, update_financial_data, update_news_data, update_returns,
                date_col, asset_col
            )
            
            # 合并现有特征和新特征
            if not new_features.empty:
                updated_features = pd.concat([existing_features, new_features], ignore_index=True)
                # 去重（按日期和资产）
                updated_features = updated_features.drop_duplicates(subset=[date_col, asset_col], keep='last')
                # 排序
                updated_features = updated_features.sort_values([asset_col, date_col])
                
                self.logger.info(f"特征更新完成，新增{new_features.shape[0]}条记录")
                
                return updated_features
            else:
                self.logger.warning("生成新特征失败，返回现有特征")
                return existing_features
        except Exception as e:
            self.logger.error(f"更新特征时发生异常: {str(e)}")
            return existing_features