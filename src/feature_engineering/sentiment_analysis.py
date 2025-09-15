"""情感分析模块，负责处理财经新闻和社交媒体数据的情感分析"""

import pandas as pd
import numpy as np
import logging
import re
import jieba
import jieba.analyse
import snownlp
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor
import time

class SentimentAnalysis:
    """情感分析类，封装了文本情感分析的核心功能"""
    
    def __init__(self):
        """初始化情感分析类"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("情感分析模块初始化完成")
        
        # 设置中文停用词表
        self.stopwords = self._load_stopwords()
        
        # 自定义财经领域关键词
        self.finance_keywords = {
            'positive': ['增长', '上涨', '盈利', '创新高', '利好', '超预期', '业绩好', '高分红', '回购', '增持', 
                        '突破', '扩张', '并购', '重组', '政策支持', '税收优惠', '订单增加', '产能释放', '毛利率提升'],
            'negative': ['下跌', '亏损', '低于预期', '利空', '业绩下滑', '减持', '质押', '违约', '监管', '处罚', 
                        '下降', '萎缩', '裁员', '债务', '诉讼', '退市', '商誉减值', '毛利率下降', '亏损', '造假']
        }
        
        # 初始化jieba分词器
        self._init_jieba()
    
    def _load_stopwords(self) -> set:
        """加载中文停用词表
        
        Returns:
            停用词集合
        """
        try:
            # 这里使用内置的停用词表，可以替换为外部文件
            stopwords = {
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', 
                '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '我们', '为', '来', '对', '而', 
                '以', '于', '上', '下', '左', '右', '前', '后', '中', '外', '里', '内', '外', '大', '小', '多', '少', 
                '高', '低', '长', '短', '宽', '窄', '胖', '瘦', '男', '女', '老', '少', '们', '之', '乎', '者', '也', 
                '矣', '焉', '哉', '乎', '其', '而', '且', '则', '乃', '若', '为', '所', '与', '及', '于', '因', '由', 
                '以', '而', '且', '则', '虽', '然', '但', '而', '然', '况', '且', '苟', '纵', '如', '若', '倘', '使', 
                '设', '假', '若', '或', '抑', '将', '及', '即', '便', '乃', '于', '是', '尔', '其', '彼', '此', '斯', 
                '兹', '夫', '盖', '惟', '窃', '伏', '敢', '蒙', '窃', '伏', '请', '谨', '敬', '幸', '猥', '枉', '蒙', 
                '承', '荷', '蒙', '惠', '垂', '赐', '幸', '甚', '不胜', '至', '极', '之', '矣', '也', '焉', '哉', '乎'
            }
            self.logger.info("中文停用词表加载完成")
            return stopwords
        except Exception as e:
            self.logger.error(f"加载停用词表时发生异常: {str(e)}")
            return set()
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        try:
            # 添加自定义词典（如果需要）
            # jieba.load_userdict("custom_dict.txt")
            
            # 设置TF-IDF关键词提取
            self.logger.info("jieba分词器初始化完成")
        except Exception as e:
            self.logger.error(f"初始化jieba分词器时发生异常: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """清洗文本数据
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        try:
            # 移除HTML标签
            text = re.sub(r'<[^>]*>', '', text)
            # 移除URL
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # 移除特殊字符和数字
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
            # 移除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            self.logger.error(f"清洗文本时发生异常: {str(e)}")
            return ""
    
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词
        
        Args:
            text: 清洗后的文本
        
        Returns:
            分词后的词语列表
        """
        try:
            # 使用jieba进行分词
            words = jieba.cut(text)
            # 过滤停用词
            words = [word for word in words if word not in self.stopwords and len(word) > 1]
            
            return words
        except Exception as e:
            self.logger.error(f"分词时发生异常: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取文本中的关键词
        
        Args:
            text: 清洗后的文本
            top_k: 返回的关键词数量
        
        Returns:
            关键词列表
        """
        try:
            # 使用TF-IDF算法提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
            
            return keywords
        except Exception as e:
            self.logger.error(f"提取关键词时发生异常: {str(e)}")
            return []
    
    def analyze_sentiment_snownlp(self, text: str) -> float:
        """使用SnowNLP进行情感分析
        
        Args:
            text: 要分析的文本
        
        Returns:
            情感分数，范围[0,1]，越接近1越积极
        """
        try:
            if not text or not isinstance(text, str):
                return 0.5  # 中性
            
            # 清洗文本
            clean_text = self._clean_text(text)
            if not clean_text:
                return 0.5
            
            # 使用SnowNLP进行情感分析
            s = snownlp.SnowNLP(clean_text)
            sentiment_score = s.sentiments
            
            return sentiment_score
        except Exception as e:
            self.logger.error(f"使用SnowNLP进行情感分析时发生异常: {str(e)}")
            return 0.5
    
    def analyze_sentiment_keyword(self, text: str) -> float:
        """基于关键词的情感分析
        
        Args:
            text: 要分析的文本
        
        Returns:
            情感分数，范围[-1,1]，正值表示积极，负值表示消极
        """
        try:
            if not text or not isinstance(text, str):
                return 0  # 中性
            
            # 清洗文本
            clean_text = self._clean_text(text)
            if not clean_text:
                return 0
            
            # 分词
            words = self._tokenize(clean_text)
            if not words:
                return 0
            
            # 计算正负关键词数量
            positive_count = 0
            negative_count = 0
            
            for word in words:
                if word in self.finance_keywords['positive']:
                    positive_count += 1
                elif word in self.finance_keywords['negative']:
                    negative_count += 1
            
            # 计算情感分数
            total_count = positive_count + negative_count
            if total_count == 0:
                return 0
            
            sentiment_score = (positive_count - negative_count) / total_count
            
            return sentiment_score
        except Exception as e:
            self.logger.error(f"基于关键词的情感分析时发生异常: {str(e)}")
            return 0
    
    def analyze_sentiment(self, text: str, method: str = 'hybrid') -> float:
        """综合情感分析
        
        Args:
            text: 要分析的文本
            method: 分析方法，可选'snownlp'、'keyword'、'hybrid'
        
        Returns:
            情感分数，范围[-1,1]，正值表示积极，负值表示消极
        """
        try:
            if method == 'snownlp':
                # 使用SnowNLP，结果从[0,1]映射到[-1,1]
                score = self.analyze_sentiment_snownlp(text)
                return 2 * (score - 0.5)
            elif method == 'keyword':
                # 使用关键词分析
                return self.analyze_sentiment_keyword(text)
            elif method == 'hybrid':
                # 混合方法，取两者的平均值
                score1 = 2 * (self.analyze_sentiment_snownlp(text) - 0.5)
                score2 = self.analyze_sentiment_keyword(text)
                return (score1 + score2) / 2
            else:
                self.logger.warning(f"不支持的情感分析方法: {method}")
                return 0
        except Exception as e:
            self.logger.error(f"综合情感分析时发生异常: {str(e)}")
            return 0
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame, text_column: str = 'content', 
                              method: str = 'hybrid') -> pd.DataFrame:
        """批量分析新闻情感
        
        Args:
            news_df: 包含新闻数据的DataFrame
            text_column: 文本内容列名
            method: 分析方法
        
        Returns:
            添加了情感分数列的DataFrame
        """
        result_df = news_df.copy()
        
        try:
            if text_column not in news_df.columns:
                self.logger.error(f"DataFrame中缺少列: {text_column}")
                return result_df
            
            self.logger.info(f"开始批量分析{len(news_df)}条新闻的情感")
            
            # 使用多线程加速处理
            with ThreadPoolExecutor(max_workers=min(10, len(news_df))) as executor:
                sentiments = list(executor.map(
                    lambda text: self.analyze_sentiment(text, method), 
                    news_df[text_column]
                ))
            
            # 添加情感分数列
            result_df[f'sentiment_score_{method}'] = sentiments
            
            # 添加情感标签
            def get_sentiment_label(score):
                if score > 0.2:
                    return 'positive'
                elif score < -0.2:
                    return 'negative'
                else:
                    return 'neutral'
            
            result_df[f'sentiment_label_{method}'] = result_df[f'sentiment_score_{method}'].apply(get_sentiment_label)
            
            self.logger.info("新闻情感分析完成")
        except Exception as e:
            self.logger.error(f"批量分析新闻情感时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_sentiment_rolling(self, sentiment_df: pd.DataFrame, score_column: str, 
                                  window: int = 5) -> pd.DataFrame:
        """计算情感分数的滚动统计
        
        Args:
            sentiment_df: 包含情感分数的DataFrame
            score_column: 情感分数列名
            window: 滚动窗口大小
        
        Returns:
            添加了滚动统计列的DataFrame
        """
        result_df = sentiment_df.copy()
        
        try:
            if score_column not in sentiment_df.columns:
                self.logger.error(f"DataFrame中缺少列: {score_column}")
                return result_df
            
            # 确保DataFrame按日期排序
            if 'date' in sentiment_df.columns:
                result_df = result_df.sort_values('date')
            
            # 计算滚动均值
            result_df[f'{score_column}_rolling_mean_{window}'] = result_df[score_column].rolling(window=window).mean()
            
            # 计算滚动标准差
            result_df[f'{score_column}_rolling_std_{window}'] = result_df[score_column].rolling(window=window).std()
            
            # 计算滚动最大值
            result_df[f'{score_column}_rolling_max_{window}'] =result_df[score_column].rolling(window=window).max()
            
            # 计算滚动最小值
            result_df[f'{score_column}_rolling_min_{window}'] = result_df[score_column].rolling(window=window).min()
            
            self.logger.info(f"计算情感分数滚动统计完成，窗口大小: {window}")
        except Exception as e:
            self.logger.error(f"计算情感分数滚动统计时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_stock_sentiment(self, news_df: pd.DataFrame, stock_code: str, 
                                date_column: str = 'date', text_column: str = 'content', 
                                method: str = 'hybrid', window: int = 5) -> pd.DataFrame:
        """计算单只股票的情感分数时间序列
        
        Args:
            news_df: 包含新闻数据的DataFrame
            stock_code: 股票代码
            date_column: 日期列名
            text_column: 文本内容列名
            method: 分析方法
            window: 滚动窗口大小
        
        Returns:
            股票情感分数时间序列DataFrame
        """
        try:
            required_columns = [date_column, text_column]
            if not all(col in news_df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in news_df.columns]
                self.logger.error(f"DataFrame中缺少必要的列: {missing_cols}")
                return pd.DataFrame()
            
            # 确保日期列格式正确
            if not pd.api.types.is_datetime64_any_dtype(news_df[date_column]):
                try:
                    news_df[date_column] = pd.to_datetime(news_df[date_column])
                except Exception as e:
                    self.logger.error(f"转换日期列时发生异常: {str(e)}")
                    return pd.DataFrame()
            
            self.logger.info(f"开始计算股票{stock_code}的情感分数时间序列")
            
            # 分析每条新闻的情感
            sentiment_df = self.analyze_news_sentiment(news_df, text_column, method)
            
            # 按日期分组，计算每日情感指标
            daily_sentiment = sentiment_df.groupby(date_column).agg({
                f'sentiment_score_{method}': ['mean', 'median', 'std', 'count'],
                f'sentiment_label_{method}': lambda x: x.value_counts().to_dict()
            }).reset_index()
            
            # 重命名列
            daily_sentiment.columns = [
                'date', 
                f'daily_sentiment_mean', f'daily_sentiment_median', f'daily_sentiment_std', 'news_count',
                'sentiment_distribution'
            ]
            
            # 计算每日正面、负面和中性新闻比例
            def extract_sentiment_counts(dist):
                positive = dist.get('positive', 0)
                negative = dist.get('negative', 0)
                neutral = dist.get('neutral', 0)
                total = positive + negative + neutral
                if total == 0:
                    return 0, 0, 0
                return positive/total, negative/total, neutral/total
            
            daily_sentiment[['positive_ratio', 'negative_ratio', 'neutral_ratio']] = daily_sentiment['sentiment_distribution'].apply(lambda x: pd.Series(extract_sentiment_counts(x)))
            
            # 计算滚动统计
            daily_sentiment = self.calculate_sentiment_rolling(
                daily_sentiment, 'daily_sentiment_mean', window
            )
            
            # 添加股票代码
            daily_sentiment['stock_code'] = stock_code
            
            self.logger.info(f"股票{stock_code}的情感分数时间序列计算完成")
            
            return daily_sentiment
        except Exception as e:
            self.logger.error(f"计算股票情感分数时间序列时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def calculate_market_sentiment(self, news_df: pd.DataFrame, date_column: str = 'date', 
                                 text_column: str = 'content', method: str = 'hybrid', 
                                 window: int = 5) -> pd.DataFrame:
        """计算市场整体情感分数
        
        Args:
            news_df: 包含市场新闻数据的DataFrame
            date_column: 日期列名
            text_column: 文本内容列名
            method: 分析方法
            window: 滚动窗口大小
        
        Returns:
            市场情感分数时间序列DataFrame
        """
        try:
            self.logger.info("开始计算市场整体情感分数")
            
            # 使用单只股票的方法计算市场情感
            market_sentiment = self.calculate_stock_sentiment(
                news_df, 'MARKET', date_column, text_column, method, window
            )
            
            # 重命名股票代码列为市场代码
            market_sentiment = market_sentiment.rename(columns={'stock_code': 'market_code'})
            
            self.logger.info("市场整体情感分数计算完成")
            
            return market_sentiment
        except Exception as e:
            self.logger.error(f"计算市场整体情感分数时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def sentiment_to_factor(self, sentiment_df: pd.DataFrame, score_column: str, 
                          normalize: bool = True) -> pd.DataFrame:
        """将情感分数转换为交易因子
        
        Args:
            sentiment_df: 包含情感分数的DataFrame
            score_column: 情感分数列名
            normalize: 是否标准化
        
        Returns:
            添加了情感因子列的DataFrame
        """
        result_df = sentiment_df.copy()
        
        try:
            if score_column not in sentiment_df.columns:
                self.logger.error(f"DataFrame中缺少列: {score_column}")
                return result_df
            
            # 创建情感因子
            result_df['sentiment_factor'] = result_df[score_column]
            
            # 如果需要标准化
            if normalize:
                # Z-score标准化
                mean = result_df['sentiment_factor'].mean()
                std = result_df['sentiment_factor'].std()
                if std > 0:
                    result_df['sentiment_factor'] = (result_df['sentiment_factor'] - mean) / std
                else:
                    result_df['sentiment_factor'] = 0
                
            self.logger.info("情感分数转换为交易因子完成")
        except Exception as e:
            self.logger.error(f"将情感分数转换为交易因子时发生异常: {str(e)}")
        
        return result_df
    
    def add_event_sentiment(self, stock_data: pd.DataFrame, news_df: pd.DataFrame, 
                           stock_code_column: str = 'stock_code', date_column: str = 'date',
                           method: str = 'hybrid') -> pd.DataFrame:
        """为股票数据添加事件情感信息
        
        Args:
            stock_data: 股票价格数据DataFrame
            news_df: 包含新闻数据的DataFrame
            stock_code_column: 股票代码列名
            date_column: 日期列名
            method: 分析方法
        
        Returns:
            添加了事件情感信息的股票数据DataFrame
        """
        result_df = stock_data.copy()
        
        try:
            # 确保日期列格式正确
            for df in [stock_data, news_df]:
                if date_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                    try:
                        df[date_column] = pd.to_datetime(df[date_column])
                    except Exception as e:
                        self.logger.error(f"转换日期列时发生异常: {str(e)}")
                        return result_df
            
            self.logger.info("开始为股票数据添加事件情感信息")
            
            # 获取所有唯一的股票代码
            stock_codes = stock_data[stock_code_column].unique()
            
            merged_dfs = []
            
            for stock_code in stock_codes:
                # 筛选当前股票的价格数据
                stock_df = stock_data[stock_data[stock_code_column] == stock_code].copy()
                
                # 筛选当前股票的新闻数据
                # 这里假设新闻数据中也有股票代码列，实际情况可能需要根据标题或内容进行匹配
                stock_news = news_df[news_df[stock_code_column] == stock_code].copy()
                
                if stock_news.empty:
                    # 如果没有该股票的新闻数据，添加默认值
                    stock_df['has_news'] = False
                    stock_df['news_count'] = 0
                    stock_df[f'sentiment_score_{method}'] = 0
                    merged_dfs.append(stock_df)
                    continue
                
                # 分析新闻情感
                sentiment_news = self.analyze_news_sentiment(stock_news, method=method)
                
                # 按日期聚合新闻情感数据
                daily_sentiment = sentiment_news.groupby(date_column).agg({
                    f'sentiment_score_{method}': 'mean',
                    f'sentiment_label_{method}': 'count'
                }).reset_index()
                
                daily_sentiment.columns = [date_column, f'sentiment_score_{method}', 'news_count']
                
                # 将情感数据合并到股票数据中
                merged = pd.merge(stock_df, daily_sentiment, on=[date_column], how='left')
                
                # 填充缺失值
                merged['has_news'] = merged['news_count'].notna() & (merged['news_count'] > 0)
                merged['news_count'] = merged['news_count'].fillna(0)
                merged[f'sentiment_score_{method}'] = merged[f'sentiment_score_{method}'].fillna(0)
                
                merged_dfs.append(merged)
            
            # 合并所有股票的数据
            result_df = pd.concat(merged_dfs, ignore_index=True)
            
            self.logger.info("为股票数据添加事件情感信息完成")
        except Exception as e:
            self.logger.error(f"为股票数据添加事件情感信息时发生异常: {str(e)}")
        
        return result_df
    
    def batch_process_news_data(self, news_data_list: List[Dict], batch_size: int = 100) -> List[Dict]:
        """批量处理新闻数据
        
        Args:
            news_data_list: 新闻数据列表，每个元素是包含新闻信息的字典
            batch_size: 批次大小
        
        Returns:
            处理后的新闻数据列表
        """
        try:
            self.logger.info(f"开始批量处理{len(news_data_list)}条新闻数据")
            
            processed_news = []
            
            # 分批处理
            for i in range(0, len(news_data_list), batch_size):
                batch = news_data_list[i:i+batch_size]
                
                # 转换为DataFrame
                batch_df = pd.DataFrame(batch)
                
                # 分析情感
                if 'content' in batch_df.columns:
                    batch_df = self.analyze_news_sentiment(batch_df)
                elif 'title' in batch_df.columns:
                    batch_df = self.analyze_news_sentiment(batch_df, text_column='title')
                
                # 转换回字典列表
                processed_news.extend(batch_df.to_dict('records'))
                
                # 添加处理延迟，避免API调用过于频繁
                if i + batch_size < len(news_data_list):
                    time.sleep(0.5)
            
            self.logger.info("批量处理新闻数据完成")
            
            return processed_news
        except Exception as e:
            self.logger.error(f"批量处理新闻数据时发生异常: {str(e)}")
            return news_data_list