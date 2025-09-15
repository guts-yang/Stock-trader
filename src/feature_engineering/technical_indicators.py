"""技术指标计算模块，使用TA-Lib计算各种技术指标"""

import pandas as pd
import numpy as np
import logging
import talib as ta
from typing import Dict, List, Optional, Union

class TechnicalIndicators:
    """技术指标计算类，封装了常用的技术指标计算方法"""
    
    def __init__(self):
        """初始化技术指标计算类"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("技术指标计算模块初始化完成")
        
        # 记录已支持的技术指标
        self.supported_indicators = {
            'MACD': self.calculate_macd,
            'RSI': self.calculate_rsi,
            'BBANDS': self.calculate_bbands,
            'MA': self.calculate_ma,
            'EMA': self.calculate_ema,
            'KDJ': self.calculate_kdj,
            'MACD_SIGNAL': self.calculate_macd,
            'MACD_HIST': self.calculate_macd,
            'ROC': self.calculate_roc,
            'ATR': self.calculate_atr,
            'OBV': self.calculate_obv,
            'WILLR': self.calculate_willr,
            'ADX': self.calculate_adx,
            'CCI': self.calculate_cci,
            'WR': self.calculate_willr,
            'BIAS': self.calculate_bias,
            'PSY': self.calculate_psy
        }
    
    def calculate_macd(self, df: pd.DataFrame, fastperiod: int = 12, 
                       slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 包含收盘价的DataFrame
            fastperiod: 快线周期
            slowperiod: 慢线周期
            signalperiod: 信号线周期
        
        Returns:
            添加了MACD相关列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算MACD
            macd, macd_signal, macd_hist = ta.MACD(
                df['close'].values, 
                fastperiod=fastperiod, 
                slowperiod=slowperiod, 
                signalperiod=signalperiod
            )
            
            # 添加MACD相关列
            result_df[f'MACD_{fastperiod}_{slowperiod}_{signalperiod}'] = macd
            result_df[f'MACD_SIGNAL_{fastperiod}_{slowperiod}_{signalperiod}'] = macd_signal
            result_df[f'MACD_HIST_{fastperiod}_{slowperiod}_{signalperiod}'] = macd_hist
            
            self.logger.info(f"MACD指标计算完成，参数: fastperiod={fastperiod}, slowperiod={slowperiod}, signalperiod={signalperiod}")
        except Exception as e:
            self.logger.error(f"计算MACD指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_rsi(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算RSI指标
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了RSI列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算RSI
            rsi = ta.RSI(df['close'].values, timeperiod=timeperiod)
            
            # 添加RSI列
            result_df[f'RSI_{timeperiod}'] = rsi
            
            self.logger.info(f"RSI指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算RSI指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_bbands(self, df: pd.DataFrame, timeperiod: int = 20, 
                        nbdevup: int = 2, nbdevdn: int = 2) -> pd.DataFrame:
        """计算布林带指标
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
            nbdevup: 上轨标准差倍数
            nbdevdn: 下轨标准差倍数
        
        Returns:
            添加了布林带相关列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算布林带
            upperband, middleband, lowerband = ta.BBANDS(
                df['close'].values, 
                timeperiod=timeperiod, 
                nbdevup=nbdevup, 
                nbdevdn=nbdevdn
            )
            
            # 添加布林带相关列
            result_df[f'BBANDS_UPPER_{timeperiod}_{nbdevup}'] = upperband
            result_df[f'BBANDS_MIDDLE_{timeperiod}'] = middleband
            result_df[f'BBANDS_LOWER_{timeperiod}_{nbdevdn}'] = lowerband
            # 计算带宽
            result_df[f'BBANDS_WIDTH_{timeperiod}_{nbdevup}_{nbdevdn}'] = (upperband - lowerband) / middleband
            # 计算收盘价与布林带中轨的比率
            result_df[f'BBANDS_RATIO_{timeperiod}'] = df['close'] / middleband
            
            self.logger.info(f"布林带指标计算完成，参数: timeperiod={timeperiod}, nbdevup={nbdevup}, nbdevdn={nbdevdn}")
        except Exception as e:
            self.logger.error(f"计算布林带指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_ma(self, df: pd.DataFrame, timeperiod: int = 5) -> pd.DataFrame:
        """计算移动平均线指标
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了MA列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算简单移动平均线
            ma = ta.SMA(df['close'].values, timeperiod=timeperiod)
            
            # 添加MA列
            result_df[f'MA_{timeperiod}'] = ma
            # 计算收盘价与MA的比率
            result_df[f'MA_RATIO_{timeperiod}'] = df['close'] / ma
            
            self.logger.info(f"移动平均线指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算移动平均线指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_ema(self, df: pd.DataFrame, timeperiod: int = 5) -> pd.DataFrame:
        """计算指数移动平均线指标
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了EMA列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算指数移动平均线
            ema = ta.EMA(df['close'].values, timeperiod=timeperiod)
            
            # 添加EMA列
            result_df[f'EMA_{timeperiod}'] = ema
            # 计算收盘价与EMA的比率
            result_df[f'EMA_RATIO_{timeperiod}'] = df['close'] / ema
            
            self.logger.info(f"指数移动平均线指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算指数移动平均线指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_kdj(self, df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """计算KDJ指标
        
        Args:
            df: 包含最高价、最低价、收盘价的DataFrame
            n: RSV计算周期
            m1: K值平滑周期
            m2: D值平滑周期
        
        Returns:
            添加了KDJ相关列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                self.logger.error("DataFrame中缺少必要的列(high, low, close)")
                return result_df
            
            # 计算RSV
            low_n = df['low'].rolling(window=n).min()
            high_n = df['high'].rolling(window=n).max()
            rsv = (df['close'] - low_n) / (high_n - low_n) * 100
            
            # 计算K、D、J值
            k = rsv.ewm(alpha=1/m1, adjust=False).mean()
            d = k.ewm(alpha=1/m2, adjust=False).mean()
            j = 3 * k - 2 * d
            
            # 添加KDJ相关列
            result_df[f'KDJ_K_{n}_{m1}_{m2}'] = k
            result_df[f'KDJ_D_{n}_{m1}_{m2}'] = d
            result_df[f'KDJ_J_{n}_{m1}_{m2}'] = j
            
            self.logger.info(f"KDJ指标计算完成，参数: n={n}, m1={m1}, m2={m2}")
        except Exception as e:
            self.logger.error(f"计算KDJ指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_roc(self, df: pd.DataFrame, timeperiod: int = 12) -> pd.DataFrame:
        """计算ROC指标（变动率指标）
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了ROC列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 使用talib计算ROC
            roc = ta.ROC(df['close'].values, timeperiod=timeperiod)
            
            # 添加ROC列
            result_df[f'ROC_{timeperiod}'] = roc
            
            self.logger.info(f"ROC指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算ROC指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_atr(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算ATR指标（平均真实波动幅度）
        
        Args:
            df: 包含最高价、最低价、收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了ATR列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                self.logger.error("DataFrame中缺少必要的列(high, low, close)")
                return result_df
            
            # 使用talib计算ATR
            atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=timeperiod)
            
            # 添加ATR列
            result_df[f'ATR_{timeperiod}'] = atr
            # 计算ATR与收盘价的比率
            result_df[f'ATR_RATIO_{timeperiod}'] = atr / df['close']
            
            self.logger.info(f"ATR指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算ATR指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算OBV指标（能量潮指标）
        
        Args:
            df: 包含收盘价、成交量的DataFrame
        
        Returns:
            添加了OBV列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'volume']):
                self.logger.error("DataFrame中缺少必要的列(close, volume)")
                return result_df
            
            # 使用talib计算OBV
            obv = ta.OBV(df['close'].values, df['volume'].values)
            
            # 添加OBV列
            result_df['OBV'] = obv
            # 计算OBV的移动平均（5日）
            result_df['OBV_MA5'] = obv.rolling(window=5).mean()
            
            self.logger.info("OBV指标计算完成")
        except Exception as e:
            self.logger.error(f"计算OBV指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_willr(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算WILLR指标（威廉指标）
        
        Args:
            df: 包含最高价、最低价、收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了WILLR列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                self.logger.error("DataFrame中缺少必要的列(high, low, close)")
                return result_df
            
            # 使用talib计算WILLR
            willr = ta.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=timeperiod)
            
            # 添加WILLR列
            result_df[f'WILLR_{timeperiod}'] = willr
            
            self.logger.info(f"WILLR指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算WILLR指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_adx(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算ADX指标（平均趋向指标）
        
        Args:
            df: 包含最高价、最低价、收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了ADX相关列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                self.logger.error("DataFrame中缺少必要的列(high, low, close)")
                return result_df
            
            # 使用talib计算ADX
            adx = ta.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=timeperiod)
            
            # 添加ADX列
            result_df[f'ADX_{timeperiod}'] = adx
            
            self.logger.info(f"ADX指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算ADX指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_cci(self, df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
        """计算CCI指标（顺势指标）
        
        Args:
            df: 包含最高价、最低价、收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了CCI列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                self.logger.error("DataFrame中缺少必要的列(high, low, close)")
                return result_df
            
            # 使用talib计算CCI
            cci = ta.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=timeperiod)
            
            # 添加CCI列
            result_df[f'CCI_{timeperiod}'] = cci
            
            self.logger.info(f"CCI指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算CCI指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_bias(self, df: pd.DataFrame, timeperiod: int = 6) -> pd.DataFrame:
        """计算BIAS指标（乖离率）
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了BIAS列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 计算移动平均线
            ma = ta.SMA(df['close'].values, timeperiod=timeperiod)
            
            # 计算乖离率
            bias = (df['close'] - ma) / ma * 100
            
            # 添加BIAS列
            result_df[f'BIAS_{timeperiod}'] = bias
            
            self.logger.info(f"BIAS指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算BIAS指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_psy(self, df: pd.DataFrame, timeperiod: int = 12) -> pd.DataFrame:
        """计算PSY指标（心理线指标）
        
        Args:
            df: 包含收盘价的DataFrame
            timeperiod: 计算周期
        
        Returns:
            添加了PSY列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if 'close' not in df.columns:
                self.logger.error("DataFrame中缺少'close'列")
                return result_df
            
            # 计算收盘价的涨跌
            change = df['close'].diff()
            # 标记上涨为1，下跌为0
            up_days = (change > 0).astype(int)
            # 计算PSY值
            psy = up_days.rolling(window=timeperiod).sum() / timeperiod * 100
            
            # 添加PSY列
            result_df[f'PSY_{timeperiod}'] = psy
            
            self.logger.info(f"PSY指标计算完成，参数: timeperiod={timeperiod}")
        except Exception as e:
            self.logger.error(f"计算PSY指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有支持的技术指标
        
        Args:
            df: 包含基本行情数据的DataFrame
        
        Returns:
            添加了所有技术指标列的DataFrame
        """
        result_df = df.copy()
        
        try:
            self.logger.info("开始计算所有技术指标")
            
            # 计算MACD（多种参数组合）
            result_df = self.calculate_macd(result_df, fastperiod=12, slowperiod=26, signalperiod=9)
            result_df = self.calculate_macd(result_df, fastperiod=5, slowperiod=35, signalperiod=5)
            
            # 计算RSI（多种周期）
            for period in [6, 12, 24]:
                result_df = self.calculate_rsi(result_df, timeperiod=period)
            
            # 计算布林带
            result_df = self.calculate_bbands(result_df, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # 计算移动平均线（多种周期）
            for period in [5, 10, 20, 60, 120, 250]:
                result_df = self.calculate_ma(result_df, timeperiod=period)
            
            # 计算指数移动平均线（多种周期）
            for period in [5, 10, 20, 60]:
                result_df = self.calculate_ema(result_df, timeperiod=period)
            
            # 计算KDJ
            result_df = self.calculate_kdj(result_df, n=9, m1=3, m2=3)
            
            # 计算其他指标
            result_df = self.calculate_roc(result_df, timeperiod=12)
            result_df = self.calculate_atr(result_df, timeperiod=14)
            result_df = self.calculate_obv(result_df)
            result_df = self.calculate_willr(result_df, timeperiod=14)
            result_df = self.calculate_adx(result_df, timeperiod=14)
            result_df = self.calculate_cci(result_df, timeperiod=14)
            
            # 计算乖离率（多种周期）
            for period in [6, 12, 24]:
                result_df = self.calculate_bias(result_df, timeperiod=period)
            
            # 计算心理线
            result_df = self.calculate_psy(result_df, timeperiod=12)
            
            self.logger.info("所有技术指标计算完成")
        except Exception as e:
            self.logger.error(f"计算所有技术指标时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_indicators_from_config(self, df: pd.DataFrame, config: List[Dict]) -> pd.DataFrame:
        """根据配置计算指定的技术指标
        
        Args:
            df: 包含基本行情数据的DataFrame
            config: 指标配置列表，每个元素是包含指标名称和参数的字典
        
        Returns:
            添加了指定技术指标列的DataFrame
        """
        result_df = df.copy()
        
        try:
            self.logger.info(f"根据配置计算{len(config)}个技术指标")
            
            for indicator_config in config:
                indicator_name = indicator_config.get('name')
                params = indicator_config.get('params', {})
                
                if indicator_name in self.supported_indicators:
                    try:
                        # 调用对应的计算函数
                        result_df = self.supported_indicators[indicator_name](result_df, **params)
                    except Exception as e:
                        self.logger.error(f"计算{indicator_name}指标时发生异常: {str(e)}")
                else:
                    self.logger.warning(f"不支持的技术指标: {indicator_name}")
            
            self.logger.info("根据配置计算技术指标完成")
        except Exception as e:
            self.logger.error(f"根据配置计算技术指标时发生异常: {str(e)}")
        
        return result_df