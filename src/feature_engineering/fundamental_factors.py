"""基本面因子处理模块，负责处理和计算各种基本面因子"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union

class FundamentalFactors:
    """基本面因子处理类，封装了常用的基本面因子计算方法"""
    
    def __init__(self):
        """初始化基本面因子处理类"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("基本面因子处理模块初始化完成")
        
        # 记录已支持的基本面因子
        self.supported_factors = {
            'PE': self.calculate_pe,
            'PB': self.calculate_pb,
            'ROE': self.calculate_roe,
            'ROA': self.calculate_roa,
            'EPS': self.calculate_eps,
            'BPS': self.calculate_bps,
            'DPS': self.calculate_dps,
            'DIVIDEND_RATIO': self.calculate_dividend_ratio,
            'GROWTH_RATE': self.calculate_growth_rate,
            'DEBT_EQUITY_RATIO': self.calculate_debt_equity_ratio,
            'CURRENT_RATIO': self.calculate_current_ratio,
            'QUICK_RATIO': self.calculate_quick_ratio,
            'GROSS_MARGIN': self.calculate_gross_margin,
            'NET_PROFIT_MARGIN': self.calculate_net_profit_margin,
            'OPERATING_CASH_FLOW_PER_SHARE': self.calculate_operating_cash_flow_per_share,
            'FREE_CASH_FLOW_PER_SHARE': self.calculate_free_cash_flow_per_share,
            'MARKET_CAP': self.calculate_market_cap,
            'ENTERPRISE_VALUE': self.calculate_enterprise_value,
            'EV_EBITDA': self.calculate_ev_ebitda,
            'PE_TTM': self.calculate_pe_ttm,
            'PB_LF': self.calculate_pb_lf,
            'PS_RATIO': self.calculate_ps_ratio,
            'PCF_RATIO': self.calculate_pcf_ratio,
            'ASSET_TURNOVER': self.calculate_asset_turnover,
            'INVENTORY_TURNOVER': self.calculate_inventory_turnover,
            'RECEIVABLES_TURNOVER': self.calculate_receivables_turnover,
            'CASH_RATIO': self.calculate_cash_ratio,
            'INTEREST_COVERAGE_RATIO': self.calculate_interest_coverage_ratio,
            'WORKING_CAPITAL_TURNOVER': self.calculate_working_capital_turnover,
            'NET_DEBT_EBITDA': self.calculate_net_debt_ebitda
        }
    
    def calculate_pe(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市盈率(PE)
        
        Args:
            df: 包含股价和每股收益的DataFrame
        
        Returns:
            添加了PE列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'eps']):
                self.logger.error("DataFrame中缺少必要的列(close, eps)")
                return result_df
            
            # 计算市盈率: 股价/每股收益
            result_df['PE'] = df['close'] / df['eps'].where(df['eps'] != 0)
            
            # 处理异常值：PE为负或无穷大的值设为NaN
            result_df['PE'] = result_df['PE'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PE'] < 0, 'PE'] = np.nan
            
            self.logger.info("市盈率(PE)计算完成")
        except Exception as e:
            self.logger.error(f"计算市盈率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_pb(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市净率(PB)
        
        Args:
            df: 包含股价和每股净资产的DataFrame
        
        Returns:
            添加了PB列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'bps']):
                self.logger.error("DataFrame中缺少必要的列(close, bps)")
                return result_df
            
            # 计算市净率: 股价/每股净资产
            result_df['PB'] = df['close'] / df['bps'].where(df['bps'] != 0)
            
            # 处理异常值：PB为负或无穷大的值设为NaN
            result_df['PB'] = result_df['PB'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PB'] < 0, 'PB'] = np.nan
            
            self.logger.info("市净率(PB)计算完成")
        except Exception as e:
            self.logger.error(f"计算市净率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_roe(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算净资产收益率(ROE)
        
        Args:
            df: 包含净利润和净资产的DataFrame
        
        Returns:
            添加了ROE列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['net_profit', 'total_equity']):
                self.logger.error("DataFrame中缺少必要的列(net_profit, total_equity)")
                return result_df
            
            # 计算ROE: 净利润/净资产
            result_df['ROE'] = df['net_profit'] / df['total_equity'].where(df['total_equity'] != 0) * 100
            
            # 处理异常值
            result_df['ROE'] = result_df['ROE'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("净资产收益率(ROE)计算完成")
        except Exception as e:
            self.logger.error(f"计算净资产收益率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_roa(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算总资产收益率(ROA)
        
        Args:
            df: 包含净利润和总资产的DataFrame
        
        Returns:
            添加了ROA列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['net_profit', 'total_assets']):
                self.logger.error("DataFrame中缺少必要的列(net_profit, total_assets)")
                return result_df
            
            # 计算ROA: 净利润/总资产
            result_df['ROA'] = df['net_profit'] / df['total_assets'].where(df['total_assets'] != 0) * 100
            
            # 处理异常值
            result_df['ROA'] = result_df['ROA'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("总资产收益率(ROA)计算完成")
        except Exception as e:
            self.logger.error(f"计算总资产收益率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_eps(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每股收益(EPS)
        
        Args:
            df: 包含净利润和总股本的DataFrame
        
        Returns:
            添加了EPS列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['net_profit', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(net_profit, total_shares)")
                return result_df
            
            # 计算EPS: 净利润/总股本
            result_df['EPS'] = df['net_profit'] / df['total_shares'].where(df['total_shares'] != 0)
            
            # 处理异常值
            result_df['EPS'] = result_df['EPS'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("每股收益(EPS)计算完成")
        except Exception as e:
            self.logger.error(f"计算每股收益时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_bps(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每股净资产(BPS)
        
        Args:
            df: 包含净资产和总股本的DataFrame
        
        Returns:
            添加了BPS列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['total_equity', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(total_equity, total_shares)")
                return result_df
            
            # 计算BPS: 净资产/总股本
            result_df['BPS'] = df['total_equity'] / df['total_shares'].where(df['total_shares'] != 0)
            
            # 处理异常值
            result_df['BPS'] = result_df['BPS'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("每股净资产(BPS)计算完成")
        except Exception as e:
            self.logger.error(f"计算每股净资产时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_dps(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每股分红(DPS)
        
        Args:
            df: 包含总分红和总股本的DataFrame
        
        Returns:
            添加了DPS列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['total_dividend', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(total_dividend, total_shares)")
                return result_df
            
            # 计算DPS: 总分红/总股本
            result_df['DPS'] = df['total_dividend'] / df['total_shares'].where(df['total_shares'] != 0)
            
            # 处理异常值
            result_df['DPS'] = result_df['DPS'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("每股分红(DPS)计算完成")
        except Exception as e:
            self.logger.error(f"计算每股分红时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_dividend_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算股息率
        
        Args:
            df: 包含每股分红和股价的DataFrame
        
        Returns:
            添加了股息率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['dps', 'close']):
                self.logger.error("DataFrame中缺少必要的列(dps, close)")
                return result_df
            
            # 计算股息率: 每股分红/股价
            result_df['DIVIDEND_RATIO'] = df['dps'] / df['close'].where(df['close'] != 0) * 100
            
            # 处理异常值
            result_df['DIVIDEND_RATIO'] = result_df['DIVIDEND_RATIO'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['DIVIDEND_RATIO'] < 0, 'DIVIDEND_RATIO'] = np.nan
            
            self.logger.info("股息率计算完成")
        except Exception as e:
            self.logger.error(f"计算股息率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_growth_rate(self, df: pd.DataFrame, period: int = 4) -> pd.DataFrame:
        """计算同比增长率
        
        Args:
            df: 包含财务数据的DataFrame
            period: 比较的期间（如季度数）
        
        Returns:
            添加了各指标增长率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            self.logger.info(f"计算同比增长率，比较期间: {period}")
            
            # 需要计算增长率的指标列表
            metrics = ['revenue', 'operating_profit', 'net_profit', 'eps', 'total_assets']
            
            for metric in metrics:
                if metric in df.columns:
                    # 计算同比增长率
                    result_df[f'{metric}_GROWTH_RATE'] = (
                        (df[metric] - df[metric].shift(period)) / df[metric].shift(period).abs().where(df[metric].shift(period) != 0)
                    ) * 100
                    # 处理异常值
                    result_df[f'{metric}_GROWTH_RATE'] = result_df[f'{metric}_GROWTH_RATE'].replace([np.inf, -np.inf], np.nan)
                    
                    self.logger.info(f"{metric}同比增长率计算完成")
                else:
                    self.logger.warning(f"DataFrame中缺少列: {metric}")
        except Exception as e:
            self.logger.error(f"计算同比增长率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_debt_equity_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资产负债率
        
        Args:
            df: 包含负债和资产的DataFrame
        
        Returns:
            添加了资产负债率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['total_liabilities', 'total_assets']):
                self.logger.error("DataFrame中缺少必要的列(total_liabilities, total_assets)")
                return result_df
            
            # 计算资产负债率: 总负债/总资产
            result_df['DEBT_EQUITY_RATIO'] = df['total_liabilities'] / df['total_assets'].where(df['total_assets'] != 0) * 100
            
            # 处理异常值
            result_df['DEBT_EQUITY_RATIO'] = result_df['DEBT_EQUITY_RATIO'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("资产负债率计算完成")
        except Exception as e:
            self.logger.error(f"计算资产负债率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_current_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算流动比率
        
        Args:
            df: 包含流动资产和流动负债的DataFrame
        
        Returns:
            添加了流动比率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['current_assets', 'current_liabilities']):
                self.logger.error("DataFrame中缺少必要的列(current_assets, current_liabilities)")
                return result_df
            
            # 计算流动比率: 流动资产/流动负债
            result_df['CURRENT_RATIO'] = df['current_assets'] / df['current_liabilities'].where(df['current_liabilities'] != 0)
            
            # 处理异常值
            result_df['CURRENT_RATIO'] = result_df['CURRENT_RATIO'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("流动比率计算完成")
        except Exception as e:
            self.logger.error(f"计算流动比率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_quick_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算速动比率
        
        Args:
            df: 包含流动资产、存货和流动负债的DataFrame
        
        Returns:
            添加了速动比率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
                self.logger.error("DataFrame中缺少必要的列(current_assets, inventory, current_liabilities)")
                return result_df
            
            # 计算速动比率: (流动资产-存货)/流动负债
            result_df['QUICK_RATIO'] = (df['current_assets'] - df['inventory']) / df['current_liabilities'].where(df['current_liabilities'] != 0)
            
            # 处理异常值
            result_df['QUICK_RATIO'] = result_df['QUICK_RATIO'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("速动比率计算完成")
        except Exception as e:
            self.logger.error(f"计算速动比率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_gross_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算毛利率
        
        Args:
            df: 包含营业收入和营业成本的DataFrame
        
        Returns:
            添加了毛利率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['revenue', 'cost_of_goods_sold']):
                self.logger.error("DataFrame中缺少必要的列(revenue, cost_of_goods_sold)")
                return result_df
            
            # 计算毛利率: (营业收入-营业成本)/营业收入
            result_df['GROSS_MARGIN'] = (
                (df['revenue'] - df['cost_of_goods_sold']) / df['revenue'].where(df['revenue'] != 0)
            ) * 100
            
            # 处理异常值
            result_df['GROSS_MARGIN'] = result_df['GROSS_MARGIN'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['GROSS_MARGIN'] < -100, 'GROSS_MARGIN'] = np.nan
            result_df.loc[result_df['GROSS_MARGIN'] > 100, 'GROSS_MARGIN'] = np.nan
            
            self.logger.info("毛利率计算完成")
        except Exception as e:
            self.logger.error(f"计算毛利率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_net_profit_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算净利率
        
        Args:
            df: 包含营业收入和净利润的DataFrame
        
        Returns:
            添加了净利率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['revenue', 'net_profit']):
                self.logger.error("DataFrame中缺少必要的列(revenue, net_profit)")
                return result_df
            
            # 计算净利率: 净利润/营业收入
            result_df['NET_PROFIT_MARGIN'] = df['net_profit'] / df['revenue'].where(df['revenue'] != 0) * 100
            
            # 处理异常值
            result_df['NET_PROFIT_MARGIN'] = result_df['NET_PROFIT_MARGIN'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("净利率计算完成")
        except Exception as e:
            self.logger.error(f"计算净利率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_operating_cash_flow_per_share(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每股经营现金流
        
        Args:
            df: 包含经营现金流和总股本的DataFrame
        
        Returns:
            添加了每股经营现金流列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['operating_cash_flow', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(operating_cash_flow, total_shares)")
                return result_df
            
            # 计算每股经营现金流: 经营现金流/总股本
            result_df['OPERATING_CASH_FLOW_PER_SHARE'] = df['operating_cash_flow'] / df['total_shares'].where(df['total_shares'] != 0)
            
            # 处理异常值
            result_df['OPERATING_CASH_FLOW_PER_SHARE'] = result_df['OPERATING_CASH_FLOW_PER_SHARE'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("每股经营现金流计算完成")
        except Exception as e:
            self.logger.error(f"计算每股经营现金流时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_free_cash_flow_per_share(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每股自由现金流
        
        Args:
            df: 包含经营现金流、资本支出和总股本的DataFrame
        
        Returns:
            添加了每股自由现金流列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['operating_cash_flow', 'capital_expenditure', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(operating_cash_flow, capital_expenditure, total_shares)")
                return result_df
            
            # 计算自由现金流
            free_cash_flow = df['operating_cash_flow'] - df['capital_expenditure']
            # 计算每股自由现金流: 自由现金流/总股本
            result_df['FREE_CASH_FLOW_PER_SHARE'] = free_cash_flow / df['total_shares'].where(df['total_shares'] != 0)
            
            # 处理异常值
            result_df['FREE_CASH_FLOW_PER_SHARE'] = result_df['FREE_CASH_FLOW_PER_SHARE'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("每股自由现金流计算完成")
        except Exception as e:
            self.logger.error(f"计算每股自由现金流时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市值
        
        Args:
            df: 包含股价和总股本的DataFrame
        
        Returns:
            添加了市值列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'total_shares']):
                self.logger.error("DataFrame中缺少必要的列(close, total_shares)")
                return result_df
            
            # 计算市值: 股价*总股本
            result_df['MARKET_CAP'] = df['close'] * df['total_shares']
            
            self.logger.info("市值计算完成")
        except Exception as e:
            self.logger.error(f"计算市值时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_enterprise_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算企业价值(EV)
        
        Args:
            df: 包含市值、总负债和现金及现金等价物的DataFrame
        
        Returns:
            添加了企业价值列的DataFrame
        """
        result_df = df.copy()
        
        try:
            required_columns = ['MARKET_CAP', 'total_liabilities', 'cash_and_cash_equivalents']
            
            # 如果缺少市值列，则先计算市值
            if 'MARKET_CAP' not in df.columns and all(col in df.columns for col in ['close', 'total_shares']):
                result_df = self.calculate_market_cap(result_df)
            
            if not all(col in result_df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in result_df.columns]
                self.logger.error(f"DataFrame中缺少必要的列: {missing_cols}")
                return result_df
            
            # 计算企业价值: 市值+总负债-现金及现金等价物
            result_df['ENTERPRISE_VALUE'] = result_df['MARKET_CAP'] + result_df['total_liabilities'] - result_df['cash_and_cash_equivalents']
            
            self.logger.info("企业价值(EV)计算完成")
        except Exception as e:
            self.logger.error(f"计算企业价值时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_ev_ebitda(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算EV/EBITDA比率
        
        Args:
            df: 包含必要数据的DataFrame
        
        Returns:
            添加了EV/EBITDA列的DataFrame
        """
        result_df = df.copy()
        
        try:
            # 如果缺少企业价值列，则先计算企业价值
            if 'ENTERPRISE_VALUE' not in df.columns:
                result_df = self.calculate_enterprise_value(result_df)
            
            if not all(col in result_df.columns for col in ['ENTERPRISE_VALUE', 'ebitda']):
                self.logger.error("DataFrame中缺少必要的列(ENTERPRISE_VALUE, ebitda)")
                return result_df
            
            # 计算EV/EBITDA: 企业价值/EBITDA
            result_df['EV_EBITDA'] = result_df['ENTERPRISE_VALUE'] / result_df['ebitda'].where(result_df['ebitda'] != 0)
            
            # 处理异常值
            result_df['EV_EBITDA'] = result_df['EV_EBITDA'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['EV_EBITDA'] < 0, 'EV_EBITDA'] = np.nan
            
            self.logger.info("EV/EBITDA比率计算完成")
        except Exception as e:
            self.logger.error(f"计算EV/EBITDA比率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_pe_ttm(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算滚动市盈率(PE-TTM)
        
        Args:
            df: 包含股价和最近12个月每股收益的DataFrame
        
        Returns:
            添加了PE-TTM列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'eps_ttm']):
                self.logger.error("DataFrame中缺少必要的列(close, eps_ttm)")
                return result_df
            
            # 计算PE-TTM: 股价/最近12个月每股收益
            result_df['PE_TTM'] = df['close'] / df['eps_ttm'].where(df['eps_ttm'] != 0)
            
            # 处理异常值
            result_df['PE_TTM'] = result_df['PE_TTM'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PE_TTM'] < 0, 'PE_TTM'] = np.nan
            
            self.logger.info("滚动市盈率(PE-TTM)计算完成")
        except Exception as e:
            self.logger.error(f"计算滚动市盈率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_pb_lf(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市净率(最新财报)
        
        Args:
            df: 包含股价和最新财报每股净资产的DataFrame
        
        Returns:
            添加了PB-LF列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['close', 'bps_latest']):
                self.logger.error("DataFrame中缺少必要的列(close, bps_latest)")
                return result_df
            
            # 计算PB-LF: 股价/最新财报每股净资产
            result_df['PB_LF'] = df['close'] / df['bps_latest'].where(df['bps_latest'] != 0)
            
            # 处理异常值
            result_df['PB_LF'] = result_df['PB_LF'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PB_LF'] < 0, 'PB_LF'] = np.nan
            
            self.logger.info("市净率(最新财报)计算完成")
        except Exception as e:
            self.logger.error(f"计算市净率(最新财报)时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_ps_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市销率(PS)
        
        Args:
            df: 包含市值和营业收入的DataFrame
        
        Returns:
            添加了PS列的DataFrame
        """
        result_df = df.copy()
        
        try:
            # 如果缺少市值列，则先计算市值
            if 'MARKET_CAP' not in df.columns and all(col in df.columns for col in ['close', 'total_shares']):
                result_df = self.calculate_market_cap(result_df)
            
            if not all(col in result_df.columns for col in ['MARKET_CAP', 'revenue']):
                self.logger.error("DataFrame中缺少必要的列(MARKET_CAP, revenue)")
                return result_df
            
            # 计算市销率: 市值/营业收入
            result_df['PS_RATIO'] = result_df['MARKET_CAP'] / result_df['revenue'].where(result_df['revenue'] != 0)
            
            # 处理异常值
            result_df['PS_RATIO'] = result_df['PS_RATIO'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PS_RATIO'] < 0, 'PS_RATIO'] = np.nan
            
            self.logger.info("市销率(PS)计算完成")
        except Exception as e:
            self.logger.error(f"计算市销率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_pcf_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算市现率(PCF)
        
        Args:
            df: 包含市值和经营现金流的DataFrame
        
        Returns:
            添加了PCF列的DataFrame
        """
        result_df = df.copy()
        
        try:
            # 如果缺少市值列，则先计算市值
            if 'MARKET_CAP' not in df.columns and all(col in df.columns for col in ['close', 'total_shares']):
                result_df = self.calculate_market_cap(result_df)
            
            if not all(col in result_df.columns for col in ['MARKET_CAP', 'operating_cash_flow']):
                self.logger.error("DataFrame中缺少必要的列(MARKET_CAP, operating_cash_flow)")
                return result_df
            
            # 计算市现率: 市值/经营现金流
            result_df['PCF_RATIO'] = result_df['MARKET_CAP'] / result_df['operating_cash_flow'].where(result_df['operating_cash_flow'] != 0)
            
            # 处理异常值
            result_df['PCF_RATIO'] = result_df['PCF_RATIO'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['PCF_RATIO'] < 0, 'PCF_RATIO'] = np.nan
            
            self.logger.info("市现率(PCF)计算完成")
        except Exception as e:
            self.logger.error(f"计算市现率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_asset_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资产周转率
        
        Args:
            df: 包含营业收入和总资产的DataFrame
        
        Returns:
            添加了资产周转率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['revenue', 'total_assets']):
                self.logger.error("DataFrame中缺少必要的列(revenue, total_assets)")
                return result_df
            
            # 计算资产周转率: 营业收入/总资产
            result_df['ASSET_TURNOVER'] = df['revenue'] / df['total_assets'].where(df['total_assets'] != 0)
            
            # 处理异常值
            result_df['ASSET_TURNOVER'] = result_df['ASSET_TURNOVER'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("资产周转率计算完成")
        except Exception as e:
            self.logger.error(f"计算资产周转率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_inventory_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算存货周转率
        
        Args:
            df: 包含营业成本和存货的DataFrame
        
        Returns:
            添加了存货周转率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['cost_of_goods_sold', 'inventory']):
                self.logger.error("DataFrame中缺少必要的列(cost_of_goods_sold, inventory)")
                return result_df
            
            # 计算存货周转率: 营业成本/存货
            result_df['INVENTORY_TURNOVER'] = df['cost_of_goods_sold'] / df['inventory'].where(df['inventory'] != 0)
            
            # 处理异常值
            result_df['INVENTORY_TURNOVER'] = result_df['INVENTORY_TURNOVER'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("存货周转率计算完成")
        except Exception as e:
            self.logger.error(f"计算存货周转率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_receivables_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算应收账款周转率
        
        Args:
            df: 包含营业收入和应收账款的DataFrame
        
        Returns:
            添加了应收账款周转率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['revenue', 'accounts_receivable']):
                self.logger.error("DataFrame中缺少必要的列(revenue, accounts_receivable)")
                return result_df
            
            # 计算应收账款周转率: 营业收入/应收账款
            result_df['RECEIVABLES_TURNOVER'] = df['revenue'] / df['accounts_receivable'].where(df['accounts_receivable'] != 0)
            
            # 处理异常值
            result_df['RECEIVABLES_TURNOVER'] = result_df['RECEIVABLES_TURNOVER'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("应收账款周转率计算完成")
        except Exception as e:
            self.logger.error(f"计算应收账款周转率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_cash_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算现金比率
        
        Args:
            df: 包含现金及现金等价物和流动负债的DataFrame
        
        Returns:
            添加了现金比率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['cash_and_cash_equivalents', 'current_liabilities']):
                self.logger.error("DataFrame中缺少必要的列(cash_and_cash_equivalents, current_liabilities)")
                return result_df
            
            # 计算现金比率: 现金及现金等价物/流动负债
            result_df['CASH_RATIO'] = df['cash_and_cash_equivalents'] / df['current_liabilities'].where(df['current_liabilities'] != 0)
            
            # 处理异常值
            result_df['CASH_RATIO'] = result_df['CASH_RATIO'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("现金比率计算完成")
        except Exception as e:
            self.logger.error(f"计算现金比率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_interest_coverage_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算利息保障倍数
        
        Args:
            df: 包含息税前利润和利息费用的DataFrame
        
        Returns:
            添加了利息保障倍数列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['ebit', 'interest_expense']):
                self.logger.error("DataFrame中缺少必要的列(ebit, interest_expense)")
                return result_df
            
            # 计算利息保障倍数: 息税前利润/利息费用
            result_df['INTEREST_COVERAGE_RATIO'] = df['ebit'] / df['interest_expense'].where(df['interest_expense'] != 0)
            
            # 处理异常值
            result_df['INTEREST_COVERAGE_RATIO'] = result_df['INTEREST_COVERAGE_RATIO'].replace([np.inf, -np.inf], np.nan)
            result_df.loc[result_df['INTEREST_COVERAGE_RATIO'] < 0, 'INTEREST_COVERAGE_RATIO'] = np.nan
            
            self.logger.info("利息保障倍数计算完成")
        except Exception as e:
            self.logger.error(f"计算利息保障倍数时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_working_capital_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算营运资本周转率
        
        Args:
            df: 包含营业收入、流动资产和流动负债的DataFrame
        
        Returns:
            添加了营运资本周转率列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['revenue', 'current_assets', 'current_liabilities']):
                self.logger.error("DataFrame中缺少必要的列(revenue, current_assets, current_liabilities)")
                return result_df
            
            # 计算营运资本
            working_capital = df['current_assets'] - df['current_liabilities']
            # 计算营运资本周转率: 营业收入/营运资本
            result_df['WORKING_CAPITAL_TURNOVER'] = df['revenue'] / working_capital.where(working_capital != 0)
            
            # 处理异常值
            result_df['WORKING_CAPITAL_TURNOVER'] = result_df['WORKING_CAPITAL_TURNOVER'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("营运资本周转率计算完成")
        except Exception as e:
            self.logger.error(f"计算营运资本周转率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_net_debt_ebitda(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算净债务/EBITDA比率
        
        Args:
            df: 包含总负债、现金及现金等价物和EBITDA的DataFrame
        
        Returns:
            添加了净债务/EBITDA列的DataFrame
        """
        result_df = df.copy()
        
        try:
            if not all(col in df.columns for col in ['total_liabilities', 'cash_and_cash_equivalents', 'ebitda']):
                self.logger.error("DataFrame中缺少必要的列(total_liabilities, cash_and_cash_equivalents, ebitda)")
                return result_df
            
            # 计算净债务
            net_debt = df['total_liabilities'] - df['cash_and_cash_equivalents']
            # 计算净债务/EBITDA: 净债务/EBITDA
            result_df['NET_DEBT_EBITDA'] = net_debt / df['ebitda'].where(df['ebitda'] != 0)
            
            # 处理异常值
            result_df['NET_DEBT_EBITDA'] = result_df['NET_DEBT_EBITDA'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.info("净债务/EBITDA比率计算完成")
        except Exception as e:
            self.logger.error(f"计算净债务/EBITDA比率时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有支持的基本面因子
        
        Args:
            df: 包含基本财务数据的DataFrame
        
        Returns:
            添加了所有基本面因子列的DataFrame
        """
        result_df = df.copy()
        
        try:
            self.logger.info("开始计算所有基本面因子")
            
            # 计算基本估值指标
            result_df = self.calculate_pe(result_df)
            result_df = self.calculate_pb(result_df)
            result_df = self.calculate_roe(result_df)
            result_df = self.calculate_roa(result_df)
            result_df = self.calculate_eps(result_df)
            result_df = self.calculate_bps(result_df)
            
            # 计算增长指标
            result_df = self.calculate_growth_rate(result_df, period=4)  # 同比（4个季度）
            result_df = self.calculate_growth_rate(result_df, period=1)  # 环比（1个季度）
            
            # 计算偿债能力指标
            result_df = self.calculate_debt_equity_ratio(result_df)
            result_df = self.calculate_current_ratio(result_df)
            result_df = self.calculate_quick_ratio(result_df)
            
            # 计算盈利能力指标
            result_df = self.calculate_gross_margin(result_df)
            result_df = self.calculate_net_profit_margin(result_df)
            
            # 计算现金流量指标
            result_df = self.calculate_operating_cash_flow_per_share(result_df)
            result_df = self.calculate_free_cash_flow_per_share(result_df)
            
            # 计算估值比率
            result_df = self.calculate_market_cap(result_df)
            result_df = self.calculate_enterprise_value(result_df)
            result_df = self.calculate_ev_ebitda(result_df)
            result_df = self.calculate_ps_ratio(result_df)
            result_df = self.calculate_pcf_ratio(result_df)
            
            # 计算运营效率指标
            result_df = self.calculate_asset_turnover(result_df)
            result_df = self.calculate_inventory_turnover(result_df)
            result_df = self.calculate_receivables_turnover(result_df)
            
            self.logger.info("所有基本面因子计算完成")
        except Exception as e:
            self.logger.error(f"计算所有基本面因子时发生异常: {str(e)}")
        
        return result_df
    
    def calculate_factors_from_config(self, df: pd.DataFrame, config: List[Dict]) -> pd.DataFrame:
        """根据配置计算指定的基本面因子
        
        Args:
            df: 包含基本财务数据的DataFrame
            config: 因子配置列表，每个元素是包含因子名称和参数的字典
        
        Returns:
            添加了指定基本面因子列的DataFrame
        """
        result_df = df.copy()
        
        try:
            self.logger.info(f"根据配置计算{len(config)}个基本面因子")
            
            for factor_config in config:
                factor_name = factor_config.get('name')
                params = factor_config.get('params', {})
                
                if factor_name in self.supported_factors:
                    try:
                        # 调用对应的计算函数
                        result_df = self.supported_factors[factor_name](result_df, **params)
                    except Exception as e:
                        self.logger.error(f"计算{factor_name}因子时发生异常: {str(e)}")
                else:
                    self.logger.warning(f"不支持的基本面因子: {factor_name}")
            
            self.logger.info("根据配置计算基本面因子完成")
        except Exception as e:
            self.logger.error(f"根据配置计算基本面因子时发生异常: {str(e)}")
        
        return result_df