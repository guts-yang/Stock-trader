"""因子有效性分析模块，负责评估因子的有效性"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler

class FactorAnalysis:
    """因子分析类，封装了因子有效性评估的核心功能"""
    
    def __init__(self):
        """初始化因子分析类"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("因子分析模块初始化完成")
    
    def calculate_ic(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                    factor_col: str, return_col: str, time_col: str = 'date', 
                    asset_col: str = 'stock_code', method: str = 'pearson') -> pd.DataFrame:
        """计算因子的信息系数(IC)
        
        Args:
            factor_data: 包含因子值的DataFrame
            return_data: 包含收益率的DataFrame
            factor_col: 因子列名
            return_col: 收益率列名
            time_col: 时间列名
            asset_col: 资产列名
            method: 相关系数计算方法，可选'pearson'、'spearman'
        
        Returns:
            包含IC值的DataFrame
        """
        try:
            # 检查必要的列是否存在
            required_cols = [time_col, asset_col, factor_col]
            if not all(col in factor_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in factor_data.columns]
                self.logger.error(f"因子数据中缺少必要的列: {missing_cols}")
                return pd.DataFrame()
            
            required_cols = [time_col, asset_col, return_col]
            if not all(col in return_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in return_data.columns]
                self.logger.error(f"收益率数据中缺少必要的列: {missing_cols}")
                return pd.DataFrame()
            
            self.logger.info(f"开始计算因子{factor_col}的信息系数(IC)")
            
            # 合并因子数据和收益率数据
            merged_data = pd.merge(factor_data, return_data, on=[time_col, asset_col], how='inner')
            
            # 按时间分组计算IC
            ic_values = []
            
            for date, group in merged_data.groupby(time_col):
                # 过滤掉因子值或收益率为NaN的样本
                valid_data = group.dropna(subset=[factor_col, return_col])
                
                if len(valid_data) < 20:  # 至少需要20个有效样本
                    self.logger.warning(f"日期{date}的有效样本数不足20，跳过")
                    continue
                
                if method == 'pearson':
                    ic = stats.pearsonr(valid_data[factor_col], valid_data[return_col])[0]
                elif method == 'spearman':
                    ic = stats.spearmanr(valid_data[factor_col], valid_data[return_col])[0]
                else:
                    self.logger.error(f"不支持的相关系数计算方法: {method}")
                    ic = np.nan
                
                ic_values.append({
                    time_col: date,
                    'IC': ic,
                    'sample_size': len(valid_data)
                })
            
            # 创建IC值DataFrame
            ic_df = pd.DataFrame(ic_values)
            
            # 按时间排序
            if not ic_df.empty:
                ic_df = ic_df.sort_values(time_col)
            
            self.logger.info(f"因子{factor_col}的信息系数(IC)计算完成")
            
            return ic_df
        except Exception as e:
            self.logger.error(f"计算因子信息系数时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def analyze_ic(self, ic_df: pd.DataFrame, ic_col: str = 'IC', time_col: str = 'date') -> Dict:
        """分析IC值的统计特征
        
        Args:
            ic_df: 包含IC值的DataFrame
            ic_col: IC值列名
            time_col: 时间列名
        
        Returns:
            包含IC统计特征的字典
        """
        try:
            if ic_df.empty or ic_col not in ic_df.columns:
                self.logger.error("IC数据为空或缺少IC列")
                return {}
            
            self.logger.info("开始分析IC值的统计特征")
            
            # 计算IC的统计特征
            ic_stats = {
                'mean': ic_df[ic_col].mean(),
                'median': ic_df[ic_col].median(),
                'std': ic_df[ic_col].std(),
                'abs_mean': ic_df[ic_col].abs().mean(),
                'abs_median': ic_df[ic_col].abs().median(),
                'positive_count': (ic_df[ic_col] > 0).sum(),
                'negative_count': (ic_df[ic_col] < 0).sum(),
                'positive_ratio': (ic_df[ic_col] > 0).mean(),
                't_stat': stats.ttest_1samp(ic_df[ic_col], 0)[0],
                'p_value': stats.ttest_1samp(ic_df[ic_col], 0)[1],
                'skew': ic_df[ic_col].skew(),
                'kurtosis': ic_df[ic_col].kurtosis(),
                'total_periods': len(ic_df),
                'start_date': ic_df[time_col].min(),
                'end_date': ic_df[time_col].max()
            }
            
            # 计算IC的自相关
            ic_autocorr_1 = ic_df[ic_col].autocorr(lag=1)
            ic_autocorr_5 = ic_df[ic_col].autocorr(lag=5) if len(ic_df) > 5 else np.nan
            
            ic_stats['autocorr_1'] = ic_autocorr_1
            ic_stats['autocorr_5'] = ic_autocorr_5
            
            self.logger.info("IC值统计特征分析完成")
            
            return ic_stats
        except Exception as e:
            self.logger.error(f"分析IC值统计特征时发生异常: {str(e)}")
            return {}
    
    def factor_ranking(self, factor_data: pd.DataFrame, factor_col: str, time_col: str = 'date', 
                      asset_col: str = 'stock_code', n_groups: int = 5) -> pd.DataFrame:
        """对因子进行分组
        
        Args:
            factor_data: 包含因子值的DataFrame
            factor_col: 因子列名
            time_col: 时间列名
            asset_col: 资产列名
            n_groups: 分组数量
        
        Returns:
            添加了因子分组列的DataFrame
        """
        try:
            # 检查必要的列是否存在
            required_cols = [time_col, asset_col, factor_col]
            if not all(col in factor_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in factor_data.columns]
                self.logger.error(f"因子数据中缺少必要的列: {missing_cols}")
                return pd.DataFrame()
            
            self.logger.info(f"开始对因子{factor_col}进行分组，分组数量: {n_groups}")
            
            result_df = factor_data.copy()
            
            # 按时间分组对因子进行排序并分组
            result_df[f'{factor_col}_rank'] = result_df.groupby(time_col)[factor_col].rank(method='first')
            result_df[f'{factor_col}_group'] = result_df.groupby(time_col)[factor_col].apply(
                lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
            )
            
            # 确保分组从1开始编号
            if f'{factor_col}_group' in result_df.columns:
                result_df[f'{factor_col}_group'] = result_df[f'{factor_col}_group'] + 1
            
            self.logger.info(f"因子{factor_col}分组完成")
            
            return result_df
        except Exception as e:
            self.logger.error(f"对因子进行分组时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def factor_group_backtest(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                            factor_col: str, return_col: str, time_col: str = 'date', 
                            asset_col: str = 'stock_code', n_groups: int = 5) -> pd.DataFrame:
        """因子分层回测
        
        Args:
            factor_data: 包含因子值的DataFrame
            return_data: 包含收益率的DataFrame
            factor_col: 因子列名
            return_col: 收益率列名
            time_col: 时间列名
            asset_col: 资产列名
            n_groups: 分组数量
        
        Returns:
            各因子分组的回测结果DataFrame
        """
        try:
            # 先对因子进行分组
            ranked_data = self.factor_ranking(factor_data, factor_col, time_col, asset_col, n_groups)
            
            if ranked_data.empty:
                self.logger.error("因子分组失败")
                return pd.DataFrame()
            
            # 合并分组数据和收益率数据
            merged_data = pd.merge(ranked_data, return_data, on=[time_col, asset_col], how='inner')
            
            if merged_data.empty:
                self.logger.error("合并因子分组数据和收益率数据失败")
                return pd.DataFrame()
            
            self.logger.info(f"开始进行因子{factor_col}的分层回测，分组数量: {n_groups}")
            
            # 按时间和分组计算平均收益率
            group_returns = merged_data.groupby([time_col, f'{factor_col}_group']).agg({
                return_col: ['mean', 'count'],
                factor_col: 'mean'
            }).reset_index()
            
            # 重命名列
            group_returns.columns = [time_col, 'group', 'mean_return', 'stock_count', 'mean_factor_value']
            
            # 计算多空组合收益率（最高分组 - 最低分组）
            long_short_returns = []
            
            for date, group in group_returns.groupby(time_col):
                max_group = group['group'].max()
                min_group = group['group'].min()
                
                long_return = group.loc[group['group'] == max_group, 'mean_return'].values
                short_return = group.loc[group['group'] == min_group, 'mean_return'].values
                
                if len(long_return) > 0 and len(short_return) > 0:
                    long_short_returns.append({
                        time_col: date,
                        'group': 'long_short',
                        'mean_return': long_return[0] - short_return[0],
                        'stock_count': group['stock_count'].sum(),
                        'mean_factor_value': np.nan
                    })
            
            # 添加多空组合数据
            if long_short_returns:
                long_short_df = pd.DataFrame(long_short_returns)
                group_returns = pd.concat([group_returns, long_short_df], ignore_index=True)
            
            # 按时间排序
            group_returns = group_returns.sort_values([time_col, 'group'])
            
            self.logger.info(f"因子{factor_col}的分层回测完成")
            
            return group_returns
        except Exception as e:
            self.logger.error(f"进行因子分层回测时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def calculate_factor_returns(self, group_returns: pd.DataFrame, time_col: str = 'date', 
                               return_col: str = 'mean_return') -> Dict:
        """计算各因子分组的累计收益率和绩效指标
        
        Args:
            group_returns: 分层回测结果DataFrame
            time_col: 时间列名
            return_col: 收益率列名
        
        Returns:
            包含各分组绩效指标的字典
        """
        try:
            if group_returns.empty or not all(col in group_returns.columns for col in [time_col, return_col]):
                self.logger.error("回测数据为空或缺少必要的列")
                return {}
            
            self.logger.info("开始计算各因子分组的累计收益率和绩效指标")
            
            # 获取所有分组
            groups = group_returns['group'].unique()
            
            # 计算各分组的累计收益率
            group_performance = {}
            
            for group in groups:
                # 筛选当前分组的数据
                group_data = group_returns[group_returns['group'] == group].sort_values(time_col)
                
                if len(group_data) < 1:
                    continue
                
                # 计算累计收益率
                group_data['cumulative_return'] = (1 + group_data[return_col]).cumprod() - 1
                
                # 计算绩效指标
                total_return = group_data['cumulative_return'].iloc[-1]
                n_periods = len(group_data)
                
                # 假设是日频数据，计算年化收益率
                annualized_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
                
                # 计算年化波动率
                daily_vol = group_data[return_col].std()
                annualized_vol = daily_vol * np.sqrt(252)
                
                # 计算夏普比率（假设无风险利率为0）
                sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                
                # 计算最大回撤
                rolling_max = group_data['cumulative_return'].cummax()
                drawdown = (group_data['cumulative_return'] - rolling_max) / (1 + rolling_max)
                max_drawdown = drawdown.min()
                
                # 计算胜率
                win_rate = (group_data[return_col] > 0).mean()
                
                group_performance[group] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'n_periods': n_periods,
                    'start_date': group_data[time_col].min(),
                    'end_date': group_data[time_col].max(),
                    'cumulative_returns': group_data[[time_col, 'cumulative_return']].copy()
                }
            
            self.logger.info("各因子分组的累计收益率和绩效指标计算完成")
            
            return group_performance
        except Exception as e:
            self.logger.error(f"计算因子分组绩效指标时发生异常: {str(e)}")
            return {}
    
    def factor_decay_analysis(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                            factor_col: str, return_col: str, time_col: str = 'date', 
                            asset_col: str = 'stock_code', max_lag: int = 5) -> pd.DataFrame:
        """因子衰减分析
        
        Args:
            factor_data: 包含因子值的DataFrame
            return_data: 包含收益率的DataFrame
            factor_col: 因子列名
            return_col: 收益率列名
            time_col: 时间列名
            asset_col: 资产列名
            max_lag: 最大延迟期数
        
        Returns:
            因子在不同延迟期的IC值DataFrame
        """
        try:
            self.logger.info(f"开始进行因子{factor_col}的衰减分析，最大延迟期数: {max_lag}")
            
            # 确保时间列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(factor_data[time_col]):
                factor_data[time_col] = pd.to_datetime(factor_data[time_col])
            
            if not pd.api.types.is_datetime64_any_dtype(return_data[time_col]):
                return_data[time_col] = pd.to_datetime(return_data[time_col])
            
            # 按资产和时间排序
            factor_data = factor_data.sort_values([asset_col, time_col])
            return_data = return_data.sort_values([asset_col, time_col])
            
            decay_results = []
            
            for lag in range(max_lag + 1):
                self.logger.info(f"计算延迟{lag}期的IC值")
                
                # 对收益率数据进行滞后处理
                lagged_returns = return_data.copy()
                
                # 按资产分组，然后对收益率进行滞后
                lagged_returns = lagged_returns.groupby(asset_col).apply(
                    lambda x: x.assign(**{f'{return_col}_lag{lag}': x[return_col].shift(-lag)})
                ).reset_index(drop=True)
                
                # 计算当前延迟期的IC值
                ic_df = self.calculate_ic(
                    factor_data, 
                    lagged_returns.rename(columns={f'{return_col}_lag{lag}': return_col}),
                    factor_col, 
                    return_col,
                    time_col,
                    asset_col
                )
                
                if not ic_df.empty:
                    # 分析IC值
                    ic_stats = self.analyze_ic(ic_df)
                    
                    decay_results.append({
                        'lag': lag,
                        'mean_ic': ic_stats.get('mean', np.nan),
                        'std_ic': ic_stats.get('std', np.nan),
                        'abs_mean_ic': ic_stats.get('abs_mean', np.nan),
                        't_stat': ic_stats.get('t_stat', np.nan),
                        'p_value': ic_stats.get('p_value', np.nan),
                        'positive_ratio': ic_stats.get('positive_ratio', np.nan),
                        'n_periods': ic_stats.get('total_periods', 0)
                    })
            
            # 创建衰减分析结果DataFrame
            decay_df = pd.DataFrame(decay_results)
            
            self.logger.info(f"因子{factor_col}的衰减分析完成")
            
            return decay_df
        except Exception as e:
            self.logger.error(f"进行因子衰减分析时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def factor_correlation_analysis(self, factor_data: pd.DataFrame, factor_cols: List[str], 
                                  time_col: str = 'date', asset_col: str = 'stock_code', 
                                  method: str = 'pearson') -> pd.DataFrame:
        """因子相关性分析
        
        Args:
            factor_data: 包含多个因子值的DataFrame
            factor_cols: 要分析的因子列名列表
            time_col: 时间列名
            asset_col: 资产列名
            method: 相关系数计算方法
        
        Returns:
            因子相关性矩阵DataFrame
        """
        try:
            # 检查必要的列是否存在
            required_cols = [time_col, asset_col] + factor_cols
            if not all(col in factor_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in factor_data.columns]
                self.logger.error(f"因子数据中缺少必要的列: {missing_cols}")
                return pd.DataFrame()
            
            self.logger.info(f"开始进行因子相关性分析，因子数量: {len(factor_cols)}")
            
            # 计算各期的因子相关性，然后取平均
            correlations = []
            
            for date, group in factor_data.groupby(time_col):
                # 过滤掉所有因子值都为NaN的样本
                valid_data = group.dropna(subset=factor_cols)
                
                if len(valid_data) < 20:  # 至少需要20个有效样本
                    self.logger.warning(f"日期{date}的有效样本数不足20，跳过")
                    continue
                
                # 计算因子相关性矩阵
                corr_matrix = valid_data[factor_cols].corr(method=method)
                
                # 将相关矩阵转换为长格式
                corr_long = corr_matrix.stack().reset_index()
                corr_long.columns = ['factor1', 'factor2', 'correlation']
                corr_long[time_col] = date
                
                correlations.append(corr_long)
            
            if not correlations:
                self.logger.error("没有足够的有效数据进行因子相关性分析")
                return pd.DataFrame()
            
            # 合并所有期的相关性数据
            all_correlations = pd.concat(correlations, ignore_index=True)
            
            # 计算平均相关性
            avg_correlations = all_correlations.groupby(['factor1', 'factor2'])['correlation'].mean().reset_index()
            
            # 重塑为矩阵形式
            corr_matrix_df = avg_correlations.pivot(index='factor1', columns='factor2', values='correlation')
            
            self.logger.info("因子相关性分析完成")
            
            return corr_matrix_df
        except Exception as e:
            self.logger.error(f"进行因子相关性分析时发生异常: {str(e)}")
            return pd.DataFrame()
    
    def factor_turnover_analysis(self, factor_data: pd.DataFrame, factor_col: str, 
                               time_col: str = 'date', asset_col: str = 'stock_code') -> Dict:
        """因子换手率分析
        
        Args:
            factor_data: 包含因子值的DataFrame
            factor_col: 因子列名
            time_col: 时间列名
            asset_col: 资产列名
        
        Returns:
            包含因子换手率统计的字典
        """
        try:
            # 检查必要的列是否存在
            required_cols = [time_col, asset_col, factor_col]
            if not all(col in factor_data.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in factor_data.columns]
                self.logger.error(f"因子数据中缺少必要的列: {missing_cols}")
                return {}
            
            self.logger.info(f"开始进行因子{factor_col}的换手率分析")
            
            # 确保时间列是datetime类型并排序
            if not pd.api.types.is_datetime64_any_dtype(factor_data[time_col]):
                factor_data[time_col] = pd.to_datetime(factor_data[time_col])
            
            factor_data = factor_data.sort_values([asset_col, time_col])
            
            # 计算因子排名
            factor_data[f'{factor_col}_rank'] = factor_data.groupby(time_col)[factor_col].rank()
            
            # 计算因子排名变化
            rank_changes = []
            
            for asset, group in factor_data.groupby(asset_col):
                if len(group) < 2:
                    continue
                
                # 计算相邻期的排名变化
                group = group.sort_values(time_col)
                group['prev_rank'] = group[f'{factor_col}_rank'].shift(1)
                group['rank_change'] = group[f'{factor_col}_rank'] - group['prev_rank']
                
                # 只保留有前一期数据的记录
                valid_group = group.dropna(subset=['prev_rank'])
                
                if not valid_group.empty:
                    for _, row in valid_group.iterrows():
                        rank_changes.append({
                            time_col: row[time_col],
                            asset_col: asset,
                            'rank_change': row['rank_change'],
                            'abs_rank_change': abs(row['rank_change'])
                        })
            
            if not rank_changes:
                self.logger.error("没有足够的有效数据进行因子换手率分析")
                return {}
            
            # 创建排名变化DataFrame
            rank_change_df = pd.DataFrame(rank_changes)
            
            # 计算每期的平均绝对排名变化和换手率
            turnover_results = []
            
            for date, group in rank_change_df.groupby(time_col):
                # 计算平均绝对排名变化
                avg_abs_rank_change = group['abs_rank_change'].mean()
                
                # 计算换手率（基于排名变化的标准化值）
                # 首先获取该期的股票数量
                n_stocks = factor_data[factor_data[time_col] == date].shape[0]
                
                if n_stocks > 0:
                    # 换手率 = 平均绝对排名变化 / (n_stocks - 1) * 2
                    turnover = avg_abs_rank_change / (n_stocks - 1) * 2 if (n_stocks - 1) > 0 else 0
                else:
                    turnover = 0
                
                turnover_results.append({
                    time_col: date,
                    'avg_abs_rank_change': avg_abs_rank_change,
                    'turnover': turnover,
                    'n_stocks': n_stocks
                })
            
            # 创建换手率结果DataFrame
            turnover_df = pd.DataFrame(turnover_results)
            
            # 计算换手率的统计特征
            turnover_stats = {
                'mean_turnover': turnover_df['turnover'].mean(),
                'median_turnover': turnover_df['turnover'].median(),
                'std_turnover': turnover_df['turnover'].std(),
                'max_turnover': turnover_df['turnover'].max(),
                'min_turnover': turnover_df['turnover'].min(),
                'n_periods': len(turnover_df),
                'turnover_data': turnover_df
            }
            
            self.logger.info(f"因子{factor_col}的换手率分析完成")
            
            return turnover_stats
        except Exception as e:
            self.logger.error(f"进行因子换手率分析时发生异常: {str(e)}")
            return {}
    
    def full_factor_analysis(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                           factor_col: str, return_col: str, time_col: str = 'date', 
                           asset_col: str = 'stock_code', n_groups: int = 5, 
                           max_lag: int = 5) -> Dict:
        """完整的因子分析流程
        
        Args:
            factor_data: 包含因子值的DataFrame
            return_data: 包含收益率的DataFrame
            factor_col: 因子列名
            return_col: 收益率列名
            time_col: 时间列名
            asset_col: 资产列名
            n_groups: 分层回测的分组数量
            max_lag: 因子衰减分析的最大延迟期数
        
        Returns:
            包含完整因子分析结果的字典
        """
        try:
            self.logger.info(f"开始进行因子{factor_col}的完整分析")
            
            # 1. 计算IC值
            ic_df = self.calculate_ic(factor_data, return_data, factor_col, return_col, time_col, asset_col)
            
            # 2. 分析IC值统计特征
            ic_stats = self.analyze_ic(ic_df)
            
            # 3. 因子分层回测
            group_returns = self.factor_group_backtest(
                factor_data, return_data, factor_col, return_col, time_col, asset_col, n_groups
            )
            
            # 4. 计算分组绩效指标
            group_performance = self.calculate_factor_returns(group_returns, time_col)
            
            # 5. 因子衰减分析
            decay_df = self.factor_decay_analysis(
                factor_data, return_data, factor_col, return_col, time_col, asset_col, max_lag
            )
            
            # 6. 因子换手率分析
            turnover_stats = self.factor_turnover_analysis(factor_data, factor_col, time_col, asset_col)
            
            # 整合所有分析结果
            full_analysis_result = {
                'factor_name': factor_col,
                'ic_analysis': {
                    'ic_data': ic_df,
                    'ic_stats': ic_stats
                },
                'backtest_analysis': {
                    'group_returns': group_returns,
                    'group_performance': group_performance
                },
                'decay_analysis': decay_df,
                'turnover_analysis': turnover_stats,
                'analysis_time': pd.Timestamp.now()
            }
            
            self.logger.info(f"因子{factor_col}的完整分析完成")
            
            return full_analysis_result
        except Exception as e:
            self.logger.error(f"进行因子完整分析时发生异常: {str(e)}")
            return {}
    
    def factor_combinatorial_analysis(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                                    factor_cols: List[str], return_col: str, time_col: str = 'date', 
                                    asset_col: str = 'stock_code', n_groups: int = 5) -> Dict:
        """因子组合分析
        
        Args:
            factor_data: 包含多个因子值的DataFrame
            return_data: 包含收益率的DataFrame
            factor_cols: 要分析的因子列名列表
            return_col: 收益率列名
            time_col: 时间列名
            asset_col: 资产列名
            n_groups: 分组数量
        
        Returns:
            包含因子组合分析结果的字典
        """
        try:
            self.logger.info(f"开始进行因子组合分析，因子数量: {len(factor_cols)}")
            
            # 1. 计算因子相关性
            corr_matrix = self.factor_correlation_analysis(factor_data, factor_cols, time_col, asset_col)
            
            # 2. 对每个因子进行标准化处理
            scaler = StandardScaler()
            normalized_data = factor_data.copy()
            
            for factor_col in factor_cols:
                # 按时间分组进行标准化
                normalized_data[f'{factor_col}_normalized'] = normalized_data.groupby(time_col)[factor_col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
            
            # 3. 计算等权重因子组合
            normalized_factor_cols = [f'{col}_normalized' for col in factor_cols]
            normalized_data['combined_factor_equal_weight'] = normalized_data[normalized_factor_cols].mean(axis=1)
            
            # 4. 对因子组合进行回测
            combined_backtest = self.factor_group_backtest(
                normalized_data, return_data, 'combined_factor_equal_weight', 
                return_col, time_col, asset_col, n_groups
            )
            
            # 5. 计算组合因子的绩效指标
            combined_performance = self.calculate_factor_returns(combined_backtest, time_col)
            
            # 整合分析结果
            combinatorial_result = {
                'factor_correlation_matrix': corr_matrix,
                'normalized_factor_data': normalized_data,
                'combined_factor_backtest': combined_backtest,
                'combined_factor_performance': combined_performance,
                'analysis_time': pd.Timestamp.now()
            }
            
            self.logger.info("因子组合分析完成")
            
            return combinatorial_result
        except Exception as e:
            self.logger.error(f"进行因子组合分析时发生异常: {str(e)}")
            return {}
    
    def generate_factor_report(self, analysis_result: Dict, output_path: Optional[str] = None) -> str:
        """生成因子分析报告
        
        Args:
            analysis_result: 因子分析结果字典
            output_path: 报告输出路径，如果为None则返回报告内容
        
        Returns:
            因子分析报告内容
        """
        try:
            if not analysis_result:
                self.logger.error("分析结果为空，无法生成报告")
                return ""
            
            self.logger.info(f"开始生成因子'{analysis_result.get('factor_name', 'Unknown')}'的分析报告")
            
            # 构建报告内容
            report = []
            report.append(f"# 因子分析报告: {analysis_result.get('factor_name', 'Unknown')}")
            report.append(f"生成时间: {analysis_result.get('analysis_time', pd.Timestamp.now())}")
            report.append("\n## 1. 信息系数(IC)分析")
            
            # IC统计信息
            ic_stats = analysis_result.get('ic_analysis', {}).get('ic_stats', {})
            if ic_stats:
                report.append("\n### IC统计特征:")
                report.append(f"- 平均IC: {ic_stats.get('mean', 0):.4f}")
                report.append(f"- 中位数IC: {ic_stats.get('median', 0):.4f}")
                report.append(f"- IC标准差: {ic_stats.get('std', 0):.4f}")
                report.append(f"- 平均绝对IC: {ic_stats.get('abs_mean', 0):.4f}")
                report.append(f"- 正IC比率: {ic_stats.get('positive_ratio', 0):.2%}")
                report.append(f"- t统计量: {ic_stats.get('t_stat', 0):.2f}")
                report.append(f"- p值: {ic_stats.get('p_value', 0):.4f}")
                report.append(f"- IC自相关性(1期): {ic_stats.get('autocorr_1', 0):.4f}")
            
            # 回测结果
            report.append("\n## 2. 分层回测结果")
            
            group_performance = analysis_result.get('backtest_analysis', {}).get('group_performance', {})
            if group_performance:
                report.append("\n### 各分组绩效指标:")
                
                # 按分组排序
                sorted_groups = sorted(group_performance.keys())
                
                for group in sorted_groups:
                    perf = group_performance[group]
                    report.append(f"\n#### 分组 {group}:")
                    report.append(f"- 总收益率: {perf.get('total_return', 0):.2%}")
                    report.append(f"- 年化收益率: {perf.get('annualized_return', 0):.2%}")
                    report.append(f"- 年化波动率: {perf.get('annualized_volatility', 0):.2%}")
                    report.append(f"- 夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
                    report.append(f"- 最大回撤: {perf.get('max_drawdown', 0):.2%}")
                    report.append(f"- 胜率: {perf.get('win_rate', 0):.2%}")
            
            # 因子衰减分析
            decay_df = analysis_result.get('decay_analysis', pd.DataFrame())
            if not decay_df.empty:
                report.append("\n## 3. 因子衰减分析")
                report.append("\n### 不同延迟期的平均IC:")
                
                for _, row in decay_df.iterrows():
                    report.append(f"- 延迟{int(row['lag'])}期: {row['mean_ic']:.4f}")
            
            # 因子换手率分析
            turnover_stats = analysis_result.get('turnover_analysis', {})
            if turnover_stats:
                report.append("\n## 4. 因子换手率分析")
                report.append(f"- 平均换手率: {turnover_stats.get('mean_turnover', 0):.2%}")
                report.append(f"- 换手率标准差: {turnover_stats.get('std_turnover', 0):.2%}")
            
            # 组合成完整报告
            full_report = '\n'.join(report)
            
            # 如果指定了输出路径，将报告保存到文件
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_report)
                    self.logger.info(f"因子分析报告已保存到: {output_path}")
                except Exception as e:
                    self.logger.error(f"保存因子分析报告时发生异常: {str(e)}")
            
            self.logger.info("因子分析报告生成完成")
            
            return full_report
        except Exception as e:
            self.logger.error(f"生成因子分析报告时发生异常: {str(e)}")
            return ""