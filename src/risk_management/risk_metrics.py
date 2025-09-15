import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from scipy import stats
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RiskMetrics:
    """风险指标计算模块
    提供各种风险度量指标的计算功能，用于评估投资组合的风险特征
    """
    # 波动率计算方法
    VOLATILITY_METHOD_STANDARD = 'standard'
    VOLATILITY_METHOD_EWMA = 'ewma'
    VOLATILITY_METHOD_GARCH = 'garch'  # 简化版，实际可能需要专门的GARCH实现
    
    # VaR计算方法
    VAR_METHOD_HISTORICAL = 'historical'
    VAR_METHOD_PARAMETRIC = 'parametric'
    VAR_METHOD_MONTE_CARLO = 'monte_carlo'
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 lookback_period: int = 252,
                 risk_free_rate: float = 0.03,
                 confidence_level: float = 0.95,
                 ewma_lambda: float = 0.94,
                 default_vol_method: str = VOLATILITY_METHOD_STANDARD,
                 default_var_method: str = VAR_METHOD_HISTORICAL):
        """初始化风险指标计算器
        
        Args:
            config: 配置字典
            lookback_period: 回看期长度（交易日）
            risk_free_rate: 无风险利率
            confidence_level: 置信水平（用于VaR等指标）
            ewma_lambda: EWMA波动率计算的衰减因子
            default_vol_method: 默认的波动率计算方法
            default_var_method: 默认的VaR计算方法
        """
        self.config = config or {}
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.ewma_lambda = ewma_lambda
        self.default_vol_method = default_vol_method
        self.default_var_method = default_var_method
        
        # 风险指标历史记录
        self.metrics_history = {}
        
        # 初始化日志
        self._init_logger()
        
        logger.info("RiskMetrics 初始化完成")
    
    def _init_logger(self):
        """初始化日志记录器"""
        import os
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def calculate_returns(self, 
                         prices: pd.DataFrame,
                         frequency: str = 'daily') -> pd.DataFrame:
        """计算资产收益率
        
        Args:
            prices: 价格数据，索引为日期，列为资产
            frequency: 收益率频率 ('daily', 'weekly', 'monthly')
        
        Returns:
            收益率数据
        """
        # 计算日收益率
        returns = prices.pct_change().dropna()
        
        # 如果需要其他频率的收益率
        if frequency != 'daily':
            if frequency == 'weekly':
                returns = returns.resample('W-FRI').apply(lambda x: (1 + x).prod() - 1)
            elif frequency == 'monthly':
                returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            else:
                logger.warning(f"不支持的收益率频率: {frequency}，使用日收益率")
        
        return returns
    
    def calculate_volatility(self, 
                            returns: Union[pd.Series, pd.DataFrame],
                            method: Optional[str] = None,
                            annualize: bool = True) -> Union[float, pd.Series, pd.DataFrame]:
        """计算波动率
        
        Args:
            returns: 收益率数据
            method: 波动率计算方法
            annualize: 是否年化
        
        Returns:
            波动率值
        """
        # 使用默认方法（如果未指定）
        if method is None:
            method = self.default_vol_method
        
        # 根据方法计算波动率
        if method == self.VOLATILITY_METHOD_STANDARD:
            # 标准波动率计算
            if isinstance(returns, pd.DataFrame):
                volatility = returns.std()
            else:
                volatility = returns.std()
        elif method == self.VOLATILITY_METHOD_EWMA:
            # 使用EWMA计算波动率
            if isinstance(returns, pd.DataFrame):
                # 对每个资产计算EWMA波动率
                volatility = pd.Series(index=returns.columns)
                for col in returns.columns:
                    volatility[col] = self._calculate_ewma_vol(returns[col])
            else:
                volatility = self._calculate_ewma_vol(returns)
        elif method == self.VOLATILITY_METHOD_GARCH:
            # 简化版GARCH波动率（使用EWMA近似）
            logger.warning("GARCH方法使用EWMA近似实现")
            if isinstance(returns, pd.DataFrame):
                volatility = pd.Series(index=returns.columns)
                for col in returns.columns:
                    volatility[col] = self._calculate_ewma_vol(returns[col])
            else:
                volatility = self._calculate_ewma_vol(returns)
        else:
            logger.warning(f"不支持的波动率计算方法: {method}，使用标准方法")
            if isinstance(returns, pd.DataFrame):
                volatility = returns.std()
            else:
                volatility = returns.std()
        
        # 年化处理
        if annualize:
            if method == self.VOLATILITY_METHOD_STANDARD:
                # 标准波动率年化使用交易日的平方根
                volatility = volatility * np.sqrt(252)
            else:
                # EWMA和GARCH波动率已经是年化的（假设每日计算）
                pass
        
        return volatility
    
    def _calculate_ewma_vol(self, returns: pd.Series) -> float:
        """使用EWMA方法计算波动率
        
        Args:
            returns: 收益率序列
        
        Returns:
            EWMA波动率值
        """
        # 计算平方收益率
        squared_returns = returns ** 2
        
        # 计算EWMA波动率
        n = len(squared_returns)
        weights = np.array([(1 - self.ewma_lambda) * self.ewma_lambda**i for i in range(n-1, -1, -1)])
        weights /= weights.sum()  # 归一化权重
        
        # 计算加权平均平方收益率
        ewma_variance = np.sum(weights * squared_returns)
        
        # 开平方得到波动率
        ewma_volatility = np.sqrt(ewma_variance)
        
        # 年化
        return ewma_volatility * np.sqrt(252)
    
    def calculate_var(self, 
                     returns: Union[pd.Series, pd.DataFrame],
                     method: Optional[str] = None,
                     confidence_level: Optional[float] = None,
                     lookback_period: Optional[int] = None) -> Union[float, pd.Series]:
        """计算Value at Risk (VaR)
        
        Args:
            returns: 收益率数据
            method: VaR计算方法
            confidence_level: 置信水平
            lookback_period: 回看期长度
        
        Returns:
            VaR值
        """
        # 使用默认参数（如果未指定）
        if method is None:
            method = self.default_var_method
        if confidence_level is None:
            confidence_level = self.confidence_level
        if lookback_period is None:
            lookback_period = min(self.lookback_period, len(returns))
        
        # 截取最近的回看期数据
        if len(returns) > lookback_period:
            returns = returns.iloc[-lookback_period:]
        
        # 根据方法计算VaR
        if method == self.VAR_METHOD_HISTORICAL:
            # 历史模拟法
            if isinstance(returns, pd.DataFrame):
                var = returns.quantile(1 - confidence_level)
            else:
                var = returns.quantile(1 - confidence_level)
        elif method == self.VAR_METHOD_PARAMETRIC:
            # 参数法（假设正态分布）
            if isinstance(returns, pd.DataFrame):
                mean = returns.mean()
                std = returns.std()
                var = mean + std * stats.norm.ppf(1 - confidence_level)
            else:
                mean = returns.mean()
                std = returns.std()
                var = mean + std * stats.norm.ppf(1 - confidence_level)
        elif method == self.VAR_METHOD_MONTE_CARLO:
            # 蒙特卡洛模拟法
            if isinstance(returns, pd.DataFrame):
                var = pd.Series(index=returns.columns)
                for col in returns.columns:
                    var[col] = self._calculate_monte_carlo_var(returns[col], confidence_level)
            else:
                var = self._calculate_monte_carlo_var(returns, confidence_level)
        else:
            logger.warning(f"不支持的VaR计算方法: {method}，使用历史模拟法")
            if isinstance(returns, pd.DataFrame):
                var = returns.quantile(1 - confidence_level)
            else:
                var = returns.quantile(1 - confidence_level)
        
        return var
    
    def _calculate_monte_carlo_var(self, 
                                  returns: pd.Series,
                                  confidence_level: float,
                                  n_simulations: int = 10000) -> float:
        """使用蒙特卡洛模拟计算VaR
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            n_simulations: 模拟次数
        
        Returns:
            Monte Carlo VaR值
        """
        # 估计收益率的分布参数
        mean = returns.mean()
        std = returns.std()
        
        # 生成模拟的收益率
        np.random.seed(42)  # 设置随机种子以获得可重复的结果
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # 计算VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return var
    
    def calculate_cvar(self, 
                      returns: Union[pd.Series, pd.DataFrame],
                      method: Optional[str] = None,
                      confidence_level: Optional[float] = None,
                      lookback_period: Optional[int] = None) -> Union[float, pd.Series]:
        """计算Conditional Value at Risk (CVaR)
        
        Args:
            returns: 收益率数据
            method: CVaR计算方法
            confidence_level: 置信水平
            lookback_period: 回看期长度
        
        Returns:
            CVaR值
        """
        # 使用默认参数（如果未指定）
        if method is None:
            method = self.default_var_method
        if confidence_level is None:
            confidence_level = self.confidence_level
        if lookback_period is None:
            lookback_period = min(self.lookback_period, len(returns))
        
        # 截取最近的回看期数据
        if len(returns) > lookback_period:
            returns = returns.iloc[-lookback_period:]
        
        # 先计算VaR
        var = self.calculate_var(returns, method, confidence_level, lookback_period)
        
        # 计算CVaR
        if isinstance(returns, pd.DataFrame):
            cvar = pd.Series(index=returns.columns)
            for col in returns.columns:
                # 获取低于VaR的收益率
                tail_returns = returns[col][returns[col] <= var[col]]
                # 计算尾部平均收益率
                cvar[col] = tail_returns.mean() if not tail_returns.empty else var[col]
        else:
            # 获取低于VaR的收益率
            tail_returns = returns[returns <= var]
            # 计算尾部平均收益率
            cvar = tail_returns.mean() if not tail_returns.empty else var
        
        return cvar
    
    def calculate_sharpe_ratio(self, 
                             returns: Union[pd.Series, pd.DataFrame],
                             risk_free_rate: Optional[float] = None,
                             annualize: bool = True) -> Union[float, pd.Series]:
        """计算夏普比率
        
        Args:
            returns: 收益率数据
            risk_free_rate: 无风险利率
            annualize: 是否年化
        
        Returns:
            夏普比率值
        """
        # 使用默认无风险利率（如果未指定）
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # 计算超额收益率
        if isinstance(returns, pd.DataFrame):
            excess_returns = returns - risk_free_rate / 252  # 假设日度无风险利率
            mean_excess_return = excess_returns.mean()
            volatility = returns.std()
        else:
            excess_returns = returns - risk_free_rate / 252  # 假设日度无风险利率
            mean_excess_return = excess_returns.mean()
            volatility = returns.std()
        
        # 年化处理
        if annualize:
            mean_excess_return = mean_excess_return * 252
            volatility = volatility * np.sqrt(252)
        
        # 计算夏普比率
        if isinstance(returns, pd.DataFrame):
            sharpe_ratio = mean_excess_return / volatility
        else:
            sharpe_ratio = mean_excess_return / volatility if volatility != 0 else 0
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, 
                              returns: Union[pd.Series, pd.DataFrame],
                              risk_free_rate: Optional[float] = None,
                              annualize: bool = True,
                              target_return: float = 0.0) -> Union[float, pd.Series]:
        """计算Sortino比率
        
        Args:
            returns: 收益率数据
            risk_free_rate: 无风险利率
            annualize: 是否年化
            target_return: 目标收益率（用于计算下行风险）
        
        Returns:
            Sortino比率值
        """
        # 使用默认无风险利率（如果未指定）
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # 计算超额收益率
        if isinstance(returns, pd.DataFrame):
            excess_returns = returns - risk_free_rate / 252  # 假设日度无风险利率
            mean_excess_return = excess_returns.mean()
            
            # 计算下行波动率
            downside_volatility = pd.Series(index=returns.columns)
            for col in returns.columns:
                downside_returns = returns[col][returns[col] < target_return]
                if not downside_returns.empty:
                    downside_volatility[col] = downside_returns.std()
                else:
                    downside_volatility[col] = 0.0001  # 避免除以零
        else:
            excess_returns = returns - risk_free_rate / 252  # 假设日度无风险利率
            mean_excess_return = excess_returns.mean()
            
            # 计算下行波动率
            downside_returns = returns[returns < target_return]
            if not downside_returns.empty:
                downside_volatility = downside_returns.std()
            else:
                downside_volatility = 0.0001  # 避免除以零
        
        # 年化处理
        if annualize:
            mean_excess_return = mean_excess_return * 252
            downside_volatility = downside_volatility * np.sqrt(252)
        
        # 计算Sortino比率
        if isinstance(returns, pd.DataFrame):
            sortino_ratio = mean_excess_return / downside_volatility
        else:
            sortino_ratio = mean_excess_return / downside_volatility
        
        return sortino_ratio
    
    def calculate_beta(self, 
                      portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
        """计算投资组合的Beta系数
        
        Args:
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准收益率
        
        Returns:
            Beta系数值
        """
        # 确保两个收益率序列有相同的索引
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            logger.warning("没有足够的共同数据点来计算Beta系数")
            return 0.0
        
        # 对齐数据
        aligned_portfolio = portfolio_returns.loc[common_index]
        aligned_benchmark = benchmark_returns.loc[common_index]
        
        # 计算协方差和基准方差
        covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        # 计算Beta
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        return beta
    
    def calculate_alpha(self, 
                       portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       risk_free_rate: Optional[float] = None,
                       beta: Optional[float] = None) -> float:
        """计算投资组合的Alpha系数
        
        Args:
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准收益率
            risk_free_rate: 无风险利率
            beta: Beta系数（如果未提供则计算）
        
        Returns:
            Alpha系数值
        """
        # 使用默认无风险利率（如果未指定）
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # 如果未提供Beta，计算它
        if beta is None:
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        
        # 计算平均收益率
        avg_portfolio_return = portfolio_returns.mean()
        avg_benchmark_return = benchmark_returns.mean()
        avg_risk_free_return = risk_free_rate / 252  # 假设日度无风险利率
        
        # 计算Alpha
        alpha = avg_portfolio_return - (avg_risk_free_return + beta * (avg_benchmark_return - avg_risk_free_return))
        
        # 年化Alpha
        alpha_annualized = alpha * 252
        
        return alpha_annualized
    
    def calculate_correlation(self, 
                             returns: pd.DataFrame,
                             method: str = 'pearson') -> pd.DataFrame:
        """计算相关系数矩阵
        
        Args:
            returns: 收益率数据
            method: 相关系数计算方法 ('pearson', 'kendall', 'spearman')
        
        Returns:
            相关系数矩阵
        """
        # 计算相关系数矩阵
        correlation_matrix = returns.corr(method=method)
        
        return correlation_matrix
    
    def calculate_covariance(self, 
                            returns: pd.DataFrame,
                            shrinkage: bool = False) -> pd.DataFrame:
        """计算协方差矩阵
        
        Args:
            returns: 收益率数据
            shrinkage: 是否使用收缩估计
        
        Returns:
            协方差矩阵
        """
        if shrinkage:
            # 使用Ledoit-Wolf收缩估计
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                covariance_matrix = pd.DataFrame(
                    data=lw.fit(returns).covariance_,
                    index=returns.columns,
                    columns=returns.columns
                )
            except ImportError:
                logger.warning("无法导入LedoitWolf，使用标准协方差")
                covariance_matrix = returns.cov()
            except Exception as e:
                logger.error(f"使用Ledoit-Wolf收缩估计时出错: {str(e)}，回退到标准协方差")
                covariance_matrix = returns.cov()
        else:
            # 标准协方差计算
            covariance_matrix = returns.cov()
        
        return covariance_matrix
    
    def risk_contribution(self, 
                         weights: Dict[str, float],
                         covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算单个资产对投资组合风险的贡献
        
        Args:
            weights: 资产权重
            covariance_matrix: 协方差矩阵
        
        Returns:
            每个资产的风险贡献
        """
        # 确保权重和协方差矩阵匹配
        assets = list(weights.keys())
        if not all(asset in covariance_matrix.columns for asset in assets):
            missing_assets = [asset for asset in assets if asset not in covariance_matrix.columns]
            logger.error(f"协方差矩阵中缺少以下资产: {missing_assets}")
            return {}
        
        # 过滤协方差矩阵
        filtered_covariance = covariance_matrix.loc[assets, assets]
        
        # 将权重转换为数组
        weights_array = np.array([weights[asset] for asset in assets])
        
        # 计算投资组合方差
        portfolio_variance = np.dot(weights_array.T, np.dot(filtered_covariance, weights_array))
        
        # 计算边际风险贡献
        marginal_contributions = np.dot(filtered_covariance, weights_array) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else np.zeros_like(weights_array)
        
        # 计算风险贡献
        risk_contributions = weights_array * marginal_contributions
        
        # 转换为字典
        risk_contributions_dict = {asset: contribution for asset, contribution in zip(assets, risk_contributions)}
        
        return risk_contributions_dict
    
    def risk_parity_check(self, 
                         risk_contributions: Dict[str, float],
                         tolerance: float = 0.1) -> Dict[str, Any]:
        """检查投资组合是否实现了风险平价
        
        Args:
            risk_contributions: 各资产的风险贡献
            tolerance: 允许的偏差容忍度
        
        Returns:
            风险平价检查结果
        """
        # 计算总风险
        total_risk = sum(risk_contributions.values())
        
        # 计算每个资产的风险贡献百分比
        risk_contribution_pct = {asset: (contribution / total_risk) * 100 for asset, contribution in risk_contributions.items()}
        
        # 计算目标风险贡献（等风险）
        n_assets = len(risk_contributions)
        target_contribution_pct = 100.0 / n_assets
        
        # 计算每个资产的偏差
        deviations = {asset: abs(pct - target_contribution_pct) for asset, pct in risk_contribution_pct.items()}
        
        # 确定是否实现了风险平价
        is_risk_parity = all(deviation <= tolerance * target_contribution_pct for deviation in deviations.values())
        
        # 构建检查结果
        check_result = {
            'is_risk_parity': is_risk_parity,
            'risk_contribution_pct': risk_contribution_pct,
            'target_contribution_pct': target_contribution_pct,
            'deviations': deviations,
            'max_deviation': max(deviations.values()) if deviations else 0,
            'tolerance': tolerance * target_contribution_pct,
            'total_risk': total_risk
        }
        
        return check_result
    
    def factor_risk_attribution(self, 
                              returns: pd.DataFrame,
                              factor_returns: pd.DataFrame,
                              weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """进行因子风险归因分析
        
        Args:
            returns: 资产收益率数据
            factor_returns: 因子收益率数据
            weights: 资产权重（如果未提供，使用等权重）
        
        Returns:
            因子风险归因结果
        """
        # 确保收益率和因子数据有相同的索引
        common_index = returns.index.intersection(factor_returns.index)
        if len(common_index) < 2:
            logger.warning("没有足够的共同数据点来进行因子风险归因")
            return {}
        
        # 对齐数据
        aligned_returns = returns.loc[common_index]
        aligned_factors = factor_returns.loc[common_index]
        
        # 使用等权重（如果未提供权重）
        if weights is None:
            n_assets = len(aligned_returns.columns)
            weights = {asset: 1.0 / n_assets for asset in aligned_returns.columns}
        
        # 计算投资组合收益率
        portfolio_returns = aligned_returns.dot(np.array([weights.get(asset, 0) for asset in aligned_returns.columns]))
        
        # 计算因子暴露度（简化版，实际可能需要更复杂的因子模型）
        # 这里假设因子暴露度是资产收益率与因子收益率之间的回归系数
        factor_exposures = {}
        for factor in aligned_factors.columns:
            # 对每个资产计算与因子的相关性（作为暴露度的近似）
            exposure = {}
            for asset in aligned_returns.columns:
                correlation = aligned_returns[asset].corr(aligned_factors[factor])
                exposure[asset] = correlation
            factor_exposures[factor] = exposure
        
        # 计算因子贡献
        factor_contributions = {}
        for factor in aligned_factors.columns:
            # 计算加权平均暴露度
            avg_exposure = sum(weights.get(asset, 0) * factor_exposures[factor].get(asset, 0) for asset in aligned_returns.columns)
            
            # 计算因子方差
            factor_variance = aligned_factors[factor].var()
            
            # 计算因子贡献
            factor_contribution = avg_exposure ** 2 * factor_variance
            factor_contributions[factor] = factor_contribution
        
        # 计算总风险和因子解释的风险
        total_risk = portfolio_returns.var()
        explained_risk = sum(factor_contributions.values())
        
        # 计算因子贡献百分比
        factor_contribution_pct = {factor: (contribution / explained_risk) * 100 if explained_risk > 0 else 0 
                                 for factor, contribution in factor_contributions.items()}
        
        # 计算未解释风险
        unexplained_risk = total_risk - explained_risk
        
        # 构建归因结果
        attribution_result = {
            'factor_contributions': factor_contributions,
            'factor_contribution_pct': factor_contribution_pct,
            'total_risk': total_risk,
            'explained_risk': explained_risk,
            'unexplained_risk': unexplained_risk,
            'explained_ratio': (explained_risk / total_risk) * 100 if total_risk > 0 else 0,
            'factor_exposures': factor_exposures
        }
        
        return attribution_result
    
    def calculate_concentration_risk(self, 
                                   weights: Dict[str, float],
                                   method: str = 'herfindahl') -> float:
        """计算集中度风险
        
        Args:
            weights: 资产权重
            method: 计算方法 ('herfindahl', 'gini', 'entropy')
        
        Returns:
            集中度风险值
        """
        # 将权重转换为数组
        weight_values = np.array(list(weights.values()))
        
        # 确保权重和为1
        total_weight = sum(weight_values)
        if total_weight > 0:
            weight_values = weight_values / total_weight
        
        # 根据方法计算集中度风险
        if method == 'herfindahl':
            # 赫芬达尔指数（Herfindahl-Hirschman Index）
            concentration = np.sum(weight_values ** 2)
        elif method == 'gini':
            # 基尼系数
            n = len(weight_values)
            sorted_weights = np.sort(weight_values)
            gini = (2 * np.sum(np.arange(1, n+1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        elif method == 'entropy':
            # 熵指数
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -np.sum(weight_values * np.log(weight_values))
            concentration = np.exp(entropy) / len(weight_values) if not np.isnan(entropy) else 0
        else:
            logger.warning(f"不支持的集中度风险计算方法: {method}，使用赫芬达尔指数")
            concentration = np.sum(weight_values ** 2)
        
        return concentration
    
    def calculate_liquidity_risk(self, 
                               returns: pd.DataFrame,
                               volumes: Optional[pd.DataFrame] = None,
                               method: str = 'amihud') -> Union[float, pd.Series]:
        """计算流动性风险
        
        Args:
            returns: 收益率数据
            volumes: 成交量数据（如果可用）
            method: 计算方法 ('amihud', 'turnover')
        
        Returns:
            流动性风险值
        """
        # 如果没有成交量数据，返回默认值
        if volumes is None:
            logger.warning("没有成交量数据，无法计算流动性风险")
            if isinstance(returns, pd.DataFrame):
                return pd.Series([0.0 for _ in returns.columns], index=returns.columns)
            else:
                return 0.0
        
        # 确保收益率和成交量数据有相同的索引
        common_index = returns.index.intersection(volumes.index)
        if len(common_index) < 2:
            logger.warning("没有足够的共同数据点来计算流动性风险")
            if isinstance(returns, pd.DataFrame):
                return pd.Series([0.0 for _ in returns.columns], index=returns.columns)
            else:
                return 0.0
        
        # 对齐数据
        aligned_returns = returns.loc[common_index]
        aligned_volumes = volumes.loc[common_index]
        
        # 根据方法计算流动性风险
        if method == 'amihud':
            # Amihud流动性比率
            if isinstance(returns, pd.DataFrame):
                liquidity_risk = pd.Series(index=returns.columns)
                for col in returns.columns:
                    # 计算每日Amihud比率
                    daily_amihud = abs(aligned_returns[col]) / aligned_volumes[col]
                    # 取平均值
                    liquidity_risk[col] = daily_amihud.mean()
            else:
                # 计算每日Amihud比率
                daily_amihud = abs(aligned_returns) / aligned_volumes.iloc[:, 0]  # 假设volumes只有一列
                # 取平均值
                liquidity_risk = daily_amihud.mean()
        elif method == 'turnover':
            # 周转率（简化版）
            if isinstance(returns, pd.DataFrame):
                liquidity_risk = pd.Series(index=returns.columns)
                for col in returns.columns:
                    # 计算平均成交量（作为流动性的度量）
                    avg_volume = aligned_volumes[col].mean()
                    # 假设低成交量表示高流动性风险
                    liquidity_risk[col] = 1.0 / avg_volume if avg_volume > 0 else float('inf')
            else:
                # 计算平均成交量
                avg_volume = aligned_volumes.iloc[:, 0].mean()  # 假设volumes只有一列
                # 假设低成交量表示高流动性风险
                liquidity_risk = 1.0 / avg_volume if avg_volume > 0 else float('inf')
        else:
            logger.warning(f"不支持的流动性风险计算方法: {method}，使用Amihud比率")
            if isinstance(returns, pd.DataFrame):
                liquidity_risk = pd.Series(index=returns.columns)
                for col in returns.columns:
                    daily_amihud = abs(aligned_returns[col]) / aligned_volumes[col]
                    liquidity_risk[col] = daily_amihud.mean()
            else:
                daily_amihud = abs(aligned_returns) / aligned_volumes.iloc[:, 0]
                liquidity_risk = daily_amihud.mean()
        
        return liquidity_risk
    
    def calculate_kurtosis_skewness(self, 
                                  returns: Union[pd.Series, pd.DataFrame]) -> Union[Tuple[float, float], Tuple[pd.Series, pd.Series]]:
        """计算收益率的峰度和偏度
        
        Args:
            returns: 收益率数据
        
        Returns:
            (峰度, 偏度) 元组
        """
        if isinstance(returns, pd.DataFrame):
            kurtosis = returns.kurtosis()
            skewness = returns.skew()
        else:
            kurtosis = returns.kurtosis()
            skewness = returns.skew()
        
        return kurtosis, skewness
    
    def calculate_drawdown_metrics(self, 
                                 equity_curve: pd.Series) -> Dict[str, float]:
        """计算回撤相关指标
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            回撤指标字典
        """
        # 计算累积最大权益
        cumulative_max = equity_curve.cummax()
        
        # 计算回撤
        drawdown = (equity_curve / cumulative_max) - 1
        
        # 计算最大回撤
        max_drawdown = drawdown.min()
        
        # 计算回撤持续时间
        drawdown_days = 0
        max_drawdown_days = 0
        in_drawdown = False
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                # 处于回撤状态
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_days = 1
                else:
                    drawdown_days += 1
                max_drawdown_days = max(max_drawdown_days, drawdown_days)
            else:
                # 不在回撤状态
                in_drawdown = False
                drawdown_days = 0
        
        # 计算处于回撤中的时间百分比
        time_under_water_pct = (sum(drawdown < 0) / len(drawdown)) * 100
        
        # 构建指标字典
        drawdown_metrics = {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_days': max_drawdown_days,
            'time_under_water_pct': time_under_water_pct,
            'avg_drawdown_pct': drawdown[drawdown < 0].mean() * 100 if sum(drawdown < 0) > 0 else 0
        }
        
        return drawdown_metrics
    
    def calculate_all_metrics(self, 
                             portfolio_returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None,
                             asset_returns: Optional[pd.DataFrame] = None,
                             weights: Optional[Dict[str, float]] = None,
                             volumes: Optional[pd.DataFrame] = None,
                             factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """计算所有风险指标
        
        Args:
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准收益率
            asset_returns: 各资产收益率
            weights: 资产权重
            volumes: 成交量数据
            factor_returns: 因子收益率
        
        Returns:
            所有风险指标的字典
        """
        # 计算基本风险指标
        metrics = {
            'timestamp': datetime.now(),
            'volatility': {
                'standard': self.calculate_volatility(portfolio_returns, method=self.VOLATILITY_METHOD_STANDARD),
                'ewma': self.calculate_volatility(portfolio_returns, method=self.VOLATILITY_METHOD_EWMA)
            },
            'var': {
                f'{self.confidence_level:.0%}': {
                    'historical': self.calculate_var(portfolio_returns, method=self.VAR_METHOD_HISTORICAL),
                    'parametric': self.calculate_var(portfolio_returns, method=self.VAR_METHOD_PARAMETRIC),
                    'monte_carlo': self.calculate_var(portfolio_returns, method=self.VAR_METHOD_MONTE_CARLO)
                }
            },
            'cvar': {
                f'{self.confidence_level:.0%}': self.calculate_cvar(portfolio_returns)
            },
            'performance_ratios': {
                'sharpe': self.calculate_sharpe_ratio(portfolio_returns),
                'sortino': self.calculate_sortino_ratio(portfolio_returns)
            },
            'distribution': {
                'kurtosis': portfolio_returns.kurtosis(),
                'skewness': portfolio_returns.skew()
            },
            'drawdown': self.calculate_drawdown_metrics(portfolio_returns)
        }
        
        # 如果提供了基准收益率，计算相对指标
        if benchmark_returns is not None:
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)
            alpha = self.calculate_alpha(portfolio_returns, benchmark_returns, beta=beta)
            
            # 计算跟踪误差
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            
            metrics['relative_metrics'] = {
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error,
                'information_ratio': alpha / tracking_error if tracking_error != 0 else 0
            }
        
        # 如果提供了资产收益率和权重，计算投资组合层面的风险指标
        if asset_returns is not None and weights is not None:
            # 计算协方差矩阵
            covariance_matrix = self.calculate_covariance(asset_returns)
            
            # 计算风险贡献
            risk_contribs = self.risk_contribution(weights, covariance_matrix)
            
            # 检查风险平价
            risk_parity_result = self.risk_parity_check(risk_contribs)
            
            # 计算集中度风险
            concentration = self.calculate_concentration_risk(weights)
            
            metrics['portfolio_metrics'] = {
                'risk_contributions': risk_contribs,
                'risk_parity_check': risk_parity_result,
                'concentration_risk': concentration
            }
            
            # 如果提供了成交量数据，计算流动性风险
            if volumes is not None:
                liquidity_risk = self.calculate_liquidity_risk(asset_returns, volumes)
                metrics['portfolio_metrics']['liquidity_risk'] = liquidity_risk
            
            # 如果提供了因子收益率，进行因子风险归因
            if factor_returns is not None:
                factor_attribution = self.factor_risk_attribution(asset_returns, factor_returns, weights)
                metrics['factor_attribution'] = factor_attribution
        
        # 保存到历史记录
        date_key = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.metrics_history[date_key] = metrics
        
        logger.info("所有风险指标计算完成")
        
        return metrics
    
    def generate_risk_report(self, 
                           metrics: Dict[str, Any],
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成风险报告
        
        Args:
            metrics: 风险指标数据
            output_file: 输出文件路径
        
        Returns:
            格式化的风险报告数据
        """
        # 构建报告
        report = {
            'report_date': metrics.get('timestamp', datetime.now()),
            'summary': {
                'annualized_volatility': metrics['volatility']['standard'],
                'max_drawdown_pct': metrics['drawdown']['max_drawdown_pct'],
                'sharpe_ratio': metrics['performance_ratios']['sharpe'],
                f'var_{self.confidence_level:.0%}': metrics['var'][f'{self.confidence_level:.0%}']['historical']
            },
            'detailed_metrics': metrics
        }
        
        # 如果指定了输出文件，保存报告
        if output_file:
            try:
                import os
                # 确保目录存在
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 保存为JSON文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    # 将datetime对象转换为字符串
                    report_serializable = report.copy()
                    if 'report_date' in report_serializable and isinstance(report_serializable['report_date'], datetime):
                        report_serializable['report_date'] = report_serializable['report_date'].strftime('%Y-%m-%d %H:%M:%S')
                    if 'timestamp' in report_serializable['detailed_metrics'] and isinstance(report_serializable['detailed_metrics']['timestamp'], datetime):
                        report_serializable['detailed_metrics']['timestamp'] = report_serializable['detailed_metrics']['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    json.dump(report_serializable, f, indent=2, ensure_ascii=False)
                
                logger.info(f"风险报告已保存到: {output_file}")
            except Exception as e:
                logger.error(f"保存风险报告时发生异常: {str(e)}")
        
        logger.info("风险报告生成完成")
        
        return report
    
    def save_history(self, file_path: Optional[str] = None) -> str:
        """保存风险指标历史记录
        
        Args:
            file_path: 保存路径
        
        Returns:
            实际保存路径
        """
        try:
            import os
            logger.info("保存风险指标历史记录")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if file_path is None:
                history_dir = os.path.join(self.config.get('history_dir', './history'), 'risk_metrics')
                if not os.path.exists(history_dir):
                    os.makedirs(history_dir)
                
                file_path = os.path.join(history_dir, f"risk_metrics_history_{timestamp}.json")
            
            # 准备保存数据（转换datetime对象）
            save_data = {}
            for date_key, metrics in self.metrics_history.items():
                metrics_serializable = metrics.copy()
                if 'timestamp' in metrics_serializable and isinstance(metrics_serializable['timestamp'], datetime):
                    metrics_serializable['timestamp'] = metrics_serializable['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                save_data[date_key] = metrics_serializable
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"风险指标历史记录已保存到: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"保存风险指标历史记录时发生异常: {str(e)}")
            raise
    
    def load_history(self, file_path: str) -> bool:
        """加载风险指标历史记录
        
        Args:
            file_path: 历史记录文件路径
        
        Returns:
            是否加载成功
        """
        try:
            import os
            logger.info(f"加载风险指标历史记录: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"风险指标历史记录文件不存在: {file_path}")
                return False
            
            # 读取历史记录
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            # 恢复datetime对象
            for date_key, metrics in loaded_history.items():
                if 'timestamp' in metrics:
                    try:
                        metrics['timestamp'] = datetime.strptime(metrics['timestamp'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass  # 如果转换失败，保持原格式
            
            # 合并历史记录
            self.metrics_history.update(loaded_history)
            
            logger.info(f"成功加载 {len(loaded_history)} 条风险指标历史记录")
            
            return True
        except Exception as e:
            logger.error(f"加载风险指标历史记录时发生异常: {str(e)}")
            return False
    
    def stress_test(self, 
                  returns: Union[pd.Series, pd.DataFrame],
                  scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """进行压力测试
        
        Args:
            returns: 收益率数据
            scenarios: 自定义压力测试情景
        
        Returns:
            压力测试结果
        """
        # 定义默认的压力测试情景
        default_scenarios = [
            {'name': 'Market Crash', 'shock': -0.20},  # 市场崩盘：20%下跌
            {'name': 'Flash Crash', 'shock': -0.10},   # 闪崩：10%下跌
            {'name': 'Volatility Spike', 'vol_increase': 2.0},  # 波动率飙升：波动率翻倍
            {'name': 'Liquidity Crisis', 'correlation_increase': 0.5}  # 流动性危机：相关性增加
        ]
        
        # 使用自定义情景或默认情景
        scenarios = scenarios or default_scenarios
        
        # 准备压力测试结果
        stress_test_results = {}
        
        # 对每个情景进行测试
        for scenario in scenarios:
            scenario_name = scenario['name']
            logger.info(f"运行压力测试情景: {scenario_name}")
            
            # 根据情景类型应用冲击
            if 'shock' in scenario:
                # 直接冲击测试
                shocked_returns = returns + scenario['shock']
            elif 'vol_increase' in scenario:
                # 波动率增加测试
                if isinstance(returns, pd.DataFrame):
                    shocked_returns = returns.copy()
                    for col in returns.columns:
                        mean_return = returns[col].mean()
                        std_return = returns[col].std()
                        # 增加波动率，保持均值不变
                        shocked_returns[col] = mean_return + scenario['vol_increase'] * (returns[col] - mean_return)
                else:
                    mean_return = returns.mean()
                    std_return = returns.std()
                    # 增加波动率，保持均值不变
                    shocked_returns = mean_return + scenario['vol_increase'] * (returns - mean_return)
            elif 'correlation_increase' in scenario and isinstance(returns, pd.DataFrame):
                # 相关性增加测试（仅适用于DataFrame）
                # 这是一个简化版的实现，实际可能需要更复杂的处理
                correlation_matrix = returns.corr()
                # 增加相关性
                n_assets = len(correlation_matrix)
                ones_matrix = np.ones((n_assets, n_assets))
                shocked_correlation = correlation_matrix + scenario['correlation_increase'] * (ones_matrix - np.eye(n_assets) - correlation_matrix)
                # 确保对角线为1
                np.fill_diagonal(shocked_correlation.values, 1.0)
                # 这里应该生成符合新相关矩阵的收益率，但为简化起见，我们只记录相关矩阵
                stress_test_results[scenario_name] = {
                    'original_correlation': correlation_matrix.to_dict(),
                    'shocked_correlation': shocked_correlation.to_dict()
                }
                continue  # 跳过后续计算
            else:
                logger.warning(f"不支持的压力测试情景类型: {scenario}")
                continue
            
            # 计算压力情景下的风险指标
            if isinstance(shocked_returns, pd.DataFrame):
                # 对每个资产计算指标
                asset_results = {}
                for col in shocked_returns.columns:
                    asset_results[col] = {
                        'volatility': self.calculate_volatility(shocked_returns[col]),
                        'var': self.calculate_var(shocked_returns[col]),
                        'cvar': self.calculate_cvar(shocked_returns[col]),
                        'max_drawdown': self.calculate_drawdown_metrics(shocked_returns[col])['max_drawdown']
                    }
                stress_test_results[scenario_name] = asset_results
            else:
                # 对单个资产/投资组合计算指标
                stress_test_results[scenario_name] = {
                    'volatility': self.calculate_volatility(shocked_returns),
                    'var': self.calculate_var(shocked_returns),
                    'cvar': self.calculate_cvar(shocked_returns),
                    'max_drawdown': self.calculate_drawdown_metrics(shocked_returns)['max_drawdown']
                }
        
        logger.info("压力测试完成")
        
        return stress_test_results

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['RiskMetrics']