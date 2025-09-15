import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from scipy.optimize import minimize, Bounds, LinearConstraint

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PortfolioOptimizer:
    """投资组合优化器
    实现多种投资组合优化策略，包括均值-方差优化、风险平价、最大夏普率等
    """
    # 支持的优化目标
    OBJECTIVE_MIN_VOLATILITY = 'min_volatility'
    OBJECTIVE_MAX_SHARPE = 'max_sharpe'
    OBJECTIVE_RISK_PARITY = 'risk_parity'
    OBJECTIVE_TARGET_RETURN = 'target_return'
    OBJECTIVE_EQUALLY_WEIGHTED = 'equally_weighted'
    OBJECTIVE_MAX_DIVERSIFICATION = 'max_diversification'
    OBJECTIVE_MIN_TRACKING_ERROR = 'min_tracking_error'
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 risk_free_rate: float = 0.03,
                 lookback_period: int = 252,
                 covariance_method: str = 'standard',
                 rebalance_frequency: str = 'monthly'):
        """初始化投资组合优化器
        
        Args:
            config: 配置字典
            risk_free_rate: 无风险利率
            lookback_period: 回看期长度（交易日）
            covariance_method: 协方差计算方法
            rebalance_frequency: 再平衡频率
        """
        self.config = config or {}
        self.risk_free_rate = risk_free_rate
        self.lookback_period = lookback_period
        self.covariance_method = covariance_method
        self.rebalance_frequency = rebalance_frequency
        
        # 优化结果缓存
        self.optimization_results = {}
        
        # 初始化日志
        self._init_logger()
        
        logger.info("PortfolioOptimizer 初始化完成")
    
    def _init_logger(self):
        """初始化日志记录器"""
        import os
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def calculate_returns(self, prices: pd.DataFrame, frequency: str = 'daily') -> pd.DataFrame:
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
    
    def calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算协方差矩阵
        
        Args:
            returns: 收益率数据
        
        Returns:
            协方差矩阵
        """
        # 根据指定的方法计算协方差
        if self.covariance_method == 'standard':
            # 标准协方差计算
            covariance = returns.cov()
        elif self.covariance_method == 'shrinkage':
            # 使用Ledoit-Wolf收缩估计
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            try:
                covariance = pd.DataFrame(
                    data=lw.fit(returns).covariance_,
                    index=returns.columns,
                    columns=returns.columns
                )
            except Exception as e:
                logger.error(f"使用Ledoit-Wolf收缩估计时出错: {str(e)}，回退到标准协方差")
                covariance = returns.cov()
        elif self.covariance_method == 'ewma':
            # 使用EWMA计算协方差
            lambda_ = 0.94  # 典型的衰减因子
            n = len(returns)
            weights = np.array([(1 - lambda_) * lambda_**i for i in range(n-1, -1, -1)])
            weights /= weights.sum()
            mean_returns = returns.mean()
            centered_returns = returns - mean_returns
            covariance = centered_returns.T.dot(np.diag(weights)).dot(centered_returns)
        else:
            logger.warning(f"不支持的协方差计算方法: {self.covariance_method}，使用标准协方差")
            covariance = returns.cov()
        
        return covariance
    
    def optimize(self, 
                returns: pd.DataFrame,
                objective: str = OBJECTIVE_MAX_SHARPE,
                constraints: Optional[Dict[str, Any]] = None,
                risk_aversion: float = 1.0,
                target_return: float = None,
                benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """优化投资组合权重
        
        Args:
            returns: 收益率数据
            objective: 优化目标
            constraints: 约束条件
            risk_aversion: 风险厌恶系数
            target_return: 目标收益率
            benchmark_weights: 基准权重（用于跟踪误差最小化）
        
        Returns:
            优化后的权重字典
        """
        # 确保收益率数据不为空
        if returns.empty or len(returns.columns) < 2:
            logger.warning("收益率数据不足，无法进行优化")
            # 返回等权重
            n_assets = len(returns.columns)
            return {asset: 1.0 / n_assets for asset in returns.columns}
        
        # 提取资产列表
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # 计算预期收益率和协方差矩阵
        expected_returns = returns.mean() * 252  # 年化
        covariance = self.calculate_covariance(returns)
        
        # 处理约束条件
        bounds = Bounds(0, 1)  # 默认权重在0到1之间
        linear_constraints = LinearConstraint(np.ones((1, n_assets)), [1.0], [1.0])  # 权重和为1
        
        # 应用用户自定义约束
        if constraints:
            # 最小权重约束
            if 'min_weights' in constraints:
                min_weights = [constraints['min_weights'].get(asset, 0.0) for asset in assets]
                bounds = Bounds(min_weights, [1.0] * n_assets)
            
            # 最大权重约束
            if 'max_weights' in constraints:
                max_weights = [constraints['max_weights'].get(asset, 1.0) for asset in assets]
                current_lb = bounds.lb if isinstance(bounds, Bounds) else [0.0] * n_assets
                bounds = Bounds(current_lb, max_weights)
            
            # 行业约束
            if 'sector_constraints' in constraints and 'sector_map' in constraints:
                # 实现行业约束（这是简化版，实际应用需要更复杂的处理）
                pass
        
        # 根据优化目标选择目标函数
        if objective == self.OBJECTIVE_EQUALLY_WEIGHTED:
            # 等权重配置
            weights = {asset: 1.0 / n_assets for asset in assets}
        elif objective == self.OBJECTIVE_MIN_VOLATILITY:
            # 最小化波动率
            weights = self._minimize_volatility(expected_returns, covariance, bounds, linear_constraints)
        elif objective == self.OBJECTIVE_MAX_SHARPE:
            # 最大化夏普率
            weights = self._maximize_sharpe(expected_returns, covariance, self.risk_free_rate, bounds, linear_constraints)
        elif objective == self.OBJECTIVE_RISK_PARITY:
            # 风险平价
            weights = self._risk_parity(covariance, bounds, linear_constraints)
        elif objective == self.OBJECTIVE_TARGET_RETURN and target_return is not None:
            # 目标收益率下最小化风险
            weights = self._target_return(expected_returns, covariance, target_return, bounds, linear_constraints)
        elif objective == self.OBJECTIVE_MAX_DIVERSIFICATION:
            # 最大化多样化比率
            weights = self._max_diversification(covariance, bounds, linear_constraints)
        elif objective == self.OBJECTIVE_MIN_TRACKING_ERROR and benchmark_weights is not None:
            # 最小化跟踪误差
            weights = self._min_tracking_error(expected_returns, covariance, benchmark_weights, bounds, linear_constraints)
        else:
            logger.warning(f"不支持的优化目标: {objective}，使用等权重")
            weights = {asset: 1.0 / n_assets for asset in assets}
        
        # 规范化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {asset: weight / total_weight for asset, weight in weights.items()}
        
        # 保存优化结果
        optimization_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.optimization_results[optimization_id] = {
            'weights': weights,
            'objective': objective,
            'expected_returns': expected_returns.to_dict(),
            'covariance': covariance.to_dict(),
            'timestamp': datetime.now(),
            'constraints': constraints
        }
        
        logger.info(f"投资组合优化完成，目标: {objective}")
        
        return weights
    
    def _minimize_volatility(self, 
                            expected_returns: pd.Series,
                            covariance: pd.DataFrame,
                            bounds: Bounds,
                            linear_constraints: LinearConstraint) -> Dict[str, float]:
        """最小化投资组合波动率
        
        Args:
            expected_returns: 预期收益率
            covariance: 协方差矩阵
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        n_assets = len(expected_returns)
        
        # 定义目标函数：波动率
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)  # 年化波动率
        
        # 初始猜测（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints
        )
        
        # 提取结果
        weights = {asset: weight for asset, weight in zip(expected_returns.index, result.x)}
        
        return weights
    
    def _maximize_sharpe(self, 
                        expected_returns: pd.Series,
                        covariance: pd.DataFrame,
                        risk_free_rate: float,
                        bounds: Bounds,
                        linear_constraints: LinearConstraint) -> Dict[str, float]:
        """最大化夏普率
        
        Args:
            expected_returns: 预期收益率
            covariance: 协方差矩阵
            risk_free_rate: 无风险利率
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        n_assets = len(expected_returns)
        
        # 定义目标函数：负的夏普率（因为要最小化）
        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # 取负值以进行最小化
        
        # 初始猜测（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints
        )
        
        # 提取结果
        weights = {asset: weight for asset, weight in zip(expected_returns.index, result.x)}
        
        return weights
    
    def _risk_parity(self, 
                    covariance: pd.DataFrame,
                    bounds: Bounds,
                    linear_constraints: LinearConstraint) -> Dict[str, float]:
        """风险平价优化
        
        Args:
            covariance: 协方差矩阵
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        n_assets = len(covariance)
        
        # 定义目标函数：风险贡献的平方差之和
        def risk_parity_objective(weights):
            # 计算投资组合波动率
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)
            
            # 计算边际风险贡献
            mrc = np.dot(covariance, weights) * np.sqrt(252) / portfolio_vol
            
            # 计算风险贡献
            rc = weights * mrc
            
            # 目标是让所有资产的风险贡献相等
            target_rc = np.ones(n_assets) / n_assets
            
            # 计算平方差之和
            return np.sum((rc - target_rc * portfolio_vol) ** 2)
        
        # 初始猜测（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints
        )
        
        # 提取结果
        assets = covariance.index.tolist()
        weights = {asset: weight for asset, weight in zip(assets, result.x)}
        
        return weights
    
    def _target_return(self, 
                      expected_returns: pd.Series,
                      covariance: pd.DataFrame,
                      target_return: float,
                      bounds: Bounds,
                      linear_constraints: LinearConstraint) -> Dict[str, float]:
        """在目标收益率下最小化风险
        
        Args:
            expected_returns: 预期收益率
            covariance: 协方差矩阵
            target_return: 目标收益率
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        n_assets = len(expected_returns)
        
        # 定义目标函数：波动率
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)
        
        # 创建收益率约束
        returns_constraint = LinearConstraint(expected_returns.values, [target_return], [np.inf])
        
        # 合并约束
        all_constraints = [linear_constraints, returns_constraint]
        
        # 初始猜测（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints
        )
        
        # 提取结果
        weights = {asset: weight for asset, weight in zip(expected_returns.index, result.x)}
        
        return weights
    
    def _max_diversification(self, 
                            covariance: pd.DataFrame,
                            bounds: Bounds,
                            linear_constraints: LinearConstraint) -> Dict[str, float]:
        """最大化多样化比率
        
        Args:
            covariance: 协方差矩阵
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        n_assets = len(covariance)
        
        # 计算资产的标准差
        std_dev = np.sqrt(np.diag(covariance)) * np.sqrt(252)  # 年化标准差
        
        # 定义目标函数：负的多样化比率（因为要最小化）
        def neg_diversification_ratio(weights):
            # 计算投资组合波动率
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)
            
            # 计算加权平均标准差
            weighted_avg_std = np.sum(weights * std_dev)
            
            # 计算多样化比率
            diversification_ratio = weighted_avg_std / portfolio_vol
            
            return -diversification_ratio  # 取负值以进行最小化
        
        # 初始猜测（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            neg_diversification_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints
        )
        
        # 提取结果
        assets = covariance.index.tolist()
        weights = {asset: weight for asset, weight in zip(assets, result.x)}
        
        return weights
    
    def _min_tracking_error(self, 
                           expected_returns: pd.Series,
                           covariance: pd.DataFrame,
                           benchmark_weights: Dict[str, float],
                           bounds: Bounds,
                           linear_constraints: LinearConstraint) -> Dict[str, float]:
        """最小化跟踪误差
        
        Args:
            expected_returns: 预期收益率
            covariance: 协方差矩阵
            benchmark_weights: 基准权重
            bounds: 权重边界
            linear_constraints: 线性约束
        
        Returns:
            优化后的权重字典
        """
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        # 将基准权重转换为数组
        benchmark_weights_array = np.array([benchmark_weights.get(asset, 0.0) for asset in assets])
        
        # 定义目标函数：跟踪误差
        def tracking_error(weights):
            weight_diff = weights - benchmark_weights_array
            return np.sqrt(np.dot(weight_diff.T, np.dot(covariance, weight_diff))) * np.sqrt(252)
        
        # 初始猜测（基准权重）
        initial_weights = benchmark_weights_array.copy()
        
        # 优化
        result = minimize(
            tracking_error,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints
        )
        
        # 提取结果
        weights = {asset: weight for asset, weight in zip(assets, result.x)}
        
        return weights
    
    def calculate_portfolio_metrics(self, 
                                   weights: Dict[str, float],
                                   returns: pd.DataFrame) -> Dict[str, float]:
        """计算投资组合指标
        
        Args:
            weights: 投资组合权重
            returns: 收益率数据
        
        Returns:
            投资组合指标字典
        """
        # 确保权重和收益率数据匹配
        common_assets = [asset for asset in weights.keys() if asset in returns.columns]
        if not common_assets:
            logger.warning("没有共同的资产，无法计算投资组合指标")
            return {}
        
        # 过滤权重和收益率数据
        filtered_weights = {asset: weights[asset] for asset in common_assets}
        filtered_returns = returns[common_assets]
        
        # 规范化权重
        total_weight = sum(filtered_weights.values())
        normalized_weights = np.array([filtered_weights[asset] / total_weight for asset in common_assets])
        
        # 计算投资组合收益率
        portfolio_returns = filtered_returns.dot(normalized_weights)
        
        # 计算指标
        metrics = {
            'annualized_return': portfolio_returns.mean() * 252,
            'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - self.risk_free_rate) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'value_at_risk': self._calculate_var(portfolio_returns),
            'conditional_var': self._calculate_cvar(portfolio_returns),
            'diversification_ratio': self._calculate_diversification_ratio(filtered_weights, returns)
        }
        
        logger.info(f"计算的投资组合指标: {metrics}")
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤
        
        Args:
            returns: 收益率序列
        
        Returns:
            最大回撤
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算Sortino比率
        
        Args:
            returns: 收益率序列
        
        Returns:
            Sortino比率
        """
        # 计算年化收益率
        annualized_return = returns.mean() * 252
        
        # 计算下行波动率
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
        else:
            downside_vol = 1e-10  # 避免除以零
        
        # 计算Sortino比率
        return (annualized_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算Calmar比率
        
        Args:
            returns: 收益率序列
        
        Returns:
            Calmar比率
        """
        # 计算年化收益率
        annualized_return = returns.mean() * 252
        
        # 计算最大回撤
        max_dd = self._calculate_max_drawdown(returns)
        
        # 计算Calmar比率
        return annualized_return / abs(max_dd) if max_dd != 0 else 0
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算Value at Risk (VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            VaR值
        """
        # 使用历史模拟法计算VaR
        var = returns.quantile(1 - confidence_level)
        return var
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算Conditional Value at Risk (CVaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            CVaR值
        """
        # 计算VaR
        var = self._calculate_var(returns, confidence_level)
        
        # 计算超过VaR的平均损失
        cvar = returns[returns <= var].mean()
        return cvar
    
    def _calculate_diversification_ratio(self, 
                                        weights: Dict[str, float],
                                        returns: pd.DataFrame) -> float:
        """计算多样化比率
        
        Args:
            weights: 投资组合权重
            returns: 收益率数据
        
        Returns:
            多样化比率
        """
        # 确保权重和收益率数据匹配
        common_assets = [asset for asset in weights.keys() if asset in returns.columns]
        if not common_assets:
            return 1.0  # 最小多样化
        
        # 过滤权重和计算协方差
        filtered_weights = np.array([weights[asset] for asset in common_assets])
        filtered_returns = returns[common_assets]
        covariance = filtered_returns.cov()
        
        # 计算资产的标准差
        std_dev = np.sqrt(np.diag(covariance)) * np.sqrt(252)  # 年化标准差
        
        # 计算投资组合波动率
        portfolio_vol = np.sqrt(np.dot(filtered_weights.T, np.dot(covariance, filtered_weights))) * np.sqrt(252)
        
        # 计算加权平均标准差
        weighted_avg_std = np.sum(filtered_weights * std_dev)
        
        # 计算多样化比率
        if portfolio_vol > 0:
            diversification_ratio = weighted_avg_std / portfolio_vol
        else:
            diversification_ratio = 1.0  # 最小多样化
        
        return diversification_ratio
    
    def generate_efficient_frontier(self, 
                                  returns: pd.DataFrame,
                                  n_points: int = 20,
                                  constraints: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """生成有效前沿
        
        Args:
            returns: 收益率数据
            n_points: 有效前沿上的点数
            constraints: 约束条件
        
        Returns:
            有效前沿数据DataFrame
        """
        # 计算预期收益率和协方差矩阵
        expected_returns = returns.mean() * 252  # 年化
        covariance = self.calculate_covariance(returns)
        
        # 计算最小方差组合
        min_vol_weights = self._minimize_volatility(expected_returns, covariance, Bounds(0, 1), LinearConstraint(np.ones((1, len(expected_returns))), [1.0], [1.0]))
        min_vol_portfolio = self.calculate_portfolio_metrics(min_vol_weights, returns)
        
        # 计算最大夏普率组合
        max_sharpe_weights = self._maximize_sharpe(expected_returns, covariance, self.risk_free_rate, Bounds(0, 1), LinearConstraint(np.ones((1, len(expected_returns))), [1.0], [1.0]))
        max_sharpe_portfolio = self.calculate_portfolio_metrics(max_sharpe_weights, returns)
        
        # 生成目标收益率序列
        if max_sharpe_portfolio['annualized_return'] > min_vol_portfolio['annualized_return']:
            target_returns = np.linspace(min_vol_portfolio['annualized_return'], max_sharpe_portfolio['annualized_return'], n_points)
        else:
            # 如果最大夏普率组合的收益率低于最小方差组合，使用不同的范围
            min_return = min(expected_returns.min(), min_vol_portfolio['annualized_return'])
            max_return = max(expected_returns.max(), max_sharpe_portfolio['annualized_return'])
            target_returns = np.linspace(min_return, max_return, n_points)
        
        # 对每个目标收益率计算最小方差组合
        frontier_data = []
        for target_return in target_returns:
            try:
                # 计算目标收益率下的最小方差组合
                weights = self.optimize(
                    returns=returns,
                    objective=self.OBJECTIVE_TARGET_RETURN,
                    target_return=target_return,
                    constraints=constraints
                )
                
                # 计算组合指标
                metrics = self.calculate_portfolio_metrics(weights, returns)
                
                # 添加到结果
                frontier_data.append({
                    'target_return': target_return,
                    'volatility': metrics['annualized_volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'weights': weights
                })
            except Exception as e:
                logger.warning(f"无法计算目标收益率 {target_return} 的组合: {str(e)}")
                continue
        
        # 创建DataFrame
        frontier_df = pd.DataFrame(frontier_data)
        
        # 添加最大夏普率和最小方差组合标记
        if not frontier_df.empty:
            # 找到最接近最大夏普率的点
            max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
            frontier_df.loc[max_sharpe_idx, 'type'] = 'max_sharpe'
            
            # 找到最接近最小方差的点
            min_vol_idx = frontier_df['volatility'].idxmin()
            frontier_df.loc[min_vol_idx, 'type'] = 'min_vol'
        
        logger.info(f"生成有效前沿完成，包含 {len(frontier_df)} 个点")
        
        return frontier_df
    
    def backtest_portfolio(self, 
                          prices: pd.DataFrame,
                          optimization_params: Dict[str, Any],
                          initial_capital: float = 1000000.0,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """回测投资组合策略
        
        Args:
            prices: 价格数据
            optimization_params: 优化参数
            initial_capital: 初始资金
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            回测结果DataFrame
        """
        # 过滤日期范围
        if start_date is not None:
            prices = prices[prices.index >= start_date]
        if end_date is not None:
            prices = prices[prices.index <= end_date]
        
        # 计算收益率
        returns = self.calculate_returns(prices)
        
        # 准备回测结果DataFrame
        backtest_results = pd.DataFrame(index=prices.index)
        backtest_results['equity'] = initial_capital
        backtest_results['portfolio_return'] = 0.0
        backtest_results['weights'] = None
        
        # 确定再平衡日期
        rebalance_dates = self._determine_rebalance_dates(prices.index)
        
        # 初始化当前权重
        current_weights = None
        
        # 回测主循环
        for i, date in enumerate(prices.index):
            if date in rebalance_dates:
                # 在再平衡日期，重新优化投资组合
                logger.info(f"在 {date} 进行投资组合再平衡")
                
                # 计算回看期数据
                lookback_start = date - pd.DateOffset(days=self.lookback_period * 5 // 7)  # 估计的交易日数量
                lookback_prices = prices[prices.index < date]
                lookback_prices = lookback_prices[lookback_prices.index >= lookback_start]
                
                if len(lookback_prices) >= 30:  # 确保有足够的数据进行优化
                    # 计算回看期收益率
                    lookback_returns = self.calculate_returns(lookback_prices)
                    
                    # 优化投资组合
                    current_weights = self.optimize(returns=lookback_returns, **optimization_params)
                    
                    # 保存权重
                    backtest_results.loc[date, 'weights'] = current_weights
            
            # 计算当日收益
            if i > 0 and current_weights is not None:
                prev_date = prices.index[i-1]
                
                # 计算每个资产的日收益率
                daily_returns = prices.loc[date] / prices.loc[prev_date] - 1
                
                # 计算投资组合日收益率
                portfolio_return = 0.0
                for asset, weight in current_weights.items():
                    if asset in daily_returns:
                        portfolio_return += weight * daily_returns[asset]
                
                # 更新权益
                backtest_results.loc[date, 'portfolio_return'] = portfolio_return
                backtest_results.loc[date, 'equity'] = backtest_results.loc[prev_date, 'equity'] * (1 + portfolio_return)
        
        # 计算累积收益率
        backtest_results['cumulative_return'] = backtest_results['equity'] / initial_capital - 1
        
        # 计算基准（等权重）组合
        n_assets = len(prices.columns)
        equal_weights = {asset: 1.0 / n_assets for asset in prices.columns}
        
        # 计算基准组合收益率
        backtest_results['benchmark_return'] = 0.0
        backtest_results['benchmark_equity'] = initial_capital
        
        for i, date in enumerate(prices.index):
            if i > 0:
                prev_date = prices.index[i-1]
                
                # 计算每个资产的日收益率
                daily_returns = prices.loc[date] / prices.loc[prev_date] - 1
                
                # 计算基准组合日收益率
                benchmark_return = 0.0
                for asset, weight in equal_weights.items():
                    if asset in daily_returns:
                        benchmark_return += weight * daily_returns[asset]
                
                # 更新基准权益
                backtest_results.loc[date, 'benchmark_return'] = benchmark_return
                backtest_results.loc[date, 'benchmark_equity'] = backtest_results.loc[prev_date, 'benchmark_equity'] * (1 + benchmark_return)
        
        # 计算基准累积收益率
        backtest_results['benchmark_cumulative_return'] = backtest_results['benchmark_equity'] / initial_capital - 1
        
        logger.info(f"投资组合回测完成，最终权益: {backtest_results['equity'].iloc[-1]:.2f}，总收益率: {backtest_results['cumulative_return'].iloc[-1]:.2%}")
        
        return backtest_results
    
    def _determine_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """确定再平衡日期
        
        Args:
            dates: 日期索引
        
        Returns:
            再平衡日期列表
        """
        if self.rebalance_frequency == 'daily':
            return dates.tolist()
        elif self.rebalance_frequency == 'weekly':
            # 选择每周最后一个交易日
            weekly_groups = dates.to_series().groupby(pd.Grouper(freq='W'))
            return [group.index[-1] for _, group in weekly_groups]
        elif self.rebalance_frequency == 'monthly':
            # 选择每月最后一个交易日
            monthly_groups = dates.to_series().groupby(pd.Grouper(freq='M'))
            return [group.index[-1] for _, group in monthly_groups]
        elif self.rebalance_frequency == 'quarterly':
            # 选择每季度最后一个交易日
            quarterly_groups = dates.to_series().groupby(pd.Grouper(freq='Q'))
            return [group.index[-1] for _, group in quarterly_groups]
        elif self.rebalance_frequency == 'yearly':
            # 选择每年最后一个交易日
            yearly_groups = dates.to_series().groupby(pd.Grouper(freq='Y'))
            return [group.index[-1] for _, group in yearly_groups]
        else:
            logger.warning(f"不支持的再平衡频率: {self.rebalance_frequency}，使用月度再平衡")
            monthly_groups = dates.to_series().groupby(pd.Grouper(freq='M'))
            return [group.index[-1] for _, group in monthly_groups]
    
    def save_optimization_results(self, 
                                 file_path: Optional[str] = None,
                                 optimization_id: Optional[str] = None) -> str:
        """保存优化结果
        
        Args:
            file_path: 保存路径
            optimization_id: 优化结果ID
        
        Returns:
            实际保存路径
        """
        try:
            import os
            logger.info("保存优化结果")
            
            # 确定要保存的结果
            if optimization_id and optimization_id in self.optimization_results:
                results_to_save = {optimization_id: self.optimization_results[optimization_id]}
            else:
                results_to_save = self.optimization_results
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if file_path is None:
                results_dir = os.path.join(self.config.get('results_dir', './results'), 'portfolio_optimization')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                file_path = os.path.join(results_dir, f"portfolio_optimization_{timestamp}.json")
            
            # 准备保存数据（转换datetime对象）
            save_data = {}
            for opt_id, result in results_to_save.items():
                result_serializable = result.copy()
                if 'timestamp' in result_serializable and isinstance(result_serializable['timestamp'], datetime):
                    result_serializable['timestamp'] = result_serializable['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                save_data[opt_id] = result_serializable
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"优化结果已保存到: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"保存优化结果时发生异常: {str(e)}")
            raise
    
    def load_optimization_results(self, file_path: str) -> bool:
        """加载优化结果
        
        Args:
            file_path: 结果文件路径
        
        Returns:
            是否加载成功
        """
        try:
            import os
            logger.info(f"加载优化结果: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"优化结果文件不存在: {file_path}")
                return False
            
            # 读取结果
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            # 恢复datetime对象
            for opt_id, result in loaded_results.items():
                if 'timestamp' in result:
                    try:
                        result['timestamp'] = datetime.strptime(result['timestamp'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass  # 如果转换失败，保持原格式
                
                # 恢复DataFrame对象（如果需要）
                if 'expected_returns' in result and isinstance(result['expected_returns'], dict):
                    result['expected_returns'] = pd.Series(result['expected_returns'])
                if 'covariance' in result and isinstance(result['covariance'], dict):
                    result['covariance'] = pd.DataFrame(result['covariance'])
            
            # 合并结果
            self.optimization_results.update(loaded_results)
            
            logger.info(f"成功加载 {len(loaded_results)} 条优化结果")
            
            return True
        except Exception as e:
            logger.error(f"加载优化结果时发生异常: {str(e)}")
            return False

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['PortfolioOptimizer']