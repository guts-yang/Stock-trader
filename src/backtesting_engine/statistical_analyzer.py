import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any, Set
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
import itertools
import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StatisticalAnalyzer:
    """统计分析器
    提供高级统计分析功能，如蒙特卡洛模拟、参数敏感性分析等
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        """初始化统计分析器
        
        Args:
            config: 配置字典
            log_level: 日志级别
        """
        self.config = config or {}
        self.returns = None  # 收益率数据
        self.trades = None  # 交易记录
        self.risk_free_rate = self.config.get('risk_free_rate', 0.0)  # 无风险利率
        
        # 初始化日志
        self._init_logger(log_level)
        
        logger.info("统计分析器初始化完成")
    
    def _init_logger(self, log_level: int):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"statistical_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def set_data(self, 
                returns: Optional[pd.Series] = None,
                trades: Optional[pd.DataFrame] = None):
        """设置分析数据
        
        Args:
            returns: 收益率数据
            trades: 交易记录
        """
        # 设置收益率数据
        if returns is not None:
            self.returns = returns
            # 确保索引是datetime类型
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                try:
                    self.returns.index = pd.to_datetime(self.returns.index)
                except:
                    logger.warning("无法将收益率索引转换为datetime类型")
        
        # 设置交易记录
        if trades is not None:
            self.trades = trades
        
        logger.info("分析数据已设置")
    
    def monte_carlo_simulation(self, 
                              num_simulations: int = 1000,
                              time_horizon: int = 252,
                              initial_value: float = 1.0,
                              distribution: str = 'normal',
                              use_bootstrap: bool = False,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """进行蒙特卡洛模拟
        
        Args:
            num_simulations: 模拟次数
            time_horizon: 时间范围（交易日）
            initial_value: 初始值
            distribution: 收益率分布 ('normal', 't', 'bootstrap')
            use_bootstrap: 是否使用自助法
            save_path: 结果保存路径
        
        Returns:
            模拟结果字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法进行蒙特卡洛模拟")
            return {}
        
        try:
            # 计算历史收益率的统计参数
            mu = self.returns.mean()
            sigma = self.returns.std()
            
            # 生成随机收益率路径
            simulated_paths = np.zeros((time_horizon, num_simulations))
            simulated_paths[0] = initial_value
            
            logger.info(f"开始蒙特卡洛模拟: {num_simulations}次模拟，时间范围{time_horizon}天")
            
            # 根据选择的分布生成随机数
            if distribution == 'normal' or (use_bootstrap and distribution != 'bootstrap'):
                # 正态分布
                for t in range(1, time_horizon):
                    # 生成随机收益率
                    if use_bootstrap:
                        # 使用自助法从历史收益率中采样
                        random_returns = self.returns.sample(n=num_simulations, replace=True).values
                    else:
                        # 从正态分布中生成随机数
                        random_returns = np.random.normal(mu, sigma, num_simulations)
                    
                    # 更新模拟路径
                    simulated_paths[t] = simulated_paths[t-1] * (1 + random_returns)
            elif distribution == 't':
                # t分布
                # 估计t分布的自由度
                df, loc, scale = stats.t.fit(self.returns)
                
                for t in range(1, time_horizon):
                    # 从t分布中生成随机数
                    random_returns = stats.t.rvs(df, loc=loc, scale=scale, size=num_simulations)
                    
                    # 更新模拟路径
                    simulated_paths[t] = simulated_paths[t-1] * (1 + random_returns)
            elif distribution == 'bootstrap':
                # 完全基于历史数据的自助法
                # 创建日期索引
                date_index = pd.date_range(start=datetime.now(), periods=time_horizon)
                
                # 初始化模拟路径数组
                simulated_returns = np.zeros((time_horizon, num_simulations))
                
                # 对每个模拟路径
                for i in range(num_simulations):
                    # 随机采样历史收益率（有放回）
                    sampled_returns = self.returns.sample(n=time_horizon, replace=True).values
                    simulated_returns[:, i] = sampled_returns
                
                # 计算累积收益
                cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)
                simulated_paths = initial_value * cumulative_returns
            else:
                raise ValueError(f"不支持的分布类型: {distribution}")
            
            # 计算模拟结果的统计量
            final_values = simulated_paths[-1]
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)
            var_95 = np.percentile(final_values, 5)
            var_99 = np.percentile(final_values, 1)
            cvar_95 = np.mean(final_values[final_values <= var_95])
            cvar_99 = np.mean(final_values[final_values <= var_99])
            
            # 创建结果字典
            results = {
                'num_simulations': num_simulations,
                'time_horizon': time_horizon,
                'initial_value': initial_value,
                'distribution': distribution,
                'use_bootstrap': use_bootstrap,
                'mean_final_value': mean_final_value,
                'median_final_value': median_final_value,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'simulated_paths': simulated_paths,
                'final_values': final_values,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存结果
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 保存为JSON文件，但排除大型数组
                results_to_save = results.copy()
                results_to_save.pop('simulated_paths', None)
                results_to_save.pop('final_values', None)
                
                # 保存JSON文件
                with open(save_path, 'w') as f:
                    json.dump(results_to_save, f, indent=4, default=str)
                
                # 保存模拟路径数组为CSV文件
                if 'simulated_paths' in results:
                    paths_df = pd.DataFrame(simulated_paths)
                    paths_csv_path = save_path.replace('.json', '_paths.csv')
                    paths_df.to_csv(paths_csv_path, index=False)
                
                logger.info(f"蒙特卡洛模拟结果已保存到: {save_path}")
            
            logger.info(f"蒙特卡洛模拟完成: 平均最终值={mean_final_value:.2f}, 中位数最终值={median_final_value:.2f}")
            return results
        except Exception as e:
            logger.error(f"进行蒙特卡洛模拟时发生异常: {str(e)}")
            return {}
    
    def plot_monte_carlo_results(self, 
                                simulation_results: Dict[str, Any],
                                num_paths_to_plot: int = 100,
                                show_confidence_bands: bool = True,
                                save_path: Optional[str] = None) -> plt.Figure:
        """绘制蒙特卡洛模拟结果
        
        Args:
            simulation_results: 蒙特卡洛模拟结果
            num_paths_to_plot: 要绘制的路径数量
            show_confidence_bands: 是否显示置信区间
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            # 从结果中提取数据
            simulated_paths = simulation_results.get('simulated_paths')
            if simulated_paths is None:
                logger.warning("模拟结果中没有找到路径数据")
                return None
            
            time_horizon = simulation_results.get('time_horizon', simulated_paths.shape[0])
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制模拟路径
            time_index = range(time_horizon)
            
            # 随机选择一部分路径进行绘制
            if simulated_paths.shape[1] > num_paths_to_plot:
                indices_to_plot = np.random.choice(simulated_paths.shape[1], num_paths_to_plot, replace=False)
                paths_to_plot = simulated_paths[:, indices_to_plot]
            else:
                paths_to_plot = simulated_paths
            
            # 绘制路径
            ax1.plot(time_index, paths_to_plot, alpha=0.1)
            
            # 绘制均值路径
            mean_path = np.mean(simulated_paths, axis=1)
            ax1.plot(time_index, mean_path, 'r-', linewidth=2, label='Mean Path')
            
            # 绘制置信区间
            if show_confidence_bands:
                upper_band = np.percentile(simulated_paths, 95, axis=1)
                lower_band = np.percentile(simulated_paths, 5, axis=1)
                ax1.fill_between(time_index, lower_band, upper_band, color='r', alpha=0.2, label='90% Confidence Band')
            
            # 设置标题和标签
            ax1.set_title('Monte Carlo Simulation Paths')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Value')
            ax1.grid(True)
            ax1.legend()
            
            # 绘制最终值分布
            final_values = simulation_results.get('final_values', simulated_paths[-1])
            ax2.hist(final_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
            
            # 标记统计量
            mean_final_value = simulation_results.get('mean_final_value', np.mean(final_values))
            median_final_value = simulation_results.get('median_final_value', np.median(final_values))
            var_95 = simulation_results.get('var_95', np.percentile(final_values, 5))
            var_99 = simulation_results.get('var_99', np.percentile(final_values, 1))
            
            ax2.axvline(mean_final_value, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_final_value:.2f}')
            ax2.axvline(median_final_value, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_final_value:.2f}')
            ax2.axvline(var_95, color='orange', linestyle='dashed', linewidth=1, label=f'VaR 95%: {var_95:.2f}')
            ax2.axvline(var_99, color='purple', linestyle='dashed', linewidth=1, label=f'VaR 99%: {var_99:.2f}')
            
            # 设置标题和标签
            ax2.set_title('Distribution of Final Values')
            ax2.set_xlabel('Final Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            ax2.legend()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"蒙特卡洛模拟结果图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制蒙特卡洛模拟结果时发生异常: {str(e)}")
            return None
    
    def parameter_sensitivity_analysis(self, 
                                      strategy_class: Callable,
                                      parameter_ranges: Dict[str, List[Any]],
                                      backtester: Any,
                                      metric: str = 'sharpe_ratio',
                                      num_jobs: int = -1) -> Dict[str, Any]:
        """进行参数敏感性分析
        
        Args:
            strategy_class: 策略类
            parameter_ranges: 参数范围字典
            backtester: 回测器实例
            metric: 评估指标
            num_jobs: 并行任务数量 (-1 表示使用所有可用CPU)
        
        Returns:
            敏感性分析结果
        """
        try:
            # 获取参数名称和值列表
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # 生成所有参数组合
            param_combinations = list(itertools.product(*param_values))
            
            logger.info(f"开始参数敏感性分析: {len(param_combinations)}个参数组合")
            
            # 定义单个参数组合的回测函数
            def backtest_single_param_set(param_values):
                try:
                    # 创建参数字典
                    params = dict(zip(param_names, param_values))
                    
                    # 创建策略实例
                    strategy = strategy_class(**params)
                    
                    # 设置回测器的策略
                    backtester.set_strategy(strategy)
                    
                    # 运行回测
                    backtester.run()
                    
                    # 获取绩效指标
                    performance = backtester.get_performance_metrics()
                    
                    # 返回结果
                    result = {
                        'params': params,
                        'metrics': performance
                    }
                    
                    return result
                except Exception as e:
                    logger.error(f"回测参数组合 {param_values} 时发生异常: {str(e)}")
                    return None
            
            # 并行运行回测
            if num_jobs == 1:
                # 串行运行
                results = []
                for params in tqdm(param_combinations, desc="参数敏感性分析"):
                    result = backtest_single_param_set(params)
                    if result is not None:
                        results.append(result)
            else:
                # 并行运行
                # 限制工作进程数量
                if num_jobs < 0:
                    num_jobs = mp.cpu_count()
                
                # 使用joblib进行并行处理
                with Parallel(n_jobs=num_jobs) as parallel:
                    results = parallel(
                        delayed(backtest_single_param_set)(params) 
                        for params in param_combinations
                    )
                
                # 过滤掉None结果
                results = [r for r in results if r is not None]
            
            # 构建结果数据框
            if results:
                # 提取参数和指标
                params_list = [r['params'] for r in results]
                metrics_list = [r['metrics'] for r in results]
                
                # 创建参数数据框
                params_df = pd.DataFrame(params_list)
                
                # 提取指定指标
                if isinstance(metrics_list[0], dict):
                    # 如果指标是字典，提取指定的指标
                    metric_values = [m.get(metric, 0) for m in metrics_list]
                else:
                    # 否则，假设指标列表中的每个元素就是指标值
                    metric_values = metrics_list
                
                # 添加指标到参数数据框
                params_df[metric] = metric_values
                
                # 按指标排序
                params_df_sorted = params_df.sort_values(by=metric, ascending=False)
                
                # 创建结果字典
                sensitivity_results = {
                    'param_names': param_names,
                    'parameter_ranges': parameter_ranges,
                    'results': results,
                    'results_df': params_df,
                    'sorted_results_df': params_df_sorted,
                    'best_params': params_df_sorted.iloc[0].to_dict() if not params_df_sorted.empty else {},
                    'metric': metric,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"参数敏感性分析完成: 找到最佳参数组合")
                return sensitivity_results
            else:
                logger.warning("参数敏感性分析没有产生任何结果")
                return {}
        except Exception as e:
            logger.error(f"进行参数敏感性分析时发生异常: {str(e)}")
            return {}
    
    def plot_parameter_sensitivity(self, 
                                 sensitivity_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """绘制参数敏感性分析结果
        
        Args:
            sensitivity_results: 参数敏感性分析结果
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            # 从结果中提取数据
            params_df = sensitivity_results.get('results_df')
            if params_df is None:
                logger.warning("敏感性分析结果中没有找到数据框")
                return None
            
            metric = sensitivity_results.get('metric', 'sharpe_ratio')
            param_names = sensitivity_results.get('param_names', [])
            
            # 确定子图数量和布局
            num_params = len(param_names)
            if num_params <= 2:
                fig, axes = plt.subplots(num_params, 1, figsize=(10, 5 * num_params))
                if num_params == 1:
                    axes = [axes]
            else:
                fig, axes = plt.subplots(int(np.ceil(num_params / 2)), 2, figsize=(15, 5 * int(np.ceil(num_params / 2))))
                axes = axes.flatten()
            
            # 对每个参数绘制敏感性图
            for i, param_name in enumerate(param_names):
                if i < len(axes):
                    # 计算每个参数值对应的平均指标值
                    param_values = params_df[param_name].unique()
                    param_values.sort()
                    
                    # 计算每个参数值的平均指标值
                    avg_metric_values = []
                    for val in param_values:
                        avg_metric = params_df[params_df[param_name] == val][metric].mean()
                        avg_metric_values.append(avg_metric)
                    
                    # 绘制图表
                    axes[i].plot(param_values, avg_metric_values, 'o-')
                    axes[i].set_title(f'Sensitivity to {param_name}')
                    axes[i].set_xlabel(param_name)
                    axes[i].set_ylabel(metric)
                    axes[i].grid(True)
            
            # 移除多余的子图
            for i in range(num_params, len(axes)):
                fig.delaxes(axes[i])
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"参数敏感性分析结果图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制参数敏感性分析结果时发生异常: {str(e)}")
            return None
    
    def correlation_analysis(self, 
                           additional_data: Optional[Dict[str, pd.Series]] = None,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """进行相关性分析
        
        Args:
            additional_data: 要进行相关性分析的其他数据
            save_path: 保存路径
        
        Returns:
            相关性分析结果
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法进行相关性分析")
            return {}
        
        try:
            # 创建数据框
            data = {
                'strategy_returns': self.returns
            }
            
            # 添加其他数据
            if additional_data:
                data.update(additional_data)
            
            # 创建数据框
            df = pd.DataFrame(data)
            
            # 计算相关性矩阵
            corr_matrix = df.corr()
            
            # 计算p值矩阵
            p_values = np.zeros((len(df.columns), len(df.columns)))
            for i in range(len(df.columns)):
                for j in range(len(df.columns)):
                    if i != j:
                        # 计算两个变量的相关性和p值
                        corr, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                        p_values[i, j] = p
                    else:
                        p_values[i, j] = 0  # 对角线元素p值设为0
            
            # 创建p值数据框
            p_values_df = pd.DataFrame(p_values, index=df.columns, columns=df.columns)
            
            # 创建结果字典
            results = {
                'correlation_matrix': corr_matrix,
                'p_values_matrix': p_values_df,
                'data': df,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存结果
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 保存相关性矩阵为CSV文件
                corr_matrix.to_csv(save_path.replace('.json', '_correlation.csv'))
                
                # 保存p值矩阵为CSV文件
                p_values_df.to_csv(save_path.replace('.json', '_pvalues.csv'))
                
                # 保存其他结果为JSON文件
                results_to_save = results.copy()
                results_to_save.pop('correlation_matrix', None)
                results_to_save.pop('p_values_matrix', None)
                results_to_save.pop('data', None)
                
                with open(save_path, 'w') as f:
                    json.dump(results_to_save, f, indent=4, default=str)
                
                logger.info(f"相关性分析结果已保存到: {save_path}")
            
            logger.info("相关性分析完成")
            return results
        except Exception as e:
            logger.error(f"进行相关性分析时发生异常: {str(e)}")
            return {}
    
    def plot_correlation_heatmap(self, 
                               correlation_results: Dict[str, Any],
                               show_p_values: bool = False,
                               save_path: Optional[str] = None) -> plt.Figure:
        """绘制相关性热力图
        
        Args:
            correlation_results: 相关性分析结果
            show_p_values: 是否显示p值
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        try:
            # 从结果中提取数据
            corr_matrix = correlation_results.get('correlation_matrix')
            if corr_matrix is None:
                logger.warning("相关性分析结果中没有找到相关性矩阵")
                return None
            
            p_values_df = correlation_results.get('p_values_matrix')
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制热力图
            if show_p_values and p_values_df is not None:
                # 绘制带p值的热力图
                # 首先绘制相关性矩阵
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, ax=ax)
                
                # 在热力图上添加p值
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        if i != j:
                            # 相关性值
                            corr_val = corr_matrix.iloc[i, j]
                            # p值
                            p_val = p_values_df.iloc[i, j]
                            # 确定文本颜色（根据背景色）
                            text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                            # 添加文本
                            ax.text(j + 0.5, i + 0.25, f'{corr_val:.2f}', ha='center', va='center', color=text_color)
                            ax.text(j + 0.5, i + 0.75, f'p={p_val:.3f}', ha='center', va='center', color=text_color, fontsize=8)
            else:
                # 绘制标准热力图
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax)
            
            # 设置标题
            ax.set_title('Correlation Matrix')
            
            # 保存图表
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存图表
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"相关性热力图已保存到: {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"绘制相关性热力图时发生异常: {str(e)}")
            return None
    
    def run_hypothesis_test(self, 
                          hypothesis: str = 'outperformance',
                          benchmark_returns: Optional[pd.Series] = None,
                          significance_level: float = 0.05) -> Dict[str, Any]:
        """运行假设检验
        
        Args:
            hypothesis: 假设类型 ('outperformance', 'normality', 'stationarity')
            benchmark_returns: 基准收益率数据（用于outperformance假设）
            significance_level: 显著性水平
        
        Returns:
            假设检验结果
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法进行假设检验")
            return {}
        
        try:
            results = {
                'hypothesis': hypothesis,
                'significance_level': significance_level,
                'timestamp': datetime.now().isoformat()
            }
            
            if hypothesis == 'outperformance':
                # 检验策略是否优于基准
                if benchmark_returns is None:
                    logger.warning("进行表现优越假设检验时，需要提供基准收益率数据")
                    return {}
                
                # 计算超额收益率
                excess_returns = self.returns - benchmark_returns.reindex(self.returns.index).fillna(0)
                
                # 执行t检验
                t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
                
                # 确定是否拒绝原假设
                reject_null = p_value < significance_level
                
                results.update({
                    'test_type': 'One-sample t-test',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'reject_null': reject_null,
                    'excess_returns_mean': excess_returns.mean(),
                    'excess_returns_std': excess_returns.std()
                })
                
            elif hypothesis == 'normality':
                # 检验收益率是否服从正态分布
                # 执行Shapiro-Wilk检验
                if len(self.returns) > 5000:
                    # Shapiro-Wilk检验在大样本时计算缓慢，使用D'Agostino's K^2检验
                    stat, p_value = stats.normaltest(self.returns)
                    test_name = "D'Agostino's K^2 test"
                else:
                    stat, p_value = stats.shapiro(self.returns)
                    test_name = "Shapiro-Wilk test"
                
                # 确定是否拒绝原假设
                reject_null = p_value < significance_level
                
                results.update({
                    'test_type': test_name,
                    'test_statistic': stat,
                    'p_value': p_value,
                    'reject_null': reject_null
                })
                
            elif hypothesis == 'stationarity':
                # 检验收益率是否平稳
                # 使用Augmented Dickey-Fuller检验
                try:
                    from statsmodels.tsa.stattools import adfuller
                    
                    # 执行ADF检验
                    result = adfuller(self.returns, autolag='AIC')
                    adf_stat = result[0]
                    p_value = result[1]
                    critical_values = result[4]
                    
                    # 确定是否拒绝原假设
                    reject_null = p_value < significance_level
                    
                    results.update({
                        'test_type': 'Augmented Dickey-Fuller test',
                        'adf_statistic': adf_stat,
                        'p_value': p_value,
                        'critical_values': critical_values,
                        'reject_null': reject_null
                    })
                except ImportError:
                    logger.error("进行平稳性检验需要statsmodels库")
                    return {}
            else:
                logger.warning(f"不支持的假设类型: {hypothesis}")
                return {}
            
            logger.info(f"假设检验完成: {hypothesis}, p值={results.get('p_value', 0):.4f}, 是否拒绝原假设={results.get('reject_null', False)}")
            return results
        except Exception as e:
            logger.error(f"进行假设检验时发生异常: {str(e)}")
            return {}
    
    def calculate_probabilistic_metrics(self) -> Dict[str, Any]:
        """计算概率指标
        
        Returns:
            概率指标字典
        """
        if self.returns is None:
            logger.warning("没有设置收益率数据，无法计算概率指标")
            return {}
        
        try:
            metrics = {}
            
            # 计算收益率分布的分位数
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            quantiles = self.returns.quantile([p/100 for p in percentiles]).values
            
            # 保存分位数
            for i, p in enumerate(percentiles):
                metrics[f'quantile_{p}'] = quantiles[i]
            
            # 计算偏度
            skewness = stats.skew(self.returns)
            metrics['skewness'] = skewness
            
            # 计算峰度
            kurtosis = stats.kurtosis(self.returns)
            metrics['kurtosis'] = kurtosis
            
            # 计算收益率大于0的概率
            prob_positive = len(self.returns[self.returns > 0]) / len(self.returns)
            metrics['probability_positive_return'] = prob_positive
            
            # 计算收益率大于特定阈值的概率
            threshold = 0.01  # 1%的阈值
            prob_above_threshold = len(self.returns[self.returns > threshold]) / len(self.returns)
            metrics[f'probability_above_{threshold:.2%}'] = prob_above_threshold
            
            # 计算连续赢/亏的概率
            if self.trades is not None and 'profit' in self.trades.columns:
                # 计算连续赢/亏的概率
                wins = self.trades['profit'] > 0
                
                # 计算连续赢的次数
                consecutive_wins = 0
                consecutive_wins_list = []
                
                for win in wins:
                    if win:
                        consecutive_wins += 1
                    else:
                        if consecutive_wins > 0:
                            consecutive_wins_list.append(consecutive_wins)
                            consecutive_wins = 0
                
                if consecutive_wins > 0:
                    consecutive_wins_list.append(consecutive_wins)
                
                # 计算连续赢的概率分布
                if consecutive_wins_list:
                    win_distribution = pd.Series(consecutive_wins_list).value_counts(normalize=True)
                    for k, v in win_distribution.items():
                        metrics[f'probability_consecutive_wins_{k}'] = v
            
            logger.info("概率指标计算完成")
            return metrics
        except Exception as e:
            logger.error(f"计算概率指标时发生异常: {str(e)}")
            return {}
    
    def save_analysis_results(self, 
                             results: Dict[str, Any],
                             save_path: str) -> bool:
        """保存分析结果
        
        Args:
            results: 分析结果
            save_path: 保存路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存为JSON文件
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            logger.info(f"分析结果已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存分析结果时发生异常: {str(e)}")
            return False
    
    def load_analysis_results(self, 
                             load_path: str) -> Dict[str, Any]:
        """加载分析结果
        
        Args:
            load_path: 加载路径
        
        Returns:
            分析结果
        """
        try:
            if not os.path.exists(load_path):
                logger.error(f"分析结果文件不存在: {load_path}")
                return {}
            
            # 从文件加载
            with open(load_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"分析结果已从: {load_path} 加载")
            return results
        except Exception as e:
            logger.error(f"加载分析结果时发生异常: {str(e)}")
            return {}

# 模块版本
__version__ = '0.1.0'