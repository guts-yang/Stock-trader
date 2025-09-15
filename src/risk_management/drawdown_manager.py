import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import json

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DrawdownManager:
    """回撤管理器
    负责监控、分析和控制投资组合的回撤风险
    """
    # 回撤类型
    DRAWDOWN_TYPE_ABSOLUTE = 'absolute'
    DRAWDOWN_TYPE_RELATIVE = 'relative'
    
    # 预警级别
    ALERT_LEVEL_LOW = 'low'
    ALERT_LEVEL_MEDIUM = 'medium'
    ALERT_LEVEL_HIGH = 'high'
    ALERT_LEVEL_CRITICAL = 'critical'
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 max_drawdown_limit: float = 0.20,
                 drawdown_recovery_period: int = 30,
                 lookback_period: int = 60,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 risk_free_rate: float = 0.03,
                 benchmark_prices: Optional[pd.DataFrame] = None):
        """初始化回撤管理器
        
        Args:
            config: 配置字典
            max_drawdown_limit: 最大回撤限制（0-1之间的小数）
            drawdown_recovery_period: 期望的回撤恢复期（交易日）
            lookback_period: 回撤分析的回看期（交易日）
            alert_thresholds: 不同级别的回撤预警阈值
            risk_free_rate: 无风险利率
            benchmark_prices: 基准价格数据（用于相对回撤计算）
        """
        self.config = config or {}
        self.max_drawdown_limit = max_drawdown_limit
        self.drawdown_recovery_period = drawdown_recovery_period
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.benchmark_prices = benchmark_prices
        
        # 设置默认预警阈值
        self.alert_thresholds = {
            self.ALERT_LEVEL_LOW: 0.05,
            self.ALERT_LEVEL_MEDIUM: 0.10,
            self.ALERT_LEVEL_HIGH: 0.15,
            self.ALERT_LEVEL_CRITICAL: 0.20
        }
        
        # 覆盖默认阈值
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # 回撤历史数据
        self.drawdown_history = {}
        
        # 当前回撤状态
        self.current_drawdown = None
        self.current_peak_date = None
        self.current_peak_value = None
        self.current_drawdown_days = 0
        self.current_recovery_days = 0
        self.current_alert_level = None
        
        # 初始化日志
        self._init_logger()
        
        logger.info("DrawdownManager 初始化完成")
    
    def _init_logger(self):
        """初始化日志记录器"""
        import os
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"drawdown_management_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def calculate_drawdown(self, 
                          equity_curve: Union[pd.Series, pd.DataFrame],
                          drawdown_type: str = DRAWDOWN_TYPE_ABSOLUTE,
                          column_name: Optional[str] = None) -> pd.DataFrame:
        """计算回撤
        
        Args:
            equity_curve: 权益曲线数据
            drawdown_type: 回撤类型（'absolute' 或 'relative'）
            column_name: 如果是DataFrame，指定要计算的列名
        
        Returns:
            包含回撤数据的DataFrame
        """
        # 处理输入数据
        if isinstance(equity_curve, pd.DataFrame):
            if column_name and column_name in equity_curve.columns:
                ec = equity_curve[column_name].copy()
            else:
                raise ValueError(f"列名 '{column_name}' 不在数据中")
        else:
            ec = equity_curve.copy()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=ec.index)
        result['equity'] = ec
        
        # 计算累积最大权益
        result['cumulative_max'] = ec.cummax()
        
        # 计算绝对回撤
        result['drawdown'] = (ec / result['cumulative_max']) - 1
        result['drawdown_pct'] = result['drawdown'] * 100
        
        # 如果是相对回撤，计算相对于基准的回撤
        if drawdown_type == self.DRAWDOWN_TYPE_RELATIVE and self.benchmark_prices is not None:
            # 确保基准数据与权益曲线有相同的索引
            benchmark_aligned = self.benchmark_prices.reindex(ec.index).dropna()
            
            if not benchmark_aligned.empty:
                # 计算基准累积回报
                benchmark_returns = benchmark_aligned.pct_change().dropna()
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                # 计算投资组合累积回报
                portfolio_returns = ec.pct_change().dropna()
                portfolio_cumulative = (1 + portfolio_returns).cumprod()
                
                # 计算相对累积回报
                relative_cumulative = portfolio_cumulative / benchmark_cumulative.iloc[-1] if len(benchmark_cumulative) > 0 else portfolio_cumulative
                
                # 计算相对回撤
                result['relative_cumulative'] = relative_cumulative
                result['relative_cumulative_max'] = relative_cumulative.cummax()
                result['relative_drawdown'] = (relative_cumulative / result['relative_cumulative_max']) - 1
                result['relative_drawdown_pct'] = result['relative_drawdown'] * 100
        
        # 计算回撤天数
        result['drawdown_days'] = 0
        in_drawdown = False
        days_counter = 0
        
        for i, date in enumerate(result.index):
            if result.loc[date, 'drawdown'] < 0:
                # 处于回撤状态
                if not in_drawdown:
                    in_drawdown = True
                    days_counter = 1
                else:
                    days_counter += 1
                result.loc[date, 'drawdown_days'] = days_counter
            else:
                # 不在回撤状态
                in_drawdown = False
                days_counter = 0
        
        return result
    
    def analyze_drawdown_history(self, 
                                drawdown_data: pd.DataFrame,
                                min_drawdown_pct: float = 5.0) -> Dict[str, Any]:
        """分析回撤历史
        
        Args:
            drawdown_data: 回撤数据
            min_drawdown_pct: 最小回撤百分比（低于此值的回撤将被忽略）
        
        Returns:
            回撤分析结果
        """
        # 提取主要回撤
        drawdown_periods = []
        current_period = None
        
        for i, date in enumerate(drawdown_data.index):
            drawdown_pct = drawdown_data.loc[date, 'drawdown_pct']
            
            if drawdown_pct < -min_drawdown_pct:
                # 开始或继续一个回撤期
                if current_period is None:
                    current_period = {
                        'start_date': date,
                        'peak_date': drawdown_data.loc[date, 'cumulative_max'] == drawdown_data.loc[date, 'equity']
                        if date in drawdown_data.index[:i] else drawdown_data.index[i-1],
                        'peak_value': drawdown_data.loc[date, 'cumulative_max'],
                        'min_value': drawdown_data.loc[date, 'equity'],
                        'trough_date': date,
                        'max_drawdown_pct': drawdown_pct,
                        'length_days': 1
                    }
                else:
                    # 更新当前回撤期
                    if drawdown_pct < current_period['max_drawdown_pct']:
                        current_period['max_drawdown_pct'] = drawdown_pct
                        current_period['min_value'] = drawdown_data.loc[date, 'equity']
                        current_period['trough_date'] = date
                    current_period['length_days'] += 1
            else:
                # 结束一个回撤期
                if current_period is not None:
                    # 计算恢复期
                    recovery_start = current_period['trough_date']
                    recovery_end = None
                    
                    # 寻找恢复期（回到峰值）
                    for j in range(i, len(drawdown_data.index)):
                        recovery_date = drawdown_data.index[j]
                        if drawdown_data.loc[recovery_date, 'equity'] >= current_period['peak_value']:
                            recovery_end = recovery_date
                            break
                    
                    # 添加恢复期信息
                    current_period['recovery_period_days'] = (recovery_end - recovery_start).days if recovery_end else None
                    current_period['fully_recovered'] = recovery_end is not None
                    
                    # 添加到回撤期列表
                    drawdown_periods.append(current_period)
                    current_period = None
        
        # 计算统计信息
        if drawdown_periods:
            max_drawdown = min([period['max_drawdown_pct'] for period in drawdown_periods])
            avg_drawdown = np.mean([period['max_drawdown_pct'] for period in drawdown_periods])
            median_drawdown = np.median([period['max_drawdown_pct'] for period in drawdown_periods])
            avg_length = np.mean([period['length_days'] for period in drawdown_periods])
            avg_recovery = np.mean([period['recovery_period_days'] for period in drawdown_periods if period['fully_recovered']]) if any([period['fully_recovered'] for period in drawdown_periods]) else None
            
            # 计算回撤频率（每年发生的次数）
            total_days = (drawdown_data.index[-1] - drawdown_data.index[0]).days
            annualized_frequency = len(drawdown_periods) / (total_days / 365.25) if total_days > 0 else 0
            
            # 计算最长回撤期
            longest_drawdown = max([period['length_days'] for period in drawdown_periods])
            
            # 计算最大回撤持续时间
            max_duration_period = max(drawdown_periods, key=lambda x: x['length_days'])
        else:
            max_drawdown = 0
            avg_drawdown = 0
            median_drawdown = 0
            avg_length = 0
            avg_recovery = 0
            annualized_frequency = 0
            longest_drawdown = 0
            max_duration_period = None
        
        # 构建分析结果
        analysis_result = {
            'total_drawdown_periods': len(drawdown_periods),
            'max_drawdown_pct': max_drawdown,
            'avg_drawdown_pct': avg_drawdown,
            'median_drawdown_pct': median_drawdown,
            'avg_length_days': avg_length,
            'avg_recovery_days': avg_recovery,
            'annualized_frequency': annualized_frequency,
            'longest_drawdown_days': longest_drawdown,
            'max_duration_period': max_duration_period,
            'drawdown_periods': drawdown_periods
        }
        
        logger.info(f"回撤历史分析完成，共识别 {len(drawdown_periods)} 个主要回撤期")
        
        return analysis_result
    
    def update_drawdown_status(self, 
                              current_equity: float,
                              current_date: datetime,
                              save_history: bool = True) -> Dict[str, Any]:
        """更新当前回撤状态
        
        Args:
            current_equity: 当前权益值
            current_date: 当前日期
            save_history: 是否保存历史记录
        
        Returns:
            更新后的回撤状态
        """
        # 初始化峰值信息
        if self.current_peak_value is None or current_equity > self.current_peak_value:
            # 达到新峰值
            self.current_peak_value = current_equity
            self.current_peak_date = current_date
            self.current_drawdown = 0.0
            self.current_drawdown_days = 0
            self.current_recovery_days = 0
            self.current_alert_level = None
        else:
            # 处于回撤状态
            self.current_drawdown = (current_equity / self.current_peak_value) - 1
            self.current_drawdown_days += 1
            
            # 计算恢复进度
            peak_to_current = current_equity - self.current_peak_value
            peak_to_trough = current_equity - self.current_peak_value  # 简化版，实际应跟踪最低点
            recovery_progress = peak_to_current / peak_to_trough if peak_to_trough != 0 else 0
            
            # 根据恢复进度更新恢复天数
            if recovery_progress > 0:
                self.current_recovery_days += 1
            else:
                self.current_recovery_days = 0
            
            # 确定预警级别
            self.current_alert_level = self._determine_alert_level(abs(self.current_drawdown))
        
        # 构建当前状态
        current_status = {
            'date': current_date,
            'current_equity': current_equity,
            'peak_value': self.current_peak_value,
            'peak_date': self.current_peak_date,
            'drawdown': self.current_drawdown,
            'drawdown_pct': self.current_drawdown * 100,
            'drawdown_days': self.current_drawdown_days,
            'recovery_days': self.current_recovery_days,
            'alert_level': self.current_alert_level,
            'is_at_peak': self.current_drawdown == 0.0,
            'is_recovering': self.current_recovery_days > 0,
            'is_above_max_limit': abs(self.current_drawdown) > self.max_drawdown_limit
        }
        
        # 保存历史记录
        if save_history:
            date_key = current_date.strftime('%Y-%m-%d')
            self.drawdown_history[date_key] = current_status
        
        # 如果超过最大回撤限制，记录日志
        if current_status['is_above_max_limit']:
            logger.warning(f"回撤超过限制！当前回撤: {current_status['drawdown_pct']:.2f}%，最大限制: {self.max_drawdown_limit*100:.2f}%")
        
        return current_status
    
    def _determine_alert_level(self, drawdown_pct: float) -> Optional[str]:
        """确定回撤预警级别
        
        Args:
            drawdown_pct: 回撤百分比（绝对值）
        
        Returns:
            预警级别
        """
        if drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_CRITICAL]:
            return self.ALERT_LEVEL_CRITICAL
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_HIGH]:
            return self.ALERT_LEVEL_HIGH
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_MEDIUM]:
            return self.ALERT_LEVEL_MEDIUM
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_LOW]:
            return self.ALERT_LEVEL_LOW
        else:
            return None
    
    def generate_alert(self, 
                      drawdown_status: Dict[str, Any],
                      previous_status: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """生成回撤预警
        
        Args:
            drawdown_status: 当前回撤状态
            previous_status: 之前的回撤状态
        
        Returns:
            预警信息（如果需要预警）
        """
        # 检查是否需要预警
        if drawdown_status['alert_level'] is None:
            return None
        
        # 检查是否是新的预警级别
        if previous_status and previous_status['alert_level'] == drawdown_status['alert_level']:
            # 同一预警级别，可能不需要重复预警
            # 但如果是严重级别，可以考虑重复预警
            if drawdown_status['alert_level'] != self.ALERT_LEVEL_CRITICAL:
                return None
        
        # 构建预警信息
        alert_message = self._get_alert_message(drawdown_status['alert_level'], drawdown_status)
        
        alert = {
            'timestamp': datetime.now(),
            'date': drawdown_status['date'],
            'alert_level': drawdown_status['alert_level'],
            'drawdown_pct': drawdown_status['drawdown_pct'],
            'drawdown_days': drawdown_status['drawdown_days'],
            'peak_date': drawdown_status['peak_date'],
            'peak_value': drawdown_status['peak_value'],
            'current_equity': drawdown_status['current_equity'],
            'message': alert_message,
            'recommended_actions': self._get_recommended_actions(drawdown_status['alert_level'])
        }
        
        # 根据预警级别记录日志
        if drawdown_status['alert_level'] == self.ALERT_LEVEL_CRITICAL:
            logger.critical(f"严重回撤预警: {alert_message}")
        elif drawdown_status['alert_level'] == self.ALERT_LEVEL_HIGH:
            logger.error(f"高回撤预警: {alert_message}")
        elif drawdown_status['alert_level'] == self.ALERT_LEVEL_MEDIUM:
            logger.warning(f"中回撤预警: {alert_message}")
        elif drawdown_status['alert_level'] == self.ALERT_LEVEL_LOW:
            logger.info(f"低回撤预警: {alert_message}")
        
        return alert
    
    def _get_alert_message(self, alert_level: str, drawdown_status: Dict[str, Any]) -> str:
        """获取预警消息
        
        Args:
            alert_level: 预警级别
            drawdown_status: 回撤状态
        
        Returns:
            预警消息字符串
        """
        level_names = {
            self.ALERT_LEVEL_LOW: '低级别',
            self.ALERT_LEVEL_MEDIUM: '中级别', 
            self.ALERT_LEVEL_HIGH: '高级别',
            self.ALERT_LEVEL_CRITICAL: '严重'
        }
        
        message = f"[{level_names.get(alert_level, '未知')}回撤预警] 日期: {drawdown_status['date'].strftime('%Y-%m-%d')}, "
        message += f"当前回撤: {abs(drawdown_status['drawdown_pct']):.2f}%, "
        message += f"持续天数: {drawdown_status['drawdown_days']}, "
        message += f"峰值日期: {drawdown_status['peak_date'].strftime('%Y-%m-%d')}"
        
        # 如果超过最大限制，添加额外警告
        if drawdown_status['is_above_max_limit']:
            message += f"\n警告: 回撤已超过最大限制 {self.max_drawdown_limit*100:.2f}%！"
        
        return message
    
    def _get_recommended_actions(self, alert_level: str) -> List[str]:
        """获取推荐的操作
        
        Args:
            alert_level: 预警级别
        
        Returns:
            推荐操作列表
        """
        if alert_level == self.ALERT_LEVEL_CRITICAL:
            return [
                "立即减少风险敞口",
                "考虑部分或全部平仓",
                "重新评估交易策略",
                "检查是否存在系统性风险",
                "考虑暂时停止交易"
            ]
        elif alert_level == self.ALERT_LEVEL_HIGH:
            return [
                "减少风险敞口",
                "提高仓位监控频率",
                "分析回撤原因",
                "考虑对冲策略"
            ]
        elif alert_level == self.ALERT_LEVEL_MEDIUM:
            return [
                "密切关注市场动向",
                "审查当前持仓",
                "准备风险控制措施",
                "重新评估止盈止损位"
            ]
        elif alert_level == self.ALERT_LEVEL_LOW:
            return [
                "保持监控",
                "注意市场变化",
                "确保风险管理措施正常运行"
            ]
        else:
            return []
    
    def calculate_recovery_metrics(self, 
                                  equity_curve: pd.Series,
                                  drawdown_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """计算恢复指标
        
        Args:
            equity_curve: 权益曲线
            drawdown_data: 回撤数据（可选，如果未提供则计算）
        
        Returns:
            恢复指标字典
        """
        # 如果未提供回撤数据，计算它
        if drawdown_data is None:
            drawdown_data = self.calculate_drawdown(equity_curve)
        
        # 分析回撤历史
        drawdown_analysis = self.analyze_drawdown_history(drawdown_data)
        
        # 计算恢复指标
        metrics = {
            'avg_recovery_period_days': drawdown_analysis.get('avg_recovery_days', 0),
            'recovery_rate': drawdown_analysis.get('total_drawdown_periods', 0) / len(equity_curve) if len(equity_curve) > 0 else 0,
            'recovery_efficiency': 0.0,  # 简化版，实际应计算恢复效率
            'recovery_success_rate': len([p for p in drawdown_analysis.get('drawdown_periods', []) if p.get('fully_recovered', False)]) / \
                                    max(1, drawdown_analysis.get('total_drawdown_periods', 1)),
            'time_under_water_pct': sum(drawdown_data['drawdown_days'] > 0) / len(drawdown_data) * 100 if len(drawdown_data) > 0 else 0,
            'avg_time_under_water_days': np.mean(drawdown_data['drawdown_days'][drawdown_data['drawdown_days'] > 0]) if sum(drawdown_data['drawdown_days'] > 0) > 0 else 0
        }
        
        # 计算Calmar比率（年化收益率与最大回撤的比值）
        annualized_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
        max_drawdown = abs(drawdown_analysis.get('max_drawdown_pct', 0) / 100)
        metrics['calmar_ratio'] = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # 计算Sterling比率（年化收益率与平均回撤的比值）
        avg_drawdown = abs(drawdown_analysis.get('avg_drawdown_pct', 0) / 100)
        metrics['sterling_ratio'] = annualized_return / avg_drawdown if avg_drawdown > 0 else float('inf')
        
        return metrics
    
    def get_risk_reduction_recommendations(self, 
                                         current_drawdown: float,
                                         portfolio_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取风险降低建议
        
        Args:
            current_drawdown: 当前回撤
            portfolio_positions: 当前投资组合持仓
        
        Returns:
            风险降低建议列表
        """
        recommendations = []
        
        # 根据回撤大小确定建议力度
        drawdown_pct = abs(current_drawdown) * 100
        
        if drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_CRITICAL]:
            # 严重回撤：建议大幅降低风险
            recommendations.append({
                'type': 'reduce_exposure',
                'description': '大幅降低整体风险敞口',
                'severity': 'high',
                'suggested_reduction_pct': 50
            })
            
            # 建议平掉部分亏损最大的持仓
            if portfolio_positions:
                # 假设有持仓的盈亏信息
                # 实际应用中，应该根据真实的持仓盈亏数据排序
                recommendations.append({
                    'type': 'close_losing_positions',
                    'description': '平掉亏损最大的几个持仓',
                    'severity': 'high',
                    'suggested_count': min(3, len(portfolio_positions))
                })
            
            # 建议增加流动性
            recommendations.append({
                'type': 'increase_liquidity',
                'description': '增加现金或高流动性资产比例',
                'severity': 'high',
                'suggested_liquidity_pct': 30
            })
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_HIGH]:
            # 高回撤：建议显著降低风险
            recommendations.append({
                'type': 'reduce_exposure',
                'description': '显著降低整体风险敞口',
                'severity': 'medium',
                'suggested_reduction_pct': 30
            })
            
            # 建议分散风险
            recommendations.append({
                'type': 'diversify',
                'description': '增加投资组合多样性，减少集中风险',
                'severity': 'medium'
            })
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_MEDIUM]:
            # 中回撤：建议适度降低风险
            recommendations.append({
                'type': 'reduce_exposure',
                'description': '适度降低整体风险敞口',
                'severity': 'low',
                'suggested_reduction_pct': 15
            })
            
            # 建议调整止损位
            recommendations.append({
                'type': 'adjust_stop_loss',
                'description': '收紧止损位，控制单笔交易风险',
                'severity': 'low'
            })
        elif drawdown_pct >= self.alert_thresholds[self.ALERT_LEVEL_LOW]:
            # 低回撤：建议保持监控并小幅调整
            recommendations.append({
                'type': 'monitor_closely',
                'description': '密切监控市场动向和投资组合表现',
                'severity': 'low'
            })
            
            # 建议小幅调整仓位
            recommendations.append({
                'type': 'adjust_positions',
                'description': '小幅调整持仓比例，优化风险回报',
                'severity': 'low'
            })
        
        return recommendations
    
    def simulate_recovery_strategy(self, 
                                 equity_curve: pd.Series,
                                 drawdown_data: Optional[pd.DataFrame] = None,
                                 strategy_type: str = 'fixed_reduction') -> pd.DataFrame:
        """模拟不同的恢复策略
        
        Args:
            equity_curve: 权益曲线
            drawdown_data: 回撤数据（可选，如果未提供则计算）
            strategy_type: 恢复策略类型
        
        Returns:
            模拟结果DataFrame
        """
        # 如果未提供回撤数据，计算它
        if drawdown_data is None:
            drawdown_data = self.calculate_drawdown(equity_curve)
        
        # 创建模拟结果DataFrame
        simulation_result = drawdown_data[['equity', 'cumulative_max', 'drawdown']].copy()
        simulation_result['simulated_equity'] = simulation_result['equity'].copy()
        
        # 根据策略类型进行模拟
        if strategy_type == 'fixed_reduction':
            # 固定比例降低风险
            reduction_pct = 0.10  # 当回撤超过阈值时，降低10%的风险敞口
            threshold = self.alert_thresholds[self.ALERT_LEVEL_MEDIUM]  # 使用中级别阈值
            
            in_recovery_mode = False
            recovery_multiplier = 1.0
            
            for i, date in enumerate(simulation_result.index):
                drawdown = abs(simulation_result.loc[date, 'drawdown'])
                
                if drawdown > threshold:
                    if not in_recovery_mode:
                        # 进入恢复模式，降低风险敞口
                        in_recovery_mode = True
                        recovery_multiplier = 1.0 - reduction_pct
                        logger.info(f"在 {date} 进入恢复模式，风险敞口降低 {reduction_pct*100:.1f}%")
                else:
                    if in_recovery_mode:
                        # 退出恢复模式，恢复风险敞口
                        in_recovery_mode = False
                        recovery_multiplier = 1.0
                        logger.info(f"在 {date} 退出恢复模式，恢复正常风险敞口")
                
                # 应用恢复策略
                if i > 0 and in_recovery_mode:
                    # 计算原始日收益率
                    original_return = simulation_result.loc[date, 'equity'] / simulation_result.loc[simulation_result.index[i-1], 'equity'] - 1
                    
                    # 应用风险调整后的收益率
                    adjusted_return = original_return * recovery_multiplier
                    
                    # 更新模拟权益
                    simulation_result.loc[date, 'simulated_equity'] = simulation_result.loc[simulation_result.index[i-1], 'simulated_equity'] * (1 + adjusted_return)
        elif strategy_type == 'dynamic_reduction':
            # 动态比例降低风险（根据回撤大小调整）
            for i, date in enumerate(simulation_result.index):
                drawdown = abs(simulation_result.loc[date, 'drawdown'])
                
                # 根据回撤大小计算风险降低比例
                if drawdown < self.alert_thresholds[self.ALERT_LEVEL_LOW]:
                    reduction_pct = 0.0
                elif drawdown < self.alert_thresholds[self.ALERT_LEVEL_MEDIUM]:
                    reduction_pct = 0.05
                elif drawdown < self.alert_thresholds[self.ALERT_LEVEL_HIGH]:
                    reduction_pct = 0.15
                else:
                    reduction_pct = 0.30
                
                recovery_multiplier = 1.0 - reduction_pct
                
                # 应用恢复策略
                if i > 0:
                    # 计算原始日收益率
                    original_return = simulation_result.loc[date, 'equity'] / simulation_result.loc[simulation_result.index[i-1], 'equity'] - 1
                    
                    # 应用风险调整后的收益率
                    adjusted_return = original_return * recovery_multiplier
                    
                    # 更新模拟权益
                    simulation_result.loc[date, 'simulated_equity'] = simulation_result.loc[simulation_result.index[i-1], 'simulated_equity'] * (1 + adjusted_return)
        elif strategy_type == 'stop_trading':
            # 当回撤超过阈值时停止交易
            threshold = self.alert_thresholds[self.ALERT_LEVEL_HIGH]  # 使用高级别阈值
            
            stop_trading_date = None
            
            for i, date in enumerate(simulation_result.index):
                drawdown = abs(simulation_result.loc[date, 'drawdown'])
                
                if drawdown > threshold and stop_trading_date is None:
                    # 超过阈值，停止交易
                    stop_trading_date = date
                    logger.info(f"在 {date} 停止交易，回撤 {drawdown*100:.1f}% 超过阈值 {threshold*100:.1f}%")
                elif stop_trading_date is not None:
                    # 检查是否恢复交易
                    days_since_stop = (date - stop_trading_date).days
                    if days_since_stop >= self.drawdown_recovery_period:
                        # 经过恢复期后，恢复交易
                        stop_trading_date = None
                        logger.info(f"在 {date} 恢复交易，已停止 {days_since_stop} 天")
                
                # 应用恢复策略
                if i > 0 and stop_trading_date is not None:
                    # 停止交易时，权益保持不变（简化假设）
                    simulation_result.loc[date, 'simulated_equity'] = simulation_result.loc[simulation_result.index[i-1], 'simulated_equity']
        
        # 计算模拟回撤
        simulation_result['simulated_cumulative_max'] = simulation_result['simulated_equity'].cummax()
        simulation_result['simulated_drawdown'] = (simulation_result['simulated_equity'] / simulation_result['simulated_cumulative_max']) - 1
        
        logger.info(f"恢复策略模拟完成，策略类型: {strategy_type}")
        
        return simulation_result
    
    def generate_drawdown_report(self, 
                                equity_curve: pd.Series,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成回撤报告
        
        Args:
            equity_curve: 权益曲线
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径
        
        Returns:
            报告数据字典
        """
        # 过滤日期范围
        if start_date is not None:
            equity_curve = equity_curve[equity_curve.index >= start_date]
        if end_date is not None:
            equity_curve = equity_curve[equity_curve.index <= end_date]
        
        # 计算回撤数据
        drawdown_data = self.calculate_drawdown(equity_curve)
        
        # 分析回撤历史
        drawdown_analysis = self.analyze_drawdown_history(drawdown_data)
        
        # 计算恢复指标
        recovery_metrics = self.calculate_recovery_metrics(equity_curve, drawdown_data)
        
        # 构建报告
        report = {
            'report_date': datetime.now(),
            'time_period': {
                'start_date': equity_curve.index[0],
                'end_date': equity_curve.index[-1],
                'days': len(equity_curve)
            },
            'performance': {
                'start_equity': equity_curve.iloc[0],
                'end_equity': equity_curve.iloc[-1],
                'total_return_pct': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100,
                'annualized_return_pct': ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1) * 100
            },
            'drawdown_summary': {
                'max_drawdown_pct': drawdown_analysis.get('max_drawdown_pct', 0),
                'avg_drawdown_pct': drawdown_analysis.get('avg_drawdown_pct', 0),
                'median_drawdown_pct': drawdown_analysis.get('median_drawdown_pct', 0),
                'total_drawdown_periods': drawdown_analysis.get('total_drawdown_periods', 0),
                'avg_drawdown_length_days': drawdown_analysis.get('avg_length_days', 0),
                'annualized_frequency': drawdown_analysis.get('annualized_frequency', 0),
                'time_under_water_pct': recovery_metrics.get('time_under_water_pct', 0)
            },
            'recovery_metrics': recovery_metrics,
            'key_ratios': {
                'calmar_ratio': recovery_metrics.get('calmar_ratio', 0),
                'sterling_ratio': recovery_metrics.get('sterling_ratio', 0),
                'recovery_success_rate': recovery_metrics.get('recovery_success_rate', 0)
            },
            'major_drawdown_periods': drawdown_analysis.get('drawdown_periods', []),
            'current_status': self.update_drawdown_status(equity_curve.iloc[-1], equity_curve.index[-1], save_history=False)
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
                    report_serializable['report_date'] = report_serializable['report_date'].strftime('%Y-%m-%d %H:%M:%S')
                    report_serializable['time_period']['start_date'] = report_serializable['time_period']['start_date'].strftime('%Y-%m-%d')
                    report_serializable['time_period']['end_date'] = report_serializable['time_period']['end_date'].strftime('%Y-%m-%d')
                    
                    # 转换主要回撤期的日期
                    for period in report_serializable['major_drawdown_periods']:
                        period['start_date'] = period['start_date'].strftime('%Y-%m-%d')
                        period['peak_date'] = period['peak_date'].strftime('%Y-%m-%d')
                        period['trough_date'] = period['trough_date'].strftime('%Y-%m-%d')
                        if period.get('recovery_end_date'):
                            period['recovery_end_date'] = period['recovery_end_date'].strftime('%Y-%m-%d')
                    
                    # 转换当前状态的日期
                    report_serializable['current_status']['date'] = report_serializable['current_status']['date'].strftime('%Y-%m-%d')
                    report_serializable['current_status']['peak_date'] = report_serializable['current_status']['peak_date'].strftime('%Y-%m-%d')
                    
                    json.dump(report_serializable, f, indent=2, ensure_ascii=False)
                
                logger.info(f"回撤报告已保存到: {output_file}")
            except Exception as e:
                logger.error(f"保存回撤报告时发生异常: {str(e)}")
        
        logger.info("回撤报告生成完成")
        
        return report
    
    def save_history(self, file_path: Optional[str] = None) -> str:
        """保存回撤历史记录
        
        Args:
            file_path: 保存路径
        
        Returns:
            实际保存路径
        """
        try:
            import os
            logger.info("保存回撤历史记录")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if file_path is None:
                history_dir = os.path.join(self.config.get('history_dir', './history'), 'drawdowns')
                if not os.path.exists(history_dir):
                    os.makedirs(history_dir)
                
                file_path = os.path.join(history_dir, f"drawdown_history_{timestamp}.json")
            
            # 准备保存数据（转换datetime对象）
            save_data = {}
            for date_key, status in self.drawdown_history.items():
                status_serializable = status.copy()
                if 'date' in status_serializable and isinstance(status_serializable['date'], datetime):
                    status_serializable['date'] = status_serializable['date'].strftime('%Y-%m-%d')
                if 'peak_date' in status_serializable and isinstance(status_serializable['peak_date'], datetime):
                    status_serializable['peak_date'] = status_serializable['peak_date'].strftime('%Y-%m-%d')
                save_data[date_key] = status_serializable
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"回撤历史记录已保存到: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"保存回撤历史记录时发生异常: {str(e)}")
            raise
    
    def load_history(self, file_path: str) -> bool:
        """加载回撤历史记录
        
        Args:
            file_path: 历史记录文件路径
        
        Returns:
            是否加载成功
        """
        try:
            import os
            logger.info(f"加载回撤历史记录: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"回撤历史记录文件不存在: {file_path}")
                return False
            
            # 读取历史记录
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            # 恢复datetime对象
            for date_key, status in loaded_history.items():
                if 'date' in status:
                    try:
                        status['date'] = datetime.strptime(status['date'], '%Y-%m-%d')
                    except ValueError:
                        pass  # 如果转换失败，保持原格式
                if 'peak_date' in status:
                    try:
                        status['peak_date'] = datetime.strptime(status['peak_date'], '%Y-%m-%d')
                    except ValueError:
                        pass  # 如果转换失败，保持原格式
            
            # 合并历史记录
            self.drawdown_history.update(loaded_history)
            
            logger.info(f"成功加载 {len(loaded_history)} 条回撤历史记录")
            
            return True
        except Exception as e:
            logger.error(f"加载回撤历史记录时发生异常: {str(e)}")
            return False

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['DrawdownManager']