import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import json

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RiskManager:
    """风险管理核心类
    负责监控、控制和报告交易系统中的各类风险
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 initial_capital: float = 1000000.0,
                 risk_limit: float = 0.02,
                 max_position_size: float = 0.1,
                 max_drawdown: float = 0.2,
                 stop_loss_ratio: float = 0.05,
                 take_profit_ratio: float = 0.1,
                 risk_free_rate: float = 0.03,
                 portfolio_vol_target: float = 0.15,
                 sector_exposure_limit: float = 0.3,
                 leverage_limit: float = 1.5,
                 trade_limit: int = 50):
        """初始化风险管理器
        
        Args:
            config: 配置字典
            initial_capital: 初始资金
            risk_limit: 单笔交易最大风险比例
            max_position_size: 单个持仓最大占比
            max_drawdown: 最大回撤限制
            stop_loss_ratio: 止损比例
            take_profit_ratio: 止盈比例
            risk_free_rate: 无风险利率
            portfolio_vol_target: 组合波动率目标
            sector_exposure_limit: 行业暴露限制
            leverage_limit: 杠杆限制
            trade_limit: 每日交易限制
        """
        self.config = config or {}
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limit = risk_limit
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.risk_free_rate = risk_free_rate
        self.portfolio_vol_target = portfolio_vol_target
        self.sector_exposure_limit = sector_exposure_limit
        self.leverage_limit = leverage_limit
        self.trade_limit = trade_limit
        
        # 风险记录
        self.risk_records = []
        self.trade_count = 0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown_recorded = 0.0
        self.drawdown_history = []
        
        # 行业暴露跟踪
        self.sector_exposures = {}
        
        # 初始化日志
        self._init_logger()
        
        # 创建必要的目录
        self._create_directories()
        
        logger.info("RiskManager 初始化完成")
    
    def _init_logger(self):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"risk_management_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
    
    def _create_directories(self):
        """创建必要的目录"""
        # 结果目录
        results_dir = self.config.get('results_dir', './results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 风险报告目录
        risk_reports_dir = os.path.join(results_dir, 'risk_reports')
        if not os.path.exists(risk_reports_dir):
            os.makedirs(risk_reports_dir)
        
        # 图表目录
        figures_dir = self.config.get('figures_dir', './figures')
        risk_figures_dir = os.path.join(figures_dir, 'risk')
        if not os.path.exists(risk_figures_dir):
            os.makedirs(risk_figures_dir)
    
    def update_capital(self, new_capital: float) -> None:
        """更新当前资金
        
        Args:
            new_capital: 新的资金金额
        """
        self.current_capital = new_capital
        
        # 更新最大回撤
        if new_capital > self.peak_equity:
            self.peak_equity = new_capital
        
        # 计算当前回撤
        self.current_drawdown = 1.0 - (new_capital / self.peak_equity)
        self.drawdown_history.append({
            'date': datetime.now(),
            'equity': new_capital,
            'peak_equity': self.peak_equity,
            'drawdown': self.current_drawdown
        })
        
        # 更新最大回撤记录
        if self.current_drawdown > self.max_drawdown_recorded:
            self.max_drawdown_recorded = self.current_drawdown
            logger.warning(f"新的最大回撤记录: {self.max_drawdown_recorded:.2%}")
        
        # 检查每日交易计数器是否需要重置
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = current_date
    
    def check_risk_limits(self, 
                         position: Dict[str, Any],
                         portfolio: Dict[str, Any] = None) -> Tuple[bool, str]:
        """检查交易是否符合风险限制
        
        Args:
            position: 头寸信息
            portfolio: 组合信息
        
        Returns:
            Tuple[是否通过, 拒绝原因]
        """
        # 检查每日交易限制
        if self.daily_trades >= self.trade_limit:
            reason = f"每日交易限制已达: {self.daily_trades}/{self.trade_limit}"
            logger.warning(reason)
            return False, reason
        
        # 检查单笔交易风险
        trade_risk = position.get('risk_amount', 0.0) / self.current_capital
        if trade_risk > self.risk_limit:
            reason = f"单笔交易风险过大: {trade_risk:.2%} > {self.risk_limit:.2%}"
            logger.warning(reason)
            return False, reason
        
        # 检查单个持仓最大占比
        if portfolio:
            # 计算新持仓后的组合权重
            new_value = position.get('value', 0.0)
            total_value = self.current_capital
            for p in portfolio.values():
                total_value += p.get('value', 0.0)
            
            new_weight = new_value / total_value
            if new_weight > self.max_position_size:
                reason = f"单个持仓权重过大: {new_weight:.2%} > {self.max_position_size:.2%}"
                logger.warning(reason)
                return False, reason
        
        # 检查行业暴露
        sector = position.get('sector', None)
        if sector and portfolio:
            # 计算行业暴露
            sector_value = new_value
            for p in portfolio.values():
                if p.get('sector') == sector:
                    sector_value += p.get('value', 0.0)
            
            sector_exposure = sector_value / self.current_capital
            if sector_exposure > self.sector_exposure_limit:
                reason = f"行业暴露过大: {sector_exposure:.2%} > {self.sector_exposure_limit:.2%}" 
                logger.warning(reason)
                return False, reason
        
        # 检查止损/止盈
        entry_price = position.get('entry_price', 0.0)
        current_price = position.get('current_price', entry_price)
        if entry_price > 0:
            # 计算价格变动百分比
            price_change = (current_price - entry_price) / entry_price
            
            # 检查止损
            if price_change < -self.stop_loss_ratio:
                reason = f"触发止损: {price_change:.2%} < -{self.stop_loss_ratio:.2%}"
                logger.warning(reason)
                return False, reason
            
            # 检查止盈
            if 'take_profit' in position and position['take_profit'] and price_change > self.take_profit_ratio:
                reason = f"触发止盈: {price_change:.2%} > {self.take_profit_ratio:.2%}"
                logger.info(reason)
                return False, reason
        
        # 检查杠杆限制
        if portfolio:
            total_exposure = sum(p.get('value', 0.0) for p in portfolio.values()) + position.get('value', 0.0)
            leverage = total_exposure / self.current_capital
            if leverage > self.leverage_limit:
                reason = f"杠杆过高: {leverage:.2f}x > {self.leverage_limit:.2f}x"
                logger.warning(reason)
                return False, reason
        
        # 检查最大回撤
        if self.current_drawdown > self.max_drawdown:
            reason = f"超过最大回撤限制: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}"
            logger.warning(reason)
            return False, reason
        
        # 所有检查通过
        return True, "风险检查通过"
    
    def calculate_position_size(self, 
                               ticker: str,
                               price: float,
                               risk_per_trade: float = None,
                               stop_loss_price: float = None,
                               max_position_value: float = None) -> int:
        """计算合适的头寸大小
        
        Args:
            ticker: 股票代码
            price: 当前价格
            risk_per_trade: 单笔交易风险金额
            stop_loss_price: 止损价格
            max_position_value: 最大持仓价值
        
        Returns:
            建议的持仓数量
        """
        # 如果没有指定风险金额，使用默认风险比例
        if risk_per_trade is None:
            risk_per_trade = self.current_capital * self.risk_limit
        
        # 计算每单位风险
        if stop_loss_price and stop_loss_price < price:
            risk_per_share = price - stop_loss_price
        else:
            # 如果没有指定止损价格，使用波动率百分比作为风险估计
            risk_per_share = price * self.stop_loss_ratio
        
        # 计算可以购买的数量
        shares = int(risk_per_trade / risk_per_share)
        
        # 计算持仓价值
        position_value = shares * price
        
        # 应用单个持仓最大占比限制
        max_position_value_allowed = self.current_capital * self.max_position_size
        if max_position_value:
            max_position_value_allowed = min(max_position_value_allowed, max_position_value)
        
        if position_value > max_position_value_allowed:
            # 调整持仓数量以符合最大持仓限制
            shares = int(max_position_value_allowed / price)
            position_value = shares * price
            logger.info(f"调整 {ticker} 持仓大小以符合最大持仓限制: {shares} 股，价值 {position_value:.2f}")
        
        # 确保至少交易1股
        shares = max(1, shares)
        
        logger.info(f"为 {ticker} 计算的持仓大小: {shares} 股，价格 {price:.2f}，价值 {shares * price:.2f}")
        
        return shares
    
    def calculate_portfolio_risk(self, 
                                portfolio: Dict[str, Any],
                                returns_data: pd.DataFrame) -> Dict[str, float]:
        """计算组合风险指标
        
        Args:
            portfolio: 组合信息
            returns_data: 收益率数据
        
        Returns:
            风险指标字典
        """
        risk_metrics = {}
        
        # 计算组合收益率
        portfolio_returns = []
        total_value = sum(p.get('value', 0.0) for p in portfolio.values())
        
        for date, row in returns_data.iterrows():
            daily_return = 0.0
            for ticker, position in portfolio.items():
                if ticker in row:
                    weight = position.get('value', 0.0) / total_value
                    daily_return += row[ticker] * weight
            portfolio_returns.append(daily_return)
        
        # 计算风险指标
        portfolio_returns = np.array(portfolio_returns)
        
        # 波动率
        risk_metrics['volatility'] = np.std(portfolio_returns) * np.sqrt(252)  # 年化波动率
        
        # 最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        risk_metrics['max_drawdown'] = drawdown.min()
        
        # Sharpe比率
        if risk_metrics['volatility'] > 0:
            risk_metrics['sharpe_ratio'] = (np.mean(portfolio_returns) * 252 - self.risk_free_rate) / risk_metrics['volatility']
        else:
            risk_metrics['sharpe_ratio'] = np.nan
        
        # Sortino比率（只考虑下行风险）
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns) * np.sqrt(252)
            if downside_vol > 0:
                risk_metrics['sortino_ratio'] = (np.mean(portfolio_returns) * 252 - self.risk_free_rate) / downside_vol
            else:
                risk_metrics['sortino_ratio'] = np.nan
        else:
            risk_metrics['sortino_ratio'] = np.nan
        
        # Calmar比率
        if risk_metrics['max_drawdown'] < 0:
            risk_metrics['calmar_ratio'] = (np.mean(portfolio_returns) * 252) / abs(risk_metrics['max_drawdown'])
        else:
            risk_metrics['calmar_ratio'] = np.nan
        
        # Omega比率
        threshold = self.risk_free_rate / 252  # 日度无风险利率
        gains = portfolio_returns[portfolio_returns > threshold]
        losses = threshold - portfolio_returns[portfolio_returns <= threshold]
        
        if len(losses) > 0 and np.sum(losses) > 0:
            risk_metrics['omega_ratio'] = np.sum(gains) / np.sum(losses)
        else:
            risk_metrics['omega_ratio'] = np.nan
        
        logger.info(f"计算的组合风险指标: {risk_metrics}")
        
        return risk_metrics
    
    def generate_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险预警
        
        Args:
            alert_type: 预警类型
            details: 预警详情
        
        Returns:
            预警信息字典
        """
        alert = {
            'timestamp': datetime.now(),
            'alert_type': alert_type,
            'details': details,
            'severity': self._determine_severity(alert_type, details)
        }
        
        # 记录预警
        self.risk_records.append(alert)
        
        # 记录日志
        severity_str = alert['severity'].upper()
        logger.info(f"风险预警 ({severity_str}): {alert_type} - {details}")
        
        # 保存预警到文件
        self._save_risk_alert(alert)
        
        return alert
    
    def _determine_severity(self, alert_type: str, details: Dict[str, Any]) -> str:
        """确定风险预警的严重程度
        
        Args:
            alert_type: 预警类型
            details: 预警详情
        
        Returns:
            严重程度 (低/中/高)
        """
        # 定义严重程度规则
        if alert_type == 'drawdown':
            drawdown = details.get('drawdown', 0.0)
            if drawdown > self.max_drawdown:
                return '高'
            elif drawdown > self.max_drawdown * 0.8:
                return '中'
            else:
                return '低'
        elif alert_type == 'position_size':
            weight = details.get('weight', 0.0)
            if weight > self.max_position_size:
                return '高'
            elif weight > self.max_position_size * 0.8:
                return '中'
            else:
                return '低'
        elif alert_type == 'trade_limit':
            return '中'
        elif alert_type == 'leverage':
            leverage = details.get('leverage', 0.0)
            if leverage > self.leverage_limit:
                return '高'
            elif leverage > self.leverage_limit * 0.8:
                return '中'
            else:
                return '低'
        elif alert_type == 'sector_exposure':
            exposure = details.get('exposure', 0.0)
            if exposure > self.sector_exposure_limit:
                return '高'
            elif exposure > self.sector_exposure_limit * 0.8:
                return '中'
            else:
                return '低'
        else:
            return '低'
    
    def _save_risk_alert(self, alert: Dict[str, Any]) -> None:
        """保存风险预警到文件
        
        Args:
            alert: 预警信息
        """
        try:
            alerts_dir = os.path.join(self.config.get('results_dir', './results'), 'risk_alerts')
            if not os.path.exists(alerts_dir):
                os.makedirs(alerts_dir)
            
            # 使用日期作为文件名
            date_str = datetime.now().strftime('%Y%m%d')
            alert_file = os.path.join(alerts_dir, f"risk_alerts_{date_str}.json")
            
            # 转换datetime对象为字符串
            alert_serializable = alert.copy()
            alert_serializable['timestamp'] = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存到文件
            if os.path.exists(alert_file):
                with open(alert_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            alerts.append(alert_serializable)
            
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存风险预警时发生异常: {str(e)}")
    
    def monitor_daily_risk(self, portfolio: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """每日风险监控
        
        Args:
            portfolio: 组合信息
            market_data: 市场数据
        
        Returns:
            监控结果
        """
        logger.info("执行每日风险监控")
        
        # 更新资金
        portfolio_value = sum(p.get('value', 0.0) for p in portfolio.values())
        self.update_capital(portfolio_value)
        
        # 计算组合风险指标
        risk_metrics = self.calculate_portfolio_risk(portfolio, market_data)
        
        # 检查风险预警
        alerts = []
        
        # 检查最大回撤
        if self.current_drawdown > self.max_drawdown * 0.8:
            alert = self.generate_risk_alert('drawdown', {
                'current_drawdown': self.current_drawdown,
                'max_drawdown_limit': self.max_drawdown,
                'peak_equity': self.peak_equity,
                'current_equity': self.current_capital
            })
            alerts.append(alert)
        
        # 检查单个持仓大小
        for ticker, position in portfolio.items():
            position_value = position.get('value', 0.0)
            position_weight = position_value / self.current_capital
            if position_weight > self.max_position_size * 0.8:
                alert = self.generate_risk_alert('position_size', {
                    'ticker': ticker,
                    'weight': position_weight,
                    'max_position_size': self.max_position_size,
                    'value': position_value
                })
                alerts.append(alert)
        
        # 检查行业暴露
        self._update_sector_exposures(portfolio)
        for sector, exposure in self.sector_exposures.items():
            if exposure > self.sector_exposure_limit * 0.8:
                alert = self.generate_risk_alert('sector_exposure', {
                    'sector': sector,
                    'exposure': exposure,
                    'limit': self.sector_exposure_limit
                })
                alerts.append(alert)
        
        # 检查杠杆
        leverage = portfolio_value / self.current_capital
        if leverage > self.leverage_limit * 0.8:
            alert = self.generate_risk_alert('leverage', {
                'leverage': leverage,
                'limit': self.leverage_limit,
                'portfolio_value': portfolio_value,
                'capital': self.current_capital
            })
            alerts.append(alert)
        
        # 检查波动率
        if 'volatility' in risk_metrics and risk_metrics['volatility'] > self.portfolio_vol_target * 1.2:
            alert = self.generate_risk_alert('volatility', {
                'current_volatility': risk_metrics['volatility'],
                'target_volatility': self.portfolio_vol_target
            })
            alerts.append(alert)
        
        # 构建监控结果
        monitoring_result = {
            'timestamp': datetime.now(),
            'capital': self.current_capital,
            'portfolio_value': portfolio_value,
            'risk_metrics': risk_metrics,
            'alerts': alerts,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_recorded': self.max_drawdown_recorded,
            'sector_exposures': self.sector_exposures
        }
        
        # 保存监控结果
        self._save_daily_monitoring(monitoring_result)
        
        logger.info(f"每日风险监控完成，发现 {len(alerts)} 个风险预警")
        
        return monitoring_result
    
    def _update_sector_exposures(self, portfolio: Dict[str, Any]) -> None:
        """更新行业暴露数据
        
        Args:
            portfolio: 组合信息
        """
        # 重置行业暴露
        self.sector_exposures = {}
        
        total_value = self.current_capital
        
        # 计算每个行业的暴露
        for position in portfolio.values():
            sector = position.get('sector', 'Unknown')
            value = position.get('value', 0.0)
            
            if sector not in self.sector_exposures:
                self.sector_exposures[sector] = 0.0
            
            self.sector_exposures[sector] += value
        
        # 计算百分比
        for sector in self.sector_exposures:
            self.sector_exposures[sector] /= total_value
    
    def _save_daily_monitoring(self, monitoring_result: Dict[str, Any]) -> None:
        """保存每日监控结果
        
        Args:
            monitoring_result: 监控结果
        """
        try:
            monitoring_dir = os.path.join(self.config.get('results_dir', './results'), 'daily_monitoring')
            if not os.path.exists(monitoring_dir):
                os.makedirs(monitoring_dir)
            
            # 使用日期作为文件名
            date_str = datetime.now().strftime('%Y%m%d')
            monitoring_file = os.path.join(monitoring_dir, f"monitoring_{date_str}.json")
            
            # 转换datetime对象为字符串
            monitoring_serializable = monitoring_result.copy()
            monitoring_serializable['timestamp'] = monitoring_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存到文件
            with open(monitoring_file, 'w', encoding='utf-8') as f:
                json.dump(monitoring_serializable, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存每日监控结果时发生异常: {str(e)}")
    
    def generate_risk_report(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           report_path: Optional[str] = None) -> str:
        """生成风险报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            report_path: 报告保存路径
        
        Returns:
            报告文件路径
        """
        logger.info("生成风险报告")
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 默认30天
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置保存路径
        if report_path is None:
            reports_dir = os.path.join(self.config.get('results_dir', './results'), 'risk_reports')
            report_path = os.path.join(reports_dir, f"risk_report_{timestamp}.html")
        
        # 准备报告内容
        report_content = []
        
        # 添加报告头部
        report_content.append('<!DOCTYPE html>')
        report_content.append('<html>')
        report_content.append('<head>')
        report_content.append('<meta charset="UTF-8">')
        report_content.append('<title>风险报告</title>')
        report_content.append('<style>')
        report_content.append('body { font-family: Arial, sans-serif; margin: 20px; }')
        report_content.append('h1, h2, h3 { color: #333; }')
        report_content.append('.container { max-width: 1200px; margin: 0 auto; }')
        report_content.append('.summary { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }')
        report_content.append('.metrics { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }')
        report_content.append('.metric-card { background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 5px; flex: 1; min-width: 200px; }')
        report_content.append('.metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }')
        report_content.append('.metric-name { color: #7f8c8d; }')
        report_content.append('table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
        report_content.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        report_content.append('th { background-color: #f2f2f2; }')
        report_content.append('tr:nth-child(even) { background-color: #f9f9f9; }')
        report_content.append('img { max-width: 100%; height: auto; margin: 20px 0; }')
        report_content.append('.alert { padding: 15px; margin: 10px 0; border-radius: 5px; }')
        report_content.append('.alert-high { background-color: #ffebee; border-left: 4px solid #f44336; }')
        report_content.append('.alert-medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }')
        report_content.append('.alert-low { background-color: #e3f2fd; border-left: 4px solid #2196f3; }')
        report_content.append('</style>')
        report_content.append('</head>')
        report_content.append('<body>')
        report_content.append('<div class="container">')
        report_content.append(f'<h1>风险报告 - {timestamp}</h1>')
        
        # 添加报告摘要
        report_content.append('<div class="summary">')
        report_content.append(f'<h2>报告期间: {start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")}</h2>')
        report_content.append('<table>')
        report_content.append('<tr><th>项目</th><th>值</th></tr>')
        report_content.append(f'<tr><td>初始资金</td><td>{self.initial_capital:,.2f}</td></tr>')
        report_content.append(f'<tr><td>当前资金</td><td>{self.current_capital:,.2f}</td></tr>')
        report_content.append(f'<tr><td>最大回撤</td><td>{self.max_drawdown_recorded:.2%}</td></tr>')
        report_content.append(f'<tr><td>风险限制</td><td>{self.risk_limit:.2%}</td></tr>')
        report_content.append(f'<tr><td>最大持仓占比</td><td>{self.max_position_size:.2%}</td></tr>')
        report_content.append('</table>')
        report_content.append('</div>')
        
        # 添加风险指标
        report_content.append('<h2>风险指标概览</h2>')
        
        # 绘制回撤图表
        drawdown_chart_path = self._plot_drawdown_history()
        if drawdown_chart_path:
            rel_path = os.path.relpath(drawdown_chart_path, os.path.dirname(report_path))
            report_content.append('<h3>资金回撤历史</h3>')
            report_content.append(f'<img src="{rel_path}" alt="资金回撤历史">')
        
        # 绘制行业暴露图表
        sector_chart_path = self._plot_sector_exposures()
        if sector_chart_path:
            rel_path = os.path.relpath(sector_chart_path, os.path.dirname(report_path))
            report_content.append('<h3>行业暴露分布</h3>')
            report_content.append(f'<img src="{rel_path}" alt="行业暴露分布">')
        
        # 添加风险预警列表
        report_content.append('<h2>近期风险预警</h2>')
        
        # 过滤指定日期范围内的预警
        recent_alerts = []
        for alert in self.risk_records:
            alert_time = alert['timestamp']
            if isinstance(alert_time, str):
                alert_time = datetime.strptime(alert_time, '%Y-%m-%d %H:%M:%S')
            
            if start_date <= alert_time <= end_date:
                recent_alerts.append(alert)
        
        if recent_alerts:
            for alert in recent_alerts:
                severity_class = f"alert-{alert['severity'].lower()}"
                report_content.append(f'<div class="alert {severity_class}">')
                report_content.append(f'<strong>{alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")} - {alert['alert_type']} ({alert['severity']})</strong><br>')
                report_content.append(f'<pre>{json.dumps(alert['details'], indent=2, ensure_ascii=False)}</pre>')
                report_content.append('</div>')
        else:
            report_content.append('<p>报告期间内无风险预警。</p>')
        
        # 添加风险配置信息
        report_content.append('<h2>风险配置</h2>')
        report_content.append('<table>')
        report_content.append('<tr><th>配置项</th><th>值</th></tr>')
        report_content.append(f'<tr><td>单笔交易最大风险</td><td>{self.risk_limit:.2%}</td></tr>')
        report_content.append(f'<tr><td>单个持仓最大占比</td><td>{self.max_position_size:.2%}</td></tr>')
        report_content.append(f'<tr><td>最大回撤限制</td><td>{self.max_drawdown:.2%}</td></tr>')
        report_content.append(f'<tr><td>止损比例</td><td>{self.stop_loss_ratio:.2%}</td></tr>')
        report_content.append(f'<tr><td>止盈比例</td><td>{self.take_profit_ratio:.2%}</td></tr>')
        report_content.append(f'<tr><td>组合波动率目标</td><td>{self.portfolio_vol_target:.2%}</td></tr>')
        report_content.append(f'<tr><td>行业暴露限制</td><td>{self.sector_exposure_limit:.2%}</td></tr>')
        report_content.append(f'<tr><td>杠杆限制</td><td>{self.leverage_limit:.2f}x</td></tr>')
        report_content.append(f'<tr><td>每日交易限制</td><td>{self.trade_limit}</td></tr>')
        report_content.append('</table>')
        
        # 添加报告尾部
        report_content.append('</div>')
        report_content.append('</body>')
        report_content.append('</html>')
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"风险报告已生成: {report_path}")
        
        return report_path
    
    def _plot_drawdown_history(self) -> Optional[str]:
        """绘制回撤历史图表
        
        Returns:
            图表保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.drawdown_history:
                logger.warning("没有回撤历史数据可供绘制图表")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(self.drawdown_history)
            
            # 设置图表样式
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # 绘制回撤曲线
            plt.plot(df['date'], df['drawdown'] * 100, 'b-', linewidth=2)
            
            # 添加最大回撤限制线
            plt.axhline(y=self.max_drawdown * 100, color='r', linestyle='--', label=f'最大回撤限制 ({self.max_drawdown * 100:.1f}%)')
            
            # 添加当前回撤点
            if not df.empty:
                latest_date = df['date'].iloc[-1]
                latest_drawdown = df['drawdown'].iloc[-1] * 100
                plt.scatter(latest_date, latest_drawdown, color='red', s=100, zorder=5)
                plt.annotate(f'{latest_drawdown:.1f}%', 
                            (latest_date, latest_drawdown), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
            
            # 设置图表属性
            plt.title('资金回撤历史', fontsize=16)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('回撤百分比 (%)', fontsize=12)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            figures_dir = os.path.join(self.config.get('figures_dir', './figures'), 'risk')
            chart_path = os.path.join(figures_dir, f"drawdown_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            logger.error(f"绘制回撤历史图表时发生异常: {str(e)}")
            return None
    
    def _plot_sector_exposures(self) -> Optional[str]:
        """绘制行业暴露图表
        
        Returns:
            图表保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.sector_exposures:
                logger.warning("没有行业暴露数据可供绘制图表")
                return None
            
            # 创建数据
            sectors = list(self.sector_exposures.keys())
            exposures = [self.sector_exposures[s] * 100 for s in sectors]
            
            # 设置图表样式
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # 创建条形图
            bars = plt.bar(sectors, exposures, color='skyblue')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # 添加行业暴露限制线
            plt.axhline(y=self.sector_exposure_limit * 100, color='r', linestyle='--', label=f'行业暴露限制 ({self.sector_exposure_limit * 100:.1f}%)')
            
            # 设置图表属性
            plt.title('行业暴露分布', fontsize=16)
            plt.xlabel('行业', fontsize=12)
            plt.ylabel('暴露百分比 (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            figures_dir = os.path.join(self.config.get('figures_dir', './figures'), 'risk')
            chart_path = os.path.join(figures_dir, f"sector_exposures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            logger.error(f"绘制行业暴露图表时发生异常: {str(e)}")
            return None
    
    def save_state(self, file_path: Optional[str] = None) -> str:
        """保存风险管理器状态
        
        Args:
            file_path: 保存路径
        
        Returns:
            实际保存路径
        """
        try:
            logger.info("保存风险管理器状态")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if file_path is None:
                states_dir = os.path.join(self.config.get('results_dir', './results'), 'risk_states')
                if not os.path.exists(states_dir):
                    os.makedirs(states_dir)
                
                file_path = os.path.join(states_dir, f"risk_manager_state_{timestamp}.json")
            
            # 准备状态数据
            state_data = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'risk_limit': self.risk_limit,
                'max_position_size': self.max_position_size,
                'max_drawdown': self.max_drawdown,
                'stop_loss_ratio': self.stop_loss_ratio,
                'take_profit_ratio': self.take_profit_ratio,
                'risk_free_rate': self.risk_free_rate,
                'portfolio_vol_target': self.portfolio_vol_target,
                'sector_exposure_limit': self.sector_exposure_limit,
                'leverage_limit': self.leverage_limit,
                'trade_limit': self.trade_limit,
                'peak_equity': self.peak_equity,
                'current_drawdown': self.current_drawdown,
                'max_drawdown_recorded': self.max_drawdown_recorded,
                'trade_count': self.trade_count,
                'daily_trades': self.daily_trades,
                'last_reset_date': self.last_reset_date.strftime('%Y-%m-%d'),
                'timestamp': timestamp,
                'config': self.config
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"风险管理器状态已保存到: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"保存风险管理器状态时发生异常: {str(e)}")
            raise
    
    def load_state(self, file_path: str) -> bool:
        """加载风险管理器状态
        
        Args:
            file_path: 状态文件路径
        
        Returns:
            是否加载成功
        """
        try:
            logger.info(f"加载风险管理器状态: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"状态文件不存在: {file_path}")
                return False
            
            # 读取状态数据
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 恢复状态
            self.initial_capital = state_data.get('initial_capital', self.initial_capital)
            self.current_capital = state_data.get('current_capital', self.current_capital)
            self.risk_limit = state_data.get('risk_limit', self.risk_limit)
            self.max_position_size = state_data.get('max_position_size', self.max_position_size)
            self.max_drawdown = state_data.get('max_drawdown', self.max_drawdown)
            self.stop_loss_ratio = state_data.get('stop_loss_ratio', self.stop_loss_ratio)
            self.take_profit_ratio = state_data.get('take_profit_ratio', self.take_profit_ratio)
            self.risk_free_rate = state_data.get('risk_free_rate', self.risk_free_rate)
            self.portfolio_vol_target = state_data.get('portfolio_vol_target', self.portfolio_vol_target)
            self.sector_exposure_limit = state_data.get('sector_exposure_limit', self.sector_exposure_limit)
            self.leverage_limit = state_data.get('leverage_limit', self.leverage_limit)
            self.trade_limit = state_data.get('trade_limit', self.trade_limit)
            self.peak_equity = state_data.get('peak_equity', self.peak_equity)
            self.current_drawdown = state_data.get('current_drawdown', self.current_drawdown)
            self.max_drawdown_recorded = state_data.get('max_drawdown_recorded', self.max_drawdown_recorded)
            self.trade_count = state_data.get('trade_count', self.trade_count)
            self.daily_trades = state_data.get('daily_trades', self.daily_trades)
            self.last_reset_date = datetime.strptime(state_data.get('last_reset_date', datetime.now().date().strftime('%Y-%m-%d')), '%Y-%m-%d').date()
            
            # 更新配置
            if 'config' in state_data:
                self.config.update(state_data['config'])
                # 更新日志
                self.logger = self._init_logger()
            
            logger.info("风险管理器状态加载完成")
            
            return True
        except Exception as e:
            logger.error(f"加载风险管理器状态时发生异常: {str(e)}")
            return False

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['RiskManager']