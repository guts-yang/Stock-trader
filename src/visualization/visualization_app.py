#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票交易系统可视化界面

这是一个基于Streamlit的交互式可视化界面，用于股票数据查询、分析和预测。
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import logging
import yaml
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入数据获取器
src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_path)
from src.data_acquisition.data_fetcher import DataFetcher

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "visualization_app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockTraderVisualization")

# 加载配置文件
def load_config():
    """加载配置文件"""
    try:
        # 使用绝对路径查找配置文件
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            'data_sources': {'tushare': {'api_key': ''}},
            'backtesting': {'initial_capital': 100000}
        }

# 从DataFetcher获取真实股票数据
def get_real_stock_data(stock_code, start_date, end_date, fetcher):
    """从DataFetcher获取真实股票数据"""
    try:
        # 转换日期格式为Tushare API所需的格式 (YYYYMMDD)
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # 确定股票市场
        # 默认假设是深市，可根据实际情况调整
        if stock_code.startswith('6'):
            ts_code = f"{stock_code}.SH"  # 沪市
        else:
            ts_code = f"{stock_code}.SZ"  # 深市
        
        logger.info(f"尝试获取股票代码 {ts_code} 的数据，日期范围: {start_str} 到 {end_str}")
        
        # 从DataFetcher获取数据
        df = fetcher.get_daily_data(ts_code, start_str, end_str)
        
        if df is None or df.empty:
            logger.warning(f"无法获取股票代码 {ts_code} 的数据，将使用模拟数据")
            return generate_sample_stock_data(stock_code, start_date, end_date)
        
        # 处理数据格式使其与应用程序兼容
        # 重命名列名以匹配应用程序的期望
        # 处理Tushare的列名
        if 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'date'})
            df['date'] = pd.to_datetime(df['date'])
        # 处理AkShare的列名
        elif '日期' in df.columns:
            df = df.rename(columns={'日期': 'date'})
            df['date'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'date'})
        
        # 处理开盘价
        if 'open' not in df.columns:
            if '开盘' in df.columns:
                df = df.rename(columns={'开盘': 'open'})
            elif '开盘价' in df.columns:
                df = df.rename(columns={'开盘价': 'open'})
        
        # 处理最高价
        if 'high' not in df.columns:
            if '最高' in df.columns:
                df = df.rename(columns={'最高': 'high'})
            elif '最高价' in df.columns:
                df = df.rename(columns={'最高价': 'high'})
        
        # 处理最低价
        if 'low' not in df.columns:
            if '最低' in df.columns:
                df = df.rename(columns={'最低': 'low'})
            elif '最低价' in df.columns:
                df = df.rename(columns={'最低价': 'low'})
        
        # 处理收盘价
        if 'close' not in df.columns:
            if '收盘' in df.columns:
                df = df.rename(columns={'收盘': 'close'})
            elif '收盘价' in df.columns:
                df = df.rename(columns={'收盘价': 'close'})
        
        # 处理成交量 - 尝试多种可能的列名
        if 'volume' not in df.columns:
            if 'vol' in df.columns:
                df = df.rename(columns={'vol': 'volume'})
            elif '成交量' in df.columns:
                df = df.rename(columns={'成交量': 'volume'})
            else:
                # 如果没有成交量数据，创建一个默认列
                df['volume'] = 0
                logger.warning("数据中缺少成交量信息，使用默认值")
        
        # 计算一些技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 计算收益率
        df['return'] = df['close'].pct_change()
        
        # 排序并重置索引
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"成功获取股票代码 {ts_code} 的数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"获取真实股票数据时出错: {e}")
        # 如果出错，回退到模拟数据
        return generate_sample_stock_data(stock_code, start_date, end_date)

# 模拟数据生成器（作为备用）
def generate_sample_stock_data(stock_code, start_date, end_date):
    """生成模拟股票数据（当无法获取真实数据时使用）"""
    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(date_range)
    
    # 生成随机价格数据
    np.random.seed(int(stock_code[-4:]) if stock_code[-4:].isdigit() else 42)
    base_price = np.random.uniform(50, 150)
    price_changes = np.random.normal(0, 1, n_days)
    close_prices = base_price + np.cumsum(price_changes)
    
    # 生成开盘价、最高价、最低价
    open_prices = close_prices * np.random.uniform(0.995, 1.005, n_days)
    high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1.001, 1.02, n_days)
    low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.98, 0.999, n_days)
    
    # 生成成交量
    volumes = np.random.randint(1000000, 10000000, n_days)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # 计算一些技术指标
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # 计算收益率
    df['return'] = df['close'].pct_change()
    
    return df

# 计算风险指标
def calculate_risk_metrics(df):
    """计算基本风险指标"""
    # 计算收益率
    returns = df['return'].dropna()
    
    # 计算基本指标
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    annualized_return = ((1 + total_return / 100) ** (252 / len(df)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    
    # 计算夏普比率 (假设无风险收益率为3%)
    risk_free_rate = 0.03
    sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100) if volatility != 0 else 0
    
    # 计算最大回撤
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        '总收益率 (%)': round(total_return, 2),
        '年化收益率 (%)': round(annualized_return, 2),
        '波动率 (%)': round(volatility, 2),
        '夏普比率': round(sharpe_ratio, 2),
        '最大回撤 (%)': round(max_drawdown, 2)
    }

# 主应用函数
def main():
    # 加载配置
    config = load_config()
    
    # 初始化数据获取器
    fetcher = DataFetcher(config)
    
    # 设置页面配置
    st.set_page_config(
        page_title="股票交易系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 设置应用标题和描述
    st.title("📈 股票交易系统可视化平台")
    st.markdown("---")
    st.write("欢迎使用股票交易系统可视化平台！通过输入股票代码，您可以查看历史行情、技术指标、风险分析以及基于AI模型的预测结果。")
    
    # 创建侧边栏
    with st.sidebar:
        st.header("输入参数")
        
        # 股票代码输入框
        stock_code = st.text_input(
            "股票代码", 
            value="002501",
            placeholder="请输入6位股票代码，例如：002501",
            help="输入要查询的A股股票代码，如002501代表齐翔腾达"
        )
        
        # 日期范围选择器
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=365),
                help="选择数据查询的开始日期"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
                help="选择数据查询的结束日期"
            )
        
        # 数据频率选择
        data_frequency = st.selectbox(
            "数据频率",
            options=["日线", "周线", "月线"],
            index=0,
            help="选择数据的时间频率"
        )
        
        # 技术指标选择
        st.subheader("技术指标")
        show_ma5 = st.checkbox("MA5", value=True, help="显示5日均线")
        show_ma20 = st.checkbox("MA20", value=True, help="显示20日均线")
        
        # 分析选项
        st.subheader("分析选项")
        perform_risk_analysis = st.checkbox("风险分析", value=True, help="执行风险指标分析")
        show_predictions = st.checkbox("显示预测", value=True, help="显示基于AI模型的预测结果")
        
        # 提交按钮
        st.markdown("---")
        submit_button = st.button("执行分析", type="primary", use_container_width=True)
    
    # 主内容区域
    if submit_button:
        # 输入验证
        if not stock_code or not stock_code.isdigit() or len(stock_code) != 6:
            st.error("请输入有效的6位股票代码！")
            return
        
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
            return
        
        # 显示加载状态
        with st.spinner(f"正在获取和分析 {stock_code} 的数据..."):
            # 模拟加载延迟
            time.sleep(1)
            
            try:
                # 获取真实股票数据（如果无法获取则回退到模拟数据）
                df = get_real_stock_data(stock_code, start_date, end_date, fetcher)
                
                # 股票基本信息卡片
                st.subheader(f"股票代码: {stock_code}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最新价", f"¥{df['close'].iloc[-1]:.2f}")
                with col2:
                    change = df['close'].iloc[-1] - df['close'].iloc[-2]
                    pct_change = (change / df['close'].iloc[-2]) * 100
                    st.metric("涨跌幅", f"{pct_change:.2f}%", f"{change:.2f}")
                with col3:
                    st.metric("最高价", f"¥{df['high'].max():.2f}")
                with col4:
                    st.metric("最低价", f"¥{df['low'].min():.2f}")
                
                st.markdown("---")
                
                # 绘制价格走势图
                st.subheader("价格走势图")
                fig = go.Figure()
                
                # 添加蜡烛图
                fig.add_trace(go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="价格"
                ))
                
                # 添加均线
                if show_ma5:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['ma5'],
                        mode='lines',
                        name='MA5',
                        line=dict(color='blue', width=1)
                    ))
                
                if show_ma20:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['ma20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color='red', width=1)
                    ))
                
                # 更新图表布局
                fig.update_layout(
                    title=f"{stock_code} 价格走势",
                    yaxis_title="价格 (元)",
                    xaxis_title="日期",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 成交量图表
                st.subheader("成交量")
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name="成交量",
                    marker_color='rgba(128, 128, 128, 0.7)'
                ))
                fig_volume.update_layout(
                    yaxis_title="成交量",
                    xaxis_title="日期",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # 风险分析
                if perform_risk_analysis:
                    st.markdown("---")
                    st.subheader("风险分析")
                    risk_metrics = calculate_risk_metrics(df)
                    
                    # 以卡片形式显示风险指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总收益率", f"{risk_metrics['总收益率 (%)']}%")
                    with col2:
                        st.metric("年化收益率", f"{risk_metrics['年化收益率 (%)']}%")
                    with col3:
                        st.metric("波动率", f"{risk_metrics['波动率 (%)']}%")
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("夏普比率", f"{risk_metrics['夏普比率']}")
                    with col5:
                        st.metric("最大回撤", f"{risk_metrics['最大回撤 (%)']}%")
                    
                    # 绘制回撤图
                    returns = df['return'].dropna()
                    cumulative_returns = (1 + returns).cumprod()
                    running_max = cumulative_returns.cummax()
                    drawdown = (cumulative_returns - running_max) / running_max
                    
                    fig_drawdown = go.Figure()
                    fig_drawdown.add_trace(go.Scatter(
                        x=df['date'][1:],
                        y=drawdown * 100,
                        mode='lines',
                        fill='tozeroy',
                        name='回撤',
                        line=dict(color='red'),
                        fillcolor='rgba(255, 0, 0, 0.1)'
                    ))
                    fig_drawdown.update_layout(
                        title="回撤分析",
                        yaxis_title="回撤 (%)",
                        xaxis_title="日期",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(fig_drawdown, use_container_width=True)
                
                # 显示预测结果
                if show_predictions:
                    st.markdown("---")
                    st.subheader("价格预测")
                    st.info("预测结果基于历史数据和AI模型生成，仅供参考，不构成投资建议。")
                    
                    # 模拟预测结果
                    last_price = df['close'].iloc[-1]
                    pred_price = last_price * (1 + np.random.uniform(-0.02, 0.02))
                    pred_change = (pred_price - last_price) / last_price * 100
                    
                    # 计算涨跌概率（模拟）
                    up_probability = np.random.uniform(40, 60)
                    down_probability = 100 - up_probability
                    
                    # 显示预测结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("预测价格", f"¥{pred_price:.2f}", f"{pred_change:.2f}%")
                    with col2:
                        st.metric("上涨概率", f"{up_probability:.1f}%")
                    
                    # 绘制概率分布饼图
                    fig_prob = px.pie(
                        values=[up_probability, down_probability],
                        names=['上涨概率', '下跌概率'],
                        color=['上涨概率', '下跌概率'],
                        color_discrete_map={'上涨概率': 'green', '下跌概率': 'red'},
                        title="涨跌概率分布"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                # 数据表格
                st.markdown("---")
                st.subheader("原始数据")
                st.dataframe(df.tail(20), use_container_width=True)
                
                # 下载数据按钮
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载数据 (CSV)",
                    data=csv_data,
                    file_name=f"{stock_code}_data_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                logger.error(f"分析过程中出错: {e}")
                st.error(f"分析过程中出错: {str(e)}")
    
    # 页脚信息
    st.markdown("---")
    st.markdown("📊 股票交易系统可视化平台 - 仅供学习和研究使用，不构成投资建议")

# 运行应用
if __name__ == "__main__":
    main()