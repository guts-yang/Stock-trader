"""深度学习模型评估器模块，实现模型性能评估和可视化"""

import numpy as np
import pandas as pd
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ModelEvaluator:
    """深度学习模型评估器类，提供模型性能评估和可视化功能"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化模型评估器
        
        Args:
            config: 评估配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'log_dir': './logs',
            'plots_dir': './plots',
            'results_dir': './results',
            'classification_threshold': 0.5,
            'regression_metrics': ['mae', 'mse', 'rmse', 'r2', 'mape', 'smape', 'bias'],
            'classification_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'show_plots': True,
            'save_plots': True,
            'plot_dpi': 300,
            'return_results': True,
            'verbose': 1
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 创建必要的目录
        self._create_directories()
        
        self.logger.info("模型评估器初始化完成")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('ModelEvaluator')
        log_dir = self.config.get('log_dir', './logs')
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 设置日志级别
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 文件处理器
            log_file = os.path.join(log_dir, 'model_evaluation.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _create_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            self.config.get('log_dir', './logs'),
            self.config.get('plots_dir', './plots'),
            self.config.get('results_dir', './results')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.logger.info(f"创建目录: {directory}")
                except Exception as e:
                    self.logger.error(f"创建目录{directory}时发生异常: {str(e)}")
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          save_results: bool = True, 
                          plot_results: bool = True) -> Dict[str, float]:
        """评估回归模型性能
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            save_results: 是否保存评估结果
            plot_results: 是否绘制评估结果图形
        
        Returns:
            评估指标字典
        """
        try:
            self.logger.info("开始评估回归模型性能")
            
            # 检查输入
            if len(y_true) != len(y_pred):
                raise ValueError(f"真实值和预测值长度不匹配: {len(y_true)} vs {len(y_pred)}")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            
            self.logger.info(f"评估数据点数: {len(y_true)}")
            
            # 计算评估指标
            metrics_dict = {}
            
            # 平均绝对误差 (MAE)
            if 'mae' in self.config.get('regression_metrics', []):
                mae = mean_absolute_error(y_true, y_pred)
                metrics_dict['mae'] = mae
            
            # 均方误差 (MSE)
            if 'mse' in self.config.get('regression_metrics', []):
                mse = mean_squared_error(y_true, y_pred)
                metrics_dict['mse'] = mse
            
            # 均方根误差 (RMSE)
            if 'rmse' in self.config.get('regression_metrics', []):
                rmse = np.sqrt(metrics_dict.get('mse', mean_squared_error(y_true, y_pred)))
                metrics_dict['rmse'] = rmse
            
            # 决定系数 (R2)
            if 'r2' in self.config.get('regression_metrics', []):
                r2 = r2_score(y_true, y_pred)
                metrics_dict['r2'] = r2
            
            # 平均绝对百分比误差 (MAPE)
            if 'mape' in self.config.get('regression_metrics', []):
                # 避免除以零
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                else:
                    mape = 0
                metrics_dict['mape'] = mape
            
            # 对称平均绝对百分比误差 (SMAPE)
            if 'smape' in self.config.get('regression_metrics', []):
                # 避免除以零
                denominator = np.abs(y_true) + np.abs(y_pred)
                non_zero_mask = denominator != 0
                if np.any(non_zero_mask):
                    smape = (100 / np.sum(non_zero_mask)) * \
                            np.sum(2 * np.abs(y_pred[non_zero_mask] - y_true[non_zero_mask]) / \
                                  denominator[non_zero_mask])
                else:
                    smape = 0
                metrics_dict['smape'] = smape
            
            # 预测偏差
            if 'bias' in self.config.get('regression_metrics', []):
                bias = np.mean(y_pred - y_true)
                metrics_dict['bias'] = bias
            
            # 方向准确率（预测涨跌方向的准确率）
            if len(y_true) > 1:
                # 计算真实方向变化
                true_direction = np.sign(y_true[1:] - y_true[:-1])
                # 计算预测方向变化
                pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
                # 计算方向准确率
                direction_accuracy = np.mean(true_direction == pred_direction)
                metrics_dict['direction_accuracy'] = direction_accuracy
            
            # 打印评估结果
            self.logger.info(f"回归模型评估结果: {metrics_dict}")
            
            # 保存评估结果
            if save_results:
                self._save_evaluation_results(metrics_dict, 'regression')
            
            # 绘制评估结果
            if plot_results:
                self.plot_regression_results(y_true, y_pred)
            
            # 返回评估结果
            if self.config.get('return_results', True):
                return metrics_dict
        except Exception as e:
            self.logger.error(f"评估回归模型性能时发生异常: {str(e)}")
            raise
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              threshold: Optional[float] = None, 
                              save_results: bool = True, 
                              plot_results: bool = True) -> Dict[str, float]:
        """评估分类模型性能
        
        Args:
            y_true: 真实标签数组
            y_pred_proba: 预测概率数组
            threshold: 分类阈值，如果为None则使用配置中的值
            save_results: 是否保存评估结果
            plot_results: 是否绘制评估结果图形
        
        Returns:
            评估指标字典
        """
        try:
            self.logger.info("开始评估分类模型性能")
            
            # 获取分类阈值
            if threshold is None:
                threshold = self.config.get('classification_threshold', 0.5)
            
            # 检查输入
            if len(y_true) != len(y_pred_proba):
                raise ValueError(f"真实标签和预测概率长度不匹配: {len(y_true)} vs {len(y_pred_proba)}")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            
            # 如果预测概率是二维数组（多类别），取第一列作为正类概率
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                # 假设二分类问题，取第二个类别作为正类
                y_pred_proba = y_pred_proba[:, 1]
            elif y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba.squeeze()
            
            # 根据阈值生成预测标签
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            self.logger.info(f"评估数据点数: {len(y_true)}, 分类阈值: {threshold}")
            
            # 计算评估指标
            metrics_dict = {}
            
            # 准确率
            if 'accuracy' in self.config.get('classification_metrics', []):
                accuracy = accuracy_score(y_true, y_pred)
                metrics_dict['accuracy'] = accuracy
            
            # 精确率
            if 'precision' in self.config.get('classification_metrics', []):
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics_dict['precision'] = precision
            
            # 召回率
            if 'recall' in self.config.get('classification_metrics', []):
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics_dict['recall'] = recall
            
            # F1分数
            if 'f1' in self.config.get('classification_metrics', []):
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics_dict['f1'] = f1
            
            # ROC AUC
            if 'roc_auc' in self.config.get('classification_metrics', []):
                try:
                    roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
                    metrics_dict['roc_auc'] = roc_auc
                except ValueError:
                    self.logger.warning("无法计算ROC AUC，可能是因为只有一个类别")
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            self.logger.info(f"混淆矩阵:\n{cm}")
            metrics_dict['confusion_matrix'] = cm.tolist()
            
            # 分类报告
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            self.logger.info(f"分类报告:\n{classification_report(y_true, y_pred, zero_division=0)}")
            metrics_dict['classification_report'] = class_report
            
            # 打印评估结果
            self.logger.info(f"分类模型评估结果: {metrics_dict}")
            
            # 保存评估结果
            if save_results:
                self._save_evaluation_results(metrics_dict, 'classification')
            
            # 绘制评估结果
            if plot_results:
                self.plot_classification_results(y_true, y_pred, y_pred_proba)
            
            # 返回评估结果
            if self.config.get('return_results', True):
                return metrics_dict
        except Exception as e:
            self.logger.error(f"评估分类模型性能时发生异常: {str(e)}")
            raise
    
    def _save_evaluation_results(self, metrics_dict: Dict, result_type: str) -> None:
        """保存评估结果到文件
        
        Args:
            metrics_dict: 评估指标字典
            result_type: 结果类型，'regression'或'classification'
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON文件
            results_dir = self.config.get('results_dir', './results')
            result_file = os.path.join(results_dir, f'{result_type}_evaluation_{timestamp}.json')
            
            # 转换numpy数组为Python列表以便JSON序列化
            serializable_dict = {}
            for key, value in metrics_dict.items():
                if isinstance(value, np.ndarray):
                    serializable_dict[key] = value.tolist()
                elif isinstance(value, np.generic):
                    serializable_dict[key] = np.asscalar(value)
                else:
                    serializable_dict[key] = value
            
            # 保存到文件
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"评估结果已保存到: {result_file}")
        except Exception as e:
            self.logger.error(f"保存评估结果时发生异常: {str(e)}")
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """绘制回归模型评估结果图形
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
        """
        try:
            self.logger.info("绘制回归模型评估结果图形")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 创建图形
            plt.figure(figsize=(15, 10))
            
            # 1. 真实值 vs 预测值散点图
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5, s=30)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('真实值 vs 预测值')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 2. 残差图
            residuals = y_true - y_pred
            plt.subplot(2, 2, 2)
            plt.scatter(y_pred, residuals, alpha=0.5, s=30)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.title('残差图')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 3. 残差分布直方图
            plt.subplot(2, 2, 3)
            sns.histplot(residuals, kde=True, bins=30)
            plt.xlabel('残差')
            plt.ylabel('频率')
            plt.title('残差分布')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 4. 预测误差时间序列图（如果有时间维度）
            plt.subplot(2, 2, 4)
            if len(y_true) > 1:
                plt.plot(range(len(y_true)), y_true, 'b-', label='真实值')
                plt.plot(range(len(y_pred)), y_pred, 'r--', label='预测值')
                plt.xlabel('时间步')
                plt.ylabel('值')
                plt.title('预测与真实值时间序列对比')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'regression_evaluation_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"回归评估图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制回归模型评估结果图形时发生异常: {str(e)}")
    
    def plot_classification_results(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """绘制分类模型评估结果图形
        
        Args:
            y_true: 真实标签数组
            y_pred: 预测标签数组
            y_pred_proba: 预测概率数组
        """
        try:
            self.logger.info("绘制分类模型评估结果图形")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_pred_proba = np.array(y_pred_proba)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba.squeeze()
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 创建图形
            plt.figure(figsize=(15, 10))
            
            # 1. 混淆矩阵热图
            plt.subplot(2, 2, 1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['预测负类', '预测正类'], 
                        yticklabels=['真实负类', '真实正类'])
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.title('混淆矩阵')
            
            # 2. ROC曲线
            plt.subplot(2, 2, 2)
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC曲线 (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率')
                plt.ylabel('真阳性率')
                plt.title('接收器操作特征 (ROC) 曲线')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
            except ValueError:
                self.logger.warning("无法绘制ROC曲线，可能是因为只有一个类别")
                plt.text(0.5, 0.5, '无法绘制ROC曲线\n(可能是因为只有一个类别)', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.axis('off')
            
            # 3. 精确率-召回率曲线
            plt.subplot(2, 2, 3)
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                plt.plot(recall, precision, color='green', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('召回率')
                plt.ylabel('精确率')
                plt.title('精确率-召回率曲线')
                plt.grid(True, linestyle='--', alpha=0.7)
            except ValueError:
                self.logger.warning("无法绘制精确率-召回率曲线，可能是因为只有一个类别")
                plt.text(0.5, 0.5, '无法绘制精确率-召回率曲线\n(可能是因为只有一个类别)', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.axis('off')
            
            # 4. 预测概率分布直方图
            plt.subplot(2, 2, 4)
            # 分别绘制正负类的预测概率分布
            sns.histplot(y_pred_proba[y_true == 0], kde=True, bins=20, alpha=0.5, label='负类')
            sns.histplot(y_pred_proba[y_true == 1], kde=True, bins=20, alpha=0.5, label='正类')
            plt.axvline(x=self.config.get('classification_threshold', 0.5), color='r', linestyle='--', 
                       label=f'分类阈值 ({self.config.get('classification_threshold', 0.5)})')
            plt.xlabel('预测概率')
            plt.ylabel('频率')
            plt.title('预测概率分布')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'classification_evaluation_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"分类评估图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制分类模型评估结果图形时发生异常: {str(e)}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       title: str = '预测与真实值对比', 
                       xlabel: str = '样本', 
                       ylabel: str = '值') -> None:
        """绘制预测值与真实值对比图
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        try:
            self.logger.info("绘制预测值与真实值对比图")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            
            # 创建图形
            plt.figure(figsize=(12, 6))
            
            # 绘制真实值和预测值
            plt.plot(range(len(y_true)), y_true, 'b-', label='真实值', linewidth=2)
            plt.plot(range(len(y_pred)), y_pred, 'r--', label='预测值', linewidth=2)
            
            # 设置图表属性
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'predictions_comparison_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"预测对比图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制预测值与真实值对比图时发生异常: {str(e)}")
    
    def calculate_feature_importance(self, model: object, X: np.ndarray, 
                                    feature_names: Optional[List[str]] = None, 
                                    top_n: int = 10, 
                                    plot: bool = True) -> Dict[str, float]:
        """计算特征重要性
        
        Args:
            model: 训练好的模型
            X: 特征数据
            feature_names: 特征名称列表，如果为None则使用默认名称
            top_n: 显示前N个重要特征
            plot: 是否绘制特征重要性图形
        
        Returns:
            特征重要性字典
        """
        try:
            self.logger.info("计算特征重要性")
            
            # 获取特征重要性
            feature_importance = None
            
            # 尝试不同的方法获取特征重要性
            if hasattr(model, 'feature_importances_'):
                # 随机森林、梯度提升树等模型
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # 线性模型
                feature_importance = np.abs(model.coef_[0])
            elif hasattr(model, 'get_weights') or hasattr(model, 'weights'):
                # Keras模型
                try:
                    if hasattr(model, 'get_weights'):
                        weights = model.get_weights()
                    else:
                        weights = model.weights
                    
                    # 假设第一个权重矩阵是输入层权重
                    if len(weights) > 0 and isinstance(weights[0], np.ndarray):
                        # 计算每个特征的权重绝对值之和
                        if len(weights[0].shape) > 1:
                            feature_importance = np.sum(np.abs(weights[0]), axis=1)
                        else:
                            feature_importance = np.abs(weights[0])
                except Exception as e:
                    self.logger.warning(f"无法从Keras模型获取特征重要性: {str(e)}")
            
            if feature_importance is None:
                self.logger.warning("无法获取特征重要性，模型不支持")
                return {}
            
            # 确保特征重要性是一维数组
            if feature_importance.ndim > 1:
                feature_importance = feature_importance.squeeze()
            
            # 归一化特征重要性
            feature_importance = feature_importance / np.sum(feature_importance) if np.sum(feature_importance) > 0 else feature_importance
            
            # 设置特征名称
            if feature_names is None:
                feature_names = [f'特征{i+1}' for i in range(len(feature_importance))]
            
            # 创建特征重要性字典
            importance_dict = {feature_names[i]: float(feature_importance[i]) for i in range(len(feature_names))}
            
            # 按重要性排序
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # 打印前N个重要特征
            self.logger.info(f"前{min(top_n, len(sorted_importance))}个重要特征:")
            for feature, importance in sorted_importance[:top_n]:
                self.logger.info(f"{feature}: {importance:.6f}")
            
            # 绘制特征重要性图形
            if plot and len(sorted_importance) > 0:
                self._plot_feature_importance(sorted_importance, top_n)
            
            return dict(sorted_importance)
        except Exception as e:
            self.logger.error(f"计算特征重要性时发生异常: {str(e)}")
            raise
    
    def _plot_feature_importance(self, sorted_importance: List[Tuple[str, float]], top_n: int) -> None:
        """绘制特征重要性图形
        
        Args:
            sorted_importance: 排序后的特征重要性列表
            top_n: 显示前N个重要特征
        """
        try:
            self.logger.info("绘制特征重要性图形")
            
            # 获取前N个重要特征
            top_features = sorted_importance[:top_n]
            features, importances = zip(*top_features)
            
            # 创建图形
            plt.figure(figsize=(10, min(6, len(top_features) * 0.3)))
            
            # 绘制水平条形图
            plt.barh(range(len(features)), importances, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('重要性')
            plt.title(f'前{min(top_n, len(sorted_importance))}个重要特征')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'feature_importance_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"特征重要性图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制特征重要性图形时发生异常: {str(e)}")
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         X: Optional[np.ndarray] = None, 
                         feature_names: Optional[List[str]] = None) -> Dict:
        """分析模型残差
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            X: 特征数据，如果为None则不进行特征相关分析
            feature_names: 特征名称列表
        
        Returns:
            残差分析结果字典
        """
        try:
            self.logger.info("分析模型残差")
            
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 确保一维数组
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            
            # 计算残差
            residuals = y_true - y_pred
            
            # 计算残差统计信息
            residual_stats = {
                'mean': float(np.mean(residuals)),
                'median': float(np.median(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'skew': float(pd.Series(residuals).skew()),
                'kurtosis': float(pd.Series(residuals).kurtosis())
            }
            
            self.logger.info(f"残差统计信息: {residual_stats}")
            
            # 绘制残差分析图形
            self._plot_residual_analysis(y_true, y_pred, residuals)
            
            # 分析残差与特征的相关性（如果提供了特征数据）
            feature_correlations = {}
            if X is not None:
                # 转换为numpy数组
                X = np.array(X)
                
                # 确保二维数组
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                
                # 设置特征名称
                if feature_names is None:
                    feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
                
                # 计算残差与每个特征的相关性
                for i in range(X.shape[1]):
                    # 确保特征是一维的
                    feature = X[:, i].squeeze()
                    
                    # 计算相关性
                    if len(feature) == len(residuals):
                        correlation = np.corrcoef(feature, residuals)[0, 1]
                        feature_correlations[feature_names[i]] = float(correlation)
                
                # 按相关性绝对值排序
                sorted_correlations = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                
                self.logger.info("残差与特征的相关性（绝对值从大到小）:")
                for feature, corr in sorted_correlations[:10]:  # 显示前10个
                    self.logger.info(f"{feature}: {corr:.4f}")
            
            # 综合结果
            analysis_results = {
                'residual_stats': residual_stats,
                'feature_correlations': feature_correlations
            }
            
            # 保存分析结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = self.config.get('results_dir', './results')
            result_file = os.path.join(results_dir, f'residual_analysis_{timestamp}.json')
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"残差分析结果已保存到: {result_file}")
            
            return analysis_results
        except Exception as e:
            self.logger.error(f"分析模型残差时发生异常: {str(e)}")
            raise
    
    def _plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> None:
        """绘制残差分析图形
        
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            residuals: 残差数组
        """
        try:
            self.logger.info("绘制残差分析图形")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 创建图形
            plt.figure(figsize=(15, 10))
            
            # 1. 残差与预测值的关系
            plt.subplot(2, 2, 1)
            plt.scatter(y_pred, residuals, alpha=0.5, s=30)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.title('残差与预测值的关系')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 2. 残差与真实值的关系
            plt.subplot(2, 2, 2)
            plt.scatter(y_true, residuals, alpha=0.5, s=30)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('残差')
            plt.title('残差与真实值的关系')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 3. 残差分布直方图和密度图
            plt.subplot(2, 2, 3)
            sns.histplot(residuals, kde=True, bins=30)
            plt.axvline(x=0, color='r', linestyle='--', lw=2)
            plt.xlabel('残差')
            plt.ylabel('频率')
            plt.title('残差分布')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 4. 残差QQ图
            plt.subplot(2, 2, 4)
            try:
                import scipy.stats as stats
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('残差QQ图')
                plt.grid(True, linestyle='--', alpha=0.7)
            except ImportError:
                self.logger.warning("无法绘制QQ图，scipy未安装")
                plt.text(0.5, 0.5, '无法绘制QQ图\n(scipy未安装)', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'residual_analysis_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"残差分析图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制残差分析图形时发生异常: {str(e)}")
    
    def create_evaluation_report(self, metrics_dict: Dict, report_type: str = 'regression', 
                                model_name: str = 'Deep Learning Model', 
                                save_path: Optional[str] = None) -> str:
        """创建评估报告
        
        Args:
            metrics_dict: 评估指标字典
            report_type: 报告类型，'regression'或'classification'
            model_name: 模型名称
            save_path: 报告保存路径，如果为None则使用默认路径
        
        Returns:
            报告保存路径
        """
        try:
            self.logger.info("创建评估报告")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置报告保存路径
            if save_path is None:
                results_dir = self.config.get('results_dir', './results')
                save_path = os.path.join(results_dir, f'{report_type}_evaluation_report_{timestamp}.txt')
            
            # 创建报告内容
            report_content = []
            report_content.append("=" * 80)
            report_content.append(f"模型评估报告")
            report_content.append("=" * 80)
            report_content.append(f"模型名称: {model_name}")
            report_content.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"评估类型: {report_type}")
            report_content.append("-" * 80)
            
            # 添加评估指标
            report_content.append("评估指标:")
            for metric, value in metrics_dict.items():
                # 跳过复杂对象
                if isinstance(value, (dict, list, np.ndarray)) and metric not in ['confusion_matrix', 'classification_report']:
                    continue
                
                if isinstance(value, float):
                    report_content.append(f"  {metric}: {value:.6f}")
                elif isinstance(value, np.generic):
                    report_content.append(f"  {metric}: {np.asscalar(value):.6f}")
                else:
                    report_content.append(f"  {metric}: {value}")
            
            report_content.append("=" * 80)
            
            # 保存报告
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            self.logger.info(f"评估报告已保存到: {save_path}")
            
            return save_path
        except Exception as e:
            self.logger.error(f"创建评估报告时发生异常: {str(e)}")
            raise
    
    def compare_models(self, models_results: Dict[str, Dict], 
                      save_results: bool = True, 
                      plot_comparison: bool = True) -> Dict[str, Dict]:
        """比较多个模型的性能
        
        Args:
            models_results: 模型结果字典，键为模型名称，值为评估指标字典
            save_results: 是否保存比较结果
            plot_comparison: 是否绘制比较图形
        
        Returns:
            比较结果字典
        """
        try:
            self.logger.info(f"比较{len(models_results)}个模型的性能")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = self.config.get('results_dir', './results')
            plots_dir = self.config.get('plots_dir', './plots')
            
            # 获取所有指标名称
            all_metrics = set()
            for model_name, metrics in models_results.items():
                for metric in metrics:
                    # 跳过复杂对象
                    if isinstance(metrics[metric], (dict, list, np.ndarray)):
                        continue
                    all_metrics.add(metric)
            
            # 构建比较表格
            comparison_dict = {metric: {} for metric in all_metrics}
            for model_name, metrics in models_results.items():
                for metric in all_metrics:
                    if metric in metrics:
                        comparison_dict[metric][model_name] = metrics[metric]
                    else:
                        comparison_dict[metric][model_name] = None
            
            # 转换为DataFrame便于显示和处理
            comparison_df = pd.DataFrame(comparison_dict)
            
            # 打印比较结果
            self.logger.info("模型性能比较:")
            self.logger.info(f"\n{comparison_df}")
            
            # 保存比较结果
            if save_results:
                # 保存为CSV文件
                csv_path = os.path.join(results_dir, f'models_comparison_{timestamp}.csv')
                comparison_df.to_csv(csv_path, encoding='utf-8', index_label='Model')
                
                # 保存为JSON文件
                json_path = os.path.join(results_dir, f'models_comparison_{timestamp}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(comparison_dict, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"模型比较结果已保存到: {csv_path} 和 {json_path}")
            
            # 绘制比较图形
            if plot_comparison and len(all_metrics) > 0:
                self._plot_models_comparison(comparison_df, timestamp, plots_dir)
            
            return comparison_dict
        except Exception as e:
            self.logger.error(f"比较模型性能时发生异常: {str(e)}")
            raise
    
    def _plot_models_comparison(self, comparison_df: pd.DataFrame, timestamp: str, plots_dir: str) -> None:
        """绘制模型比较图形
        
        Args:
            comparison_df: 比较结果DataFrame
            timestamp: 时间戳
            plots_dir: 图形保存目录
        """
        try:
            self.logger.info("绘制模型比较图形")
            
            # 计算需要的子图数量
            num_metrics = len(comparison_df.columns)
            rows = (num_metrics + 2) // 3  # 每行显示3个指标
            cols = min(num_metrics, 3)
            
            # 创建图形
            plt.figure(figsize=(5 * cols, 4 * rows))
            
            # 为每个指标绘制条形图
            for i, metric in enumerate(comparison_df.columns):
                plt.subplot(rows, cols, i + 1)
                
                # 选择非空数据
                data = comparison_df[metric].dropna()
                
                # 绘制条形图
                data.plot(kind='bar', ax=plt.gca())
                plt.title(metric)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图像
            if self.config.get('save_plots', True):
                plot_file = os.path.join(plots_dir, f'models_comparison_{timestamp}.png')
                plt.savefig(plot_file, dpi=self.config.get('plot_dpi', 300), bbox_inches='tight')
                self.logger.info(f"模型比较图形已保存到: {plot_file}")
            
            # 显示图像
            if self.config.get('show_plots', True):
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制模型比较图形时发生异常: {str(e)}")