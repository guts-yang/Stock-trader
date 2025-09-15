"""集成学习模型模块，实现多种模型的组合和集成"""

import numpy as np
import pandas as pd
import logging
import os
import json
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class EnsembleModel:
    """集成学习模型类，提供多种模型集成方法"""
    
    def __init__(self, models: Dict[str, object], config: Optional[Dict] = None):
        """初始化集成模型
        
        Args:
            models: 模型字典，键为模型名称，值为模型对象
            config: 集成配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'ensemble_type': 'voting',  # 'voting', 'stacking', 'bagging', 'boosting'
            'weights': None,  # 模型权重，如果为None则使用等权重
            'meta_model': None,  # 元模型，用于stacking
            'n_jobs': 1,  # 并行计算的任务数
            'random_state': 42,
            'verbose': 1,
            'save_dir': './models/ensemble',
            'log_dir': './logs',
            'return_train_score': True
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 检查模型是否为空
        if not models:
            raise ValueError("模型字典不能为空")
        
        # 存储模型
        self.models = models
        self.ensemble_model = None
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 创建必要的目录
        self._create_directories()
        
        self.logger.info(f"集成模型初始化完成，包含{len(models)}个基础模型，集成类型: {self.config.get('ensemble_type')}")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('EnsembleModel')
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
            log_file = os.path.join(log_dir, 'ensemble_model.log')
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
            self.config.get('save_dir', './models/ensemble'),
            self.config.get('log_dir', './logs')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.logger.info(f"创建目录: {directory}")
                except Exception as e:
                    self.logger.error(f"创建目录{directory}时发生异常: {str(e)}")
    
    def build_ensemble(self, task_type: str = 'regression') -> None:
        """构建集成模型
        
        Args:
            task_type: 任务类型，'regression'或'classification'
        """
        try:
            self.logger.info(f"构建集成模型，任务类型: {task_type}")
            
            # 获取集成类型和配置
            ensemble_type = self.config.get('ensemble_type', 'voting')
            weights = self.config.get('weights')
            meta_model = self.config.get('meta_model')
            n_jobs = self.config.get('n_jobs', 1)
            random_state = self.config.get('random_state', 42)
            
            # 检查模型列表格式
            model_list = list(self.models.items())
            
            # 根据集成类型创建集成模型
            if ensemble_type == 'voting':
                if task_type == 'regression':
                    self.ensemble_model = VotingRegressor(
                        estimators=model_list,
                        weights=weights,
                        n_jobs=n_jobs,
                        verbose=self.config.get('verbose', 1)
                    )
                else:
                    self.ensemble_model = VotingClassifier(
                        estimators=model_list,
                        weights=weights,
                        voting='soft' if any(hasattr(model, 'predict_proba') for _, model in model_list) else 'hard',
                        n_jobs=n_jobs,
                        verbose=self.config.get('verbose', 1)
                    )
                
            elif ensemble_type == 'stacking':
                # 如果没有提供元模型，使用默认元模型
                if meta_model is None:
                    if task_type == 'regression':
                        from sklearn.linear_model import Ridge
                        meta_model = Ridge(random_state=random_state)
                    else:
                        from sklearn.linear_model import LogisticRegression
                        meta_model = LogisticRegression(random_state=random_state, max_iter=1000)
                    
                if task_type == 'regression':
                    self.ensemble_model = StackingRegressor(
                        estimators=model_list,
                        final_estimator=meta_model,
                        n_jobs=n_jobs,
                        cv=5,
                        passthrough=False,
                        verbose=self.config.get('verbose', 1)
                    )
                else:
                    self.ensemble_model = StackingClassifier(
                        estimators=model_list,
                        final_estimator=meta_model,
                        n_jobs=n_jobs,
                        cv=5,
                        passthrough=False,
                        verbose=self.config.get('verbose', 1)
                    )
            
            elif ensemble_type == 'bagging':
                # 简单的bagging实现，使用Bootstrap采样
                if task_type == 'regression':
                    from sklearn.ensemble import BaggingRegressor
                    # 使用第一个模型作为基础估计器
                    base_estimator = model_list[0][1]
                    self.ensemble_model = BaggingRegressor(
                        estimator=base_estimator,
                        n_estimators=len(model_list),
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        oob_score=True,
                        warm_start=False,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=self.config.get('verbose', 1)
                    )
                else:
                    from sklearn.ensemble import BaggingClassifier
                    # 使用第一个模型作为基础估计器
                    base_estimator = model_list[0][1]
                    self.ensemble_model = BaggingClassifier(
                        estimator=base_estimator,
                        n_estimators=len(model_list),
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        oob_score=True,
                        warm_start=False,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=self.config.get('verbose', 1)
                    )
            
            elif ensemble_type == 'boosting':
                # 注意：这里使用的是堆叠式的boosting实现，而不是传统的提升方法
                self.logger.warning("boosting类型的集成模型使用堆叠式实现，不是传统的提升方法")
                self._build_boosting_ensemble(task_type)
            
            else:
                raise ValueError(f"不支持的集成类型: {ensemble_type}")
            
            self.logger.info(f"集成模型构建完成: {self.ensemble_model.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"构建集成模型时发生异常: {str(e)}")
            raise
    
    def _build_boosting_ensemble(self, task_type: str) -> None:
        """构建堆叠式提升集成模型
        
        Args:
            task_type: 任务类型，'regression'或'classification'
        """
        try:
            # 为了简化，这里使用自定义的StackingBoostingModel类
            if task_type == 'regression':
                self.ensemble_model = StackingBoostingRegressor(
                    models=list(self.models.values()),
                    weights=self.config.get('weights'),
                    random_state=self.config.get('random_state', 42)
                )
            else:
                self.ensemble_model = StackingBoostingClassifier(
                    models=list(self.models.values()),
                    weights=self.config.get('weights'),
                    random_state=self.config.get('random_state', 42)
                )
        except Exception as e:
            self.logger.error(f"构建提升集成模型时发生异常: {str(e)}")
            raise
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> None:
        """训练集成模型
        
        Args:
            X: 特征数据
            y: 目标数据
            **fit_params: 传递给底层模型的额外参数
        """
        try:
            self.logger.info("开始训练集成模型")
            
            # 检查集成模型是否已构建
            if self.ensemble_model is None:
                # 自动判断任务类型
                if len(np.unique(y)) > 10 or np.issubdtype(y.dtype, np.floating):
                    task_type = 'regression'
                else:
                    task_type = 'classification'
                
                self.logger.info(f"自动检测任务类型: {task_type}")
                self.build_ensemble(task_type)
            
            # 记录训练开始时间
            start_time = datetime.now()
            
            # 训练集成模型
            self.ensemble_model.fit(X, y, **fit_params)
            
            # 记录训练结束时间
            end_time = datetime.now()
            training_time = end_time - start_time
            
            self.logger.info(f"集成模型训练完成，耗时: {training_time}")
        except Exception as e:
            self.logger.error(f"训练集成模型时发生异常: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用集成模型进行预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果数组
        """
        try:
            self.logger.info("使用集成模型进行预测")
            
            # 检查集成模型是否已构建
            if self.ensemble_model is None:
                raise ValueError("集成模型尚未构建，请先调用build_ensemble或fit方法")
            
            # 执行预测
            y_pred = self.ensemble_model.predict(X)
            
            self.logger.info(f"预测完成，预测样本数: {len(y_pred)}")
            
            return y_pred
        except Exception as e:
            self.logger.error(f"使用集成模型进行预测时发生异常: {str(e)}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """使用集成模型进行概率预测（分类任务）
        
        Args:
            X: 特征数据
        
        Returns:
            预测概率数组
        """
        try:
            self.logger.info("使用集成模型进行概率预测")
            
            # 检查集成模型是否已构建
            if self.ensemble_model is None:
                raise ValueError("集成模型尚未构建，请先调用build_ensemble或fit方法")
            
            # 检查模型是否支持概率预测
            if not hasattr(self.ensemble_model, 'predict_proba'):
                self.logger.warning("集成模型不支持概率预测，返回预测标签")
                y_pred = self.ensemble_model.predict(X)
                # 转换为概率格式（简单的one-hot编码）
                if len(y_pred.shape) == 1:
                    classes = np.unique(y_pred)
                    if len(classes) == 2:
                        # 二分类问题
                        y_pred_proba = np.zeros((len(y_pred), 2))
                        y_pred_proba[:, 1] = y_pred
                        y_pred_proba[:, 0] = 1 - y_pred
                    else:
                        # 多分类问题
                        y_pred_proba = np.zeros((len(y_pred), len(classes)))
                        for i, cls in enumerate(classes):
                            y_pred_proba[y_pred == cls, i] = 1
                return y_pred_proba
            
            # 执行概率预测
            y_pred_proba = self.ensemble_model.predict_proba(X)
            
            self.logger.info(f"概率预测完成，预测样本数: {len(y_pred_proba)}")
            
            return y_pred_proba
        except Exception as e:
            self.logger.error(f"使用集成模型进行概率预测时发生异常: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, task_type: str = 'regression') -> Dict[str, float]:
        """评估集成模型性能
        
        Args:
            X: 特征数据
            y: 目标数据
            task_type: 任务类型，'regression'或'classification'
        
        Returns:
            评估指标字典
        """
        try:
            self.logger.info(f"评估集成模型性能，任务类型: {task_type}")
            
            # 执行预测
            y_pred = self.predict(X)
            
            # 计算评估指标
            metrics_dict = {}
            
            if task_type == 'regression':
                # 回归指标
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                metrics_dict['mae'] = float(mean_absolute_error(y, y_pred))
                metrics_dict['mse'] = float(mean_squared_error(y, y_pred))
                metrics_dict['rmse'] = float(np.sqrt(metrics_dict['mse']))
                metrics_dict['r2'] = float(r2_score(y, y_pred))
                
                # 平均绝对百分比误差 (MAPE)
                non_zero_mask = y != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
                else:
                    mape = 0
                metrics_dict['mape'] = float(mape)
                
                # 方向准确率（预测涨跌方向的准确率）
                if len(y) > 1:
                    # 计算真实方向变化
                    true_direction = np.sign(y[1:] - y[:-1])
                    # 计算预测方向变化
                    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
                    # 计算方向准确率
                    direction_accuracy = np.mean(true_direction == pred_direction)
                    metrics_dict['direction_accuracy'] = float(direction_accuracy)
            
            else:
                # 分类指标
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    confusion_matrix, classification_report, roc_auc_score
                )
                
                metrics_dict['accuracy'] = float(accuracy_score(y, y_pred))
                metrics_dict['precision'] = float(precision_score(y, y_pred, average='weighted', zero_division=0))
                metrics_dict['recall'] = float(recall_score(y, y_pred, average='weighted', zero_division=0))
                metrics_dict['f1'] = float(f1_score(y, y_pred, average='weighted', zero_division=0))
                
                # 混淆矩阵
                cm = confusion_matrix(y, y_pred)
                metrics_dict['confusion_matrix'] = cm.tolist()
                
                # 分类报告
                class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                metrics_dict['classification_report'] = class_report
                
                # 尝试计算ROC AUC
                try:
                    # 检查是否支持概率预测
                    if hasattr(self.ensemble_model, 'predict_proba'):
                        y_pred_proba = self.predict_proba(X)
                        if y_pred_proba.shape[1] > 1:
                            # 多类别分类
                            roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr')
                        else:
                            # 二分类
                            roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
                        metrics_dict['roc_auc'] = float(roc_auc)
                except Exception as e:
                    self.logger.warning(f"无法计算ROC AUC: {str(e)}")
            
            # 打印评估结果
            self.logger.info(f"集成模型评估结果: {metrics_dict}")
            
            # 保存评估结果
            self._save_evaluation_results(metrics_dict)
            
            return metrics_dict
        except Exception as e:
            self.logger.error(f"评估集成模型性能时发生异常: {str(e)}")
            raise
    
    def _save_evaluation_results(self, metrics_dict: Dict) -> None:
        """保存评估结果到文件
        
        Args:
            metrics_dict: 评估指标字典
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON文件
            results_dir = os.path.join(self.config.get('log_dir', './logs'), 'ensemble')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            result_file = os.path.join(results_dir, f'ensemble_evaluation_{timestamp}.json')
            
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
            
            self.logger.info(f"集成模型评估结果已保存到: {result_file}")
        except Exception as e:
            self.logger.error(f"保存集成模型评估结果时发生异常: {str(e)}")
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """保存集成模型到文件
        
        Args:
            save_path: 模型保存路径，如果为None则使用默认路径
        
        Returns:
            实际的模型保存路径
        """
        try:
            self.logger.info("保存集成模型")
            
            # 检查集成模型是否已构建
            if self.ensemble_model is None:
                raise ValueError("集成模型尚未构建，请先调用build_ensemble或fit方法")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if save_path is None:
                save_dir = self.config.get('save_dir', './models/ensemble')
                save_path = os.path.join(save_dir, f'ensemble_model_{timestamp}')
            
            # 创建保存目录
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            
            # 根据模型类型选择保存方法
            if isinstance(self.ensemble_model, (VotingRegressor, VotingClassifier, 
                                              StackingRegressor, StackingClassifier)):
                # 使用joblib保存scikit-learn模型
                import joblib
                joblib_path = f"{save_path}.joblib"
                joblib.dump(self.ensemble_model, joblib_path)
                self.logger.info(f"集成模型已保存到: {joblib_path}")
                
                # 保存配置文件
                config_path = f"{save_path}_config.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                self.logger.info(f"集成模型配置已保存到: {config_path}")
                
                return joblib_path
            
            elif isinstance(self.ensemble_model, tf.keras.Model):
                # 保存Keras模型
                self.ensemble_model.save(save_path)
                self.logger.info(f"集成模型已保存到: {save_path}")
                
                # 保存配置文件
                config_path = f"{save_path}_config.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                self.logger.info(f"集成模型配置已保存到: {config_path}")
                
                return save_path
            
            else:
                # 尝试使用pickle保存
                import pickle
                pickle_path = f"{save_path}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
                self.logger.info(f"集成模型已保存到: {pickle_path}")
                
                # 保存配置文件
                config_path = f"{save_path}_config.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                self.logger.info(f"集成模型配置已保存到: {config_path}")
                
                return pickle_path
        except Exception as e:
            self.logger.error(f"保存集成模型时发生异常: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> bool:
        """从文件加载集成模型
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载集成模型: {model_path}")
            
            # 尝试加载配置文件
            config_path = model_path.replace('.joblib', '_config.json').replace('.pkl', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"集成模型配置已从{config_path}加载")
            
            # 根据文件扩展名选择加载方法
            if model_path.endswith('.joblib'):
                # 加载joblib格式的模型
                import joblib
                self.ensemble_model = joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                # 加载pickle格式的模型
                import pickle
                with open(model_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            else:
                # 尝试加载Keras模型
                try:
                    self.ensemble_model = tf.keras.models.load_model(model_path)
                except Exception as e:
                    self.logger.error(f"加载Keras模型失败: {str(e)}")
                    return False
            
            self.logger.info("集成模型加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载集成模型时发生异常: {str(e)}")
            return False
    
    def get_model_weights(self) -> Dict[str, float]:
        """获取集成模型中各基础模型的权重
        
        Returns:
            模型权重字典
        """
        try:
            self.logger.info("获取集成模型中各基础模型的权重")
            
            # 检查集成模型是否已构建
            if self.ensemble_model is None:
                raise ValueError("集成模型尚未构建，请先调用build_ensemble或fit方法")
            
            weights = {}
            
            # 尝试获取权重
            if hasattr(self.ensemble_model, 'weights') and self.ensemble_model.weights is not None:
                # VotingRegressor/VotingClassifier的权重
                if len(self.ensemble_model.weights) == len(self.models):
                    for i, model_name in enumerate(self.models.keys()):
                        weights[model_name] = float(self.ensemble_model.weights[i])
            
            elif hasattr(self.ensemble_model, 'estimators_'):
                # StackingRegressor/StackingClassifier或其他集成模型
                if len(self.ensemble_model.estimators_) == len(self.models):
                    # 如果配置中有权重，则使用配置中的权重
                    if self.config.get('weights') is not None and len(self.config.get('weights')) == len(self.models):
                        for i, model_name in enumerate(self.models.keys()):
                            weights[model_name] = float(self.config.get('weights')[i])
                    else:
                        # 否则使用等权重
                        for model_name in self.models.keys():
                            weights[model_name] = 1.0 / len(self.models)
            
            else:
                # 默认使用等权重
                for model_name in self.models.keys():
                    weights[model_name] = 1.0 / len(self.models)
            
            self.logger.info(f"集成模型权重: {weights}")
            
            return weights
        except Exception as e:
            self.logger.error(f"获取集成模型权重时发生异常: {str(e)}")
            raise
    
    def plot_model_contribution(self, X: np.ndarray, y: np.ndarray, 
                               task_type: str = 'regression', 
                               save_path: Optional[str] = None) -> None:
        """绘制各基础模型对集成模型的贡献
        
        Args:
            X: 特征数据
            y: 目标数据
            task_type: 任务类型，'regression'或'classification'
            save_path: 图像保存路径，如果为None则使用默认路径
        """
        try:
            self.logger.info("绘制各基础模型对集成模型的贡献")
            
            # 计算各基础模型的性能
            base_models_performance = {}
            for model_name, model in self.models.items():
                try:
                    if task_type == 'regression':
                        from sklearn.metrics import mean_absolute_error
                        y_pred = model.predict(X)
                        mae = mean_absolute_error(y, y_pred)
                        base_models_performance[model_name] = 1.0 / mae  # 转换为分数，MAE越小，分数越高
                    else:
                        from sklearn.metrics import accuracy_score
                        y_pred = model.predict(X)
                        accuracy = accuracy_score(y, y_pred)
                        base_models_performance[model_name] = accuracy
                except Exception as e:
                    self.logger.warning(f"计算模型{model_name}性能时发生异常: {str(e)}")
                    base_models_performance[model_name] = 0.0
            
            # 计算集成模型的性能
            try:
                if task_type == 'regression':
                    from sklearn.metrics import mean_absolute_error
                    y_pred_ensemble = self.predict(X)
                    ensemble_performance = 1.0 / mean_absolute_error(y, y_pred_ensemble)
                else:
                    from sklearn.metrics import accuracy_score
                    y_pred_ensemble = self.predict(X)
                    ensemble_performance = accuracy_score(y, y_pred_ensemble)
            except Exception as e:
                self.logger.warning(f"计算集成模型性能时发生异常: {str(e)}")
                ensemble_performance = 0.0
            
            # 计算相对贡献
            total_performance = sum(base_models_performance.values())
            relative_contributions = {}
            if total_performance > 0:
                for model_name, performance in base_models_performance.items():
                    relative_contributions[model_name] = performance / total_performance
            
            # 获取模型权重
            model_weights = self.get_model_weights()
            
            # 绘制图形
            self._plot_contribution(relative_contributions, model_weights, ensemble_performance, save_path)
        except Exception as e:
            self.logger.error(f"绘制模型贡献时发生异常: {str(e)}")
    
    def _plot_contribution(self, contributions: Dict[str, float], weights: Dict[str, float], 
                          ensemble_performance: float, save_path: Optional[str] = None) -> None:
        """绘制模型贡献图形
        
        Args:
            contributions: 各模型的贡献度
            weights: 各模型的权重
            ensemble_performance: 集成模型的性能
            save_path: 图像保存路径
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if save_path is None:
                plots_dir = os.path.join(self.config.get('log_dir', './logs'), 'ensemble')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                save_path = os.path.join(plots_dir, f'model_contributions_{timestamp}.png')
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 设置数据
            model_names = list(contributions.keys())
            contribution_values = list(contributions.values())
            weight_values = [weights.get(name, 0.0) for name in model_names]
            
            # 创建索引
            x = np.arange(len(model_names))
            width = 0.35  # 条形图宽度
            
            # 绘制条形图
            plt.bar(x - width/2, contribution_values, width, label='相对贡献')
            plt.bar(x + width/2, weight_values, width, label='模型权重')
            
            # 设置图表属性
            plt.ylabel('比例')
            plt.title(f'各基础模型对集成模型的贡献 (集成性能: {ensemble_performance:.4f})')
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"模型贡献图形已保存到: {save_path}")
            
            # 显示图像
            if self.config.get('verbose', 1) > 0:
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制模型贡献图形时发生异常: {str(e)}")
    
    def get_base_models_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有基础模型的预测结果
        
        Args:
            X: 特征数据
        
        Returns:
            各基础模型的预测结果字典
        """
        try:
            self.logger.info("获取所有基础模型的预测结果")
            
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    y_pred = model.predict(X)
                    predictions[model_name] = y_pred
                except Exception as e:
                    self.logger.warning(f"模型{model_name}预测时发生异常: {str(e)}")
                    predictions[model_name] = None
            
            self.logger.info(f"获取了{len([p for p in predictions.values() if p is not None])}个基础模型的预测结果")
            
            return predictions
        except Exception as e:
            self.logger.error(f"获取基础模型预测结果时发生异常: {str(e)}")
            raise
    
    def plot_base_models_predictions(self, X: np.ndarray, y_true: Optional[np.ndarray] = None, 
                                   sample_indices: Optional[List[int]] = None, 
                                   save_path: Optional[str] = None) -> None:
        """绘制各基础模型的预测结果对比图
        
        Args:
            X: 特征数据
            y_true: 真实值数组，如果为None则不显示真实值
            sample_indices: 要显示的样本索引列表，如果为None则显示前5个样本
            save_path: 图像保存路径，如果为None则使用默认路径
        """
        try:
            self.logger.info("绘制各基础模型的预测结果对比图")
            
            # 获取所有基础模型的预测结果
            base_predictions = self.get_base_models_predictions(X)
            
            # 获取集成模型的预测结果
            ensemble_prediction = self.predict(X)
            
            # 选择要显示的样本
            if sample_indices is None:
                sample_indices = range(min(5, len(X)))
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if save_path is None:
                plots_dir = os.path.join(self.config.get('log_dir', './logs'), 'ensemble')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                save_path = os.path.join(plots_dir, f'predictions_comparison_{timestamp}.png')
            
            # 创建图形
            plt.figure(figsize=(15, 5 * len(sample_indices)))
            
            # 为每个样本绘制对比图
            for i, idx in enumerate(sample_indices):
                plt.subplot(len(sample_indices), 1, i + 1)
                
                # 获取该样本的所有预测结果
                model_names = []
                predictions = []
                
                # 添加基础模型的预测结果
                for model_name, pred in base_predictions.items():
                    if pred is not None:
                        model_names.append(model_name)
                        # 确保获取单个值
                        if len(pred.shape) > 1 and pred.shape[1] > 1:
                            # 多分类问题，取预测概率最高的类别
                            predictions.append(np.argmax(pred[idx]))
                        else:
                            predictions.append(pred[idx])
                
                # 添加集成模型的预测结果
                model_names.append('集成模型')
                if len(ensemble_prediction.shape) > 1 and ensemble_prediction.shape[1] > 1:
                    # 多分类问题，取预测概率最高的类别
                    predictions.append(np.argmax(ensemble_prediction[idx]))
                else:
                    predictions.append(ensemble_prediction[idx])
                
                # 绘制条形图
                bars = plt.bar(model_names, predictions, color='skyblue')
                
                # 如果有真实值，绘制参考线
                if y_true is not None:
                    true_value = y_true[idx]
                    plt.axhline(y=true_value, color='r', linestyle='--', label=f'真实值: {true_value}')
                    plt.legend()
                
                # 设置图表属性
                plt.title(f'样本 {idx} 的预测结果对比')
                plt.ylabel('预测值')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 在条形图上显示数值
                for bar, pred in zip(bars, predictions):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height, 
                            f'{pred:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"预测结果对比图形已保存到: {save_path}")
            
            # 显示图像
            if self.config.get('verbose', 1) > 0:
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制基础模型预测结果对比图时发生异常: {str(e)}")
    
    def calculate_diversity(self, X: np.ndarray, y: np.ndarray, 
                           task_type: str = 'regression') -> Dict[str, float]:
        """计算基础模型之间的多样性
        
        Args:
            X: 特征数据
            y: 目标数据
            task_type: 任务类型，'regression'或'classification'
        
        Returns:
            多样性指标字典
        """
        try:
            self.logger.info("计算基础模型之间的多样性")
            
            # 获取所有基础模型的预测结果
            base_predictions = self.get_base_models_predictions(X)
            
            # 过滤掉预测失败的模型
            valid_predictions = {name: pred for name, pred in base_predictions.items() if pred is not None}
            model_names = list(valid_predictions.keys())
            
            if len(valid_predictions) < 2:
                self.logger.warning("基础模型数量不足，无法计算多样性")
                return {}
            
            diversity_metrics = {}
            
            if task_type == 'regression':
                # 回归任务的多样性指标
                
                # 1. 预测结果的相关性
                corr_matrix = np.zeros((len(model_names), len(model_names)))
                for i in range(len(model_names)):
                    for j in range(len(model_names)):
                        if i <= j:
                            corr = np.corrcoef(valid_predictions[model_names[i]], 
                                             valid_predictions[model_names[j]])[0, 1]
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
                
                # 平均相关性（排除对角线）
                avg_corr = np.mean(corr_matrix[np.triu_indices(len(model_names), k=1)])
                diversity_metrics['average_correlation'] = float(avg_corr)
                diversity_metrics['diversity_score'] = float(1 - avg_corr)  # 多样性分数
                
                # 2. Q统计量 (用于测量模型间的相关性)
                # 计算残差
                residuals = {}
                for name, pred in valid_predictions.items():
                    residuals[name] = y - pred
                
                # 计算Q统计量
                q_statistics = []
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        # 计算两个模型残差的协方差和方差
                        cov = np.cov(residuals[model_names[i]], residuals[model_names[j]])[0, 1]
                        var1 = np.var(residuals[model_names[i]])
                        var2 = np.var(residuals[model_names[j]])
                        
                        if var1 > 0 and var2 > 0:
                            q = cov / np.sqrt(var1 * var2)
                            q_statistics.append(q)
                
                if q_statistics:
                    diversity_metrics['mean_q_statistic'] = float(np.mean(q_statistics))
        
            else:
                # 分类任务的多样性指标
                
                # 1. 预测结果的一致性
                # 计算每对模型的一致性
                agreement_matrix = np.zeros((len(model_names), len(model_names)))
                for i in range(len(model_names)):
                    for j in range(len(model_names)):
                        if i <= j:
                            # 确保预测结果是相同形状的一维数组
                            pred_i = valid_predictions[model_names[i]].squeeze()
                            pred_j = valid_predictions[model_names[j]].squeeze()
                            
                            # 计算一致性比例
                            agreement = np.mean(pred_i == pred_j)
                            agreement_matrix[i, j] = agreement
                            agreement_matrix[j, i] = agreement
                
                # 平均一致性（排除对角线）
                avg_agreement = np.mean(agreement_matrix[np.triu_indices(len(model_names), k=1)])
                diversity_metrics['average_agreement'] = float(avg_agreement)
                diversity_metrics['diversity_score'] = float(1 - avg_agreement)  # 多样性分数
                
                # 2. Q统计量 (用于测量模型间的相关性)
                q_statistics = []
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        # 计算混淆矩阵
                        pred_i = valid_predictions[model_names[i]].squeeze()
                        pred_j = valid_predictions[model_names[j]].squeeze()
                        
                        # 创建2x2的混淆矩阵
                        a = np.sum((pred_i == 1) & (pred_j == 1))
                        b = np.sum((pred_i == 1) & (pred_j == 0))
                        c = np.sum((pred_i == 0) & (pred_j == 1))
                        d = np.sum((pred_i == 0) & (pred_j == 0))
                        
                        # 计算Q统计量
                        if (a + d) + (b + c) > 0:
                            q = ((a * d) - (b * c)) / ((a * d) + (b * c))
                            q_statistics.append(q)
                
                if q_statistics:
                    diversity_metrics['mean_q_statistic'] = float(np.mean(q_statistics))
                
                # 3. 熵度量 (Entropy Measure)
                # 将所有模型的预测结果组合成投票
                all_predictions = np.array([valid_predictions[name].squeeze() for name in model_names])
                
                # 计算每个样本的熵
                entropies = []
                for sample_idx in range(all_predictions.shape[1]):
                    # 获取该样本的所有预测
                    sample_preds = all_predictions[:, sample_idx]
                    # 计算类别分布
                    unique, counts = np.unique(sample_preds, return_counts=True)
                    probs = counts / len(sample_preds)
                    # 计算熵
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))  # 加一个小值避免log(0)
                    entropies.append(entropy)
                
                # 计算平均熵
                avg_entropy = np.mean(entropies)
                diversity_metrics['average_entropy'] = float(avg_entropy)
        
            self.logger.info(f"基础模型多样性指标: {diversity_metrics}")
        
            return diversity_metrics
        except Exception as e:
            self.logger.error(f"计算基础模型多样性时发生异常: {str(e)}")
            raise

# 自定义的堆叠式提升回归器
class StackingBoostingRegressor(BaseEstimator, RegressorMixin):
    """堆叠式提升回归器
    一种简单的提升方法，通过迭代训练模型并调整样本权重
    """
    
    def __init__(self, models: List[object], weights: Optional[List[float]] = None, 
                random_state: int = 42):
        self.models = models
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        # 确保y是一维数组
        if y.ndim > 1:
            y = y.squeeze()
        
        # 如果没有提供权重，使用等权重
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # 标准化权重
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        # 训练所有模型
        self.trained_models = []
        for model in self.models:
            # 训练模型
            model.fit(X, y, **fit_params)
            self.trained_models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 确保模型已训练
        if not hasattr(self, 'trained_models'):
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取所有模型的预测结果
        predictions = []
        for model in self.trained_models:
            pred = model.predict(X)
            if pred.ndim > 1:
                pred = pred.squeeze()
            predictions.append(pred)
        
        # 加权平均预测结果
        y_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            y_pred += self.weights[i] * pred
        
        return y_pred

# 自定义的堆叠式提升分类器
class StackingBoostingClassifier(BaseEstimator, ClassifierMixin):
    """堆叠式提升分类器
    一种简单的提升方法，通过迭代训练模型并调整样本权重
    """
    
    def __init__(self, models: List[object], weights: Optional[List[float]] = None, 
                random_state: int = 42):
        self.models = models
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        # 确保y是一维数组
        if y.ndim > 1:
            y = y.squeeze()
        
        # 存储类别信息
        self.classes_ = np.unique(y)
        
        # 如果没有提供权重，使用等权重
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # 标准化权重
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        # 训练所有模型
        self.trained_models = []
        for model in self.models:
            # 训练模型
            model.fit(X, y, **fit_params)
            self.trained_models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 确保模型已训练
        if not hasattr(self, 'trained_models'):
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取预测概率
        try:
            y_pred_proba = self.predict_proba(X)
            # 选择概率最高的类别
            y_pred = self.classes_[np.argmax(y_pred_proba, axis=1)]
        except Exception:
            # 如果无法获取概率，使用多数投票
            predictions = []
            for model in self.trained_models:
                pred = model.predict(X)
                if pred.ndim > 1:
                    pred = pred.squeeze()
                predictions.append(pred)
            
            # 加权投票（简单实现）
            y_pred = np.zeros(len(X), dtype=self.classes_.dtype)
            for i in range(len(X)):
                vote_counts = {cls: 0 for cls in self.classes_}
                for j, pred in enumerate(predictions):
                    if pred[i] in vote_counts:
                        vote_counts[pred[i]] += self.weights[j]
                
                # 选择得票最多的类别
                y_pred[i] = max(vote_counts.items(), key=lambda x: x[1])[0]
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # 确保模型已训练
        if not hasattr(self, 'trained_models'):
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取所有模型的预测概率
        probas = []
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probas.append(prob)
            else:
                # 如果模型不支持概率预测，使用预测结果生成伪概率
                pred = model.predict(X)
                prob = np.zeros((len(pred), len(self.classes_)))
                for i, cls in enumerate(self.classes_):
                    prob[pred == cls, i] = 1
                probas.append(prob)
        
        # 加权平均概率
        y_pred_proba = np.zeros_like(probas[0])
        for i, prob in enumerate(probas):
            y_pred_proba += self.weights[i] * prob
        
        # 归一化概率
        row_sums = y_pred_proba.sum(axis=1, keepdims=True)
        y_pred_proba = np.divide(y_pred_proba, row_sums, where=row_sums != 0)
        
        return y_pred_proba