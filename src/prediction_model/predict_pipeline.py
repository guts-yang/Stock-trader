"""预测流水线模块，实现从特征提取到预测结果输出的完整流程"""

import numpy as np
import pandas as pd
import logging
import os
import json
import joblib
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PredictPipeline:
    """预测流水线类，实现从特征提取到预测结果输出的完整流程"""
    
    def __init__(self, data_processor=None, feature_selector=None, 
                model=None, model_trainer=None, model_evaluator=None, 
                config: Optional[Dict] = None):
        """初始化预测流水线
        
        Args:
            data_processor: 数据处理器实例
            feature_selector: 特征选择器实例
            model: 模型实例
            model_trainer: 模型训练器实例
            model_evaluator: 模型评估器实例
            config: 配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'data_dir': './data',
            'models_dir': './models',
            'features_dir': './features',
            'results_dir': './results',
            'logs_dir': './logs',
            'random_state': 42,
            'batch_size': 64,
            'verbose': 1,
            'cache_features': True,
            'feature_cache_dir': './cache/features',
            'use_early_stopping': True,
            'early_stopping_patience': 10,
            'save_predictions': True,
            'save_figures': True,
            'figures_dir': './figures'
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 存储各个组件
        self.data_processor = data_processor
        self.feature_selector = feature_selector
        self.model = model
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 创建必要的目录
        self._create_directories()
        
        # 存储流程状态
        self.is_trained = False
        self.is_feature_extracted = False
        self.training_history = None
        self.evaluation_results = None
        
        self.logger.info("预测流水线初始化完成")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('PredictPipeline')
        log_dir = self.config.get('logs_dir', './logs')
        
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
            log_file = os.path.join(log_dir, 'predict_pipeline.log')
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
            self.config.get('data_dir', './data'),
            self.config.get('models_dir', './models'),
            self.config.get('features_dir', './features'),
            self.config.get('results_dir', './results'),
            self.config.get('logs_dir', './logs'),
            self.config.get('feature_cache_dir', './cache/features'),
            self.config.get('figures_dir', './figures')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.logger.info(f"创建目录: {directory}")
                except Exception as e:
                    self.logger.error(f"创建目录{directory}时发生异常: {str(e)}")
    
    def load_data(self, data_path: str, 
                  feature_columns: Optional[List[str]] = None, 
                  target_column: str = 'close', 
                  date_column: str = 'date') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """加载和预处理数据
        
        Args:
            data_path: 数据文件路径
            feature_columns: 特征列名列表，如果为None则使用所有非目标列
            target_column: 目标列名
            date_column: 日期列名
        
        Returns:
            处理后的DataFrame, 特征数据X, 目标数据y
        """
        try:
            self.logger.info(f"加载数据: {data_path}")
            
            # 读取数据
            file_ext = os.path.splitext(data_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(data_path)
            elif file_ext == '.pkl':
                df = pd.read_pickle(data_path)
            elif file_ext == '.xlsx':
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            self.logger.info(f"数据加载完成，共{len(df)}行，{len(df.columns)}列")
            
            # 转换日期列
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.sort_values(by=date_column, inplace=True)
                self.logger.info(f"日期列{date_column}已转换并排序")
            
            # 检查目标列是否存在
            if target_column not in df.columns:
                raise ValueError(f"目标列{target_column}不存在于数据中")
            
            # 提取特征列
            if feature_columns is None:
                # 排除目标列和日期列作为特征
                feature_columns = [col for col in df.columns if col != target_column and col != date_column]
            
            # 检查特征列是否都存在
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"以下特征列不存在于数据中: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in df.columns]
            
            # 提取特征和目标
            X = df[feature_columns].values
            y = df[target_column].values
            
            self.logger.info(f"特征提取完成，X形状: {X.shape}, y形状: {y.shape}")
            
            return df, X, y
        except Exception as e:
            self.logger.error(f"加载数据时发生异常: {str(e)}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       is_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据
        
        Args:
            X: 特征数据
            y: 目标数据
            is_train: 是否为训练数据
        
        Returns:
            预处理后的特征数据和目标数据
        """
        try:
            self.logger.info(f"预处理数据，是否为训练数据: {is_train}")
            
            # 检查是否有数据处理器
            if self.data_processor is None:
                self.logger.warning("未提供数据处理器，跳过预处理步骤")
                return X, y
            
            # 调用数据处理器进行预处理
            X_processed, y_processed = self.data_processor.process(X, y, is_train=is_train)
            
            self.logger.info(f"数据预处理完成，X形状: {X_processed.shape}, y形状: {y_processed.shape}")
            
            return X_processed, y_processed
        except Exception as e:
            self.logger.error(f"预处理数据时发生异常: {str(e)}")
            raise
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       is_train: bool = True) -> np.ndarray:
        """进行特征选择
        
        Args:
            X: 特征数据
            y: 目标数据
            is_train: 是否为训练数据
        
        Returns:
            选择后的特征数据
        """
        try:
            self.logger.info(f"进行特征选择，是否为训练数据: {is_train}")
            
            # 检查是否有特征选择器
            if self.feature_selector is None:
                self.logger.warning("未提供特征选择器，跳过特征选择步骤")
                return X
            
            # 调用特征选择器进行特征选择
            X_selected = self.feature_selector.select(X, y, is_train=is_train)
            
            self.logger.info(f"特征选择完成，X形状: {X_selected.shape}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"进行特征选择时发生异常: {str(e)}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  train_ratio: float = 0.7, 
                  valid_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray, np.ndarray]:
        """划分训练集、验证集和测试集
        
        Args:
            X: 特征数据
            y: 目标数据
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
        
        Returns:
            训练集、验证集、测试集的特征和目标数据
        """
        try:
            self.logger.info(f"划分训练集、验证集和测试集，训练集比例: {train_ratio}, 验证集比例: {valid_ratio}")
            
            # 计算分割点
            total_size = len(X)
            train_size = int(total_size * train_ratio)
            valid_size = int(total_size * valid_ratio)
            
            # 分割数据
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_valid = X[train_size:train_size+valid_size]
            y_valid = y[train_size:train_size+valid_size]
            
            X_test = X[train_size+valid_size:]
            y_test = y[train_size+valid_size:]
            
            self.logger.info(f"数据分割完成，训练集: {len(X_train)}, 验证集: {len(X_valid)}, 测试集: {len(X_test)}")
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        except Exception as e:
            self.logger.error(f"划分数据时发生异常: {str(e)}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_valid: Optional[np.ndarray] = None, 
              y_valid: Optional[np.ndarray] = None, 
              **train_params) -> object:
        """训练模型
        
        Args:
            X: 训练集特征数据
            y: 训练集目标数据
            X_valid: 验证集特征数据
            y_valid: 验证集目标数据
            **train_params: 传递给模型训练的额外参数
        
        Returns:
            训练好的模型
        """
        try:
            self.logger.info("开始训练模型")
            
            # 记录训练开始时间
            start_time = time.time()
            
            # 检查是否有模型
            if self.model is None:
                raise ValueError("未提供模型实例，无法进行训练")
            
            # 如果有模型训练器，使用模型训练器进行训练
            if self.model_trainer is not None:
                # 设置训练参数
                train_params.setdefault('batch_size', self.config.get('batch_size', 64))
                train_params.setdefault('use_early_stopping', self.config.get('use_early_stopping', True))
                train_params.setdefault('early_stopping_patience', self.config.get('early_stopping_patience', 10))
                
                # 训练模型
                self.training_history = self.model_trainer.train(
                    self.model, X, y, X_valid, y_valid, **train_params
                )
            else:
                # 直接调用模型的fit方法
                if X_valid is not None and y_valid is not None:
                    # 如果有验证集
                    self.training_history = self.model.fit(X, y, validation_data=(X_valid, y_valid), **train_params)
                else:
                    # 没有验证集
                    self.training_history = self.model.fit(X, y, **train_params)
            
            # 标记为已训练
            self.is_trained = True
            
            # 记录训练结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            
            # 保存训练历史
            if self.training_history is not None and self.config.get('save_predictions', True):
                self._save_training_history()
            
            # 绘制训练历史
            if self.training_history is not None and self.config.get('save_figures', True):
                self.plot_training_history()
            
            return self.model
        except Exception as e:
            self.logger.error(f"训练模型时发生异常: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        try:
            self.logger.info("使用模型进行预测")
            
            # 检查模型是否已训练
            if not self.is_trained:
                self.logger.warning("模型尚未训练，尝试直接进行预测")
            
            # 检查是否有模型
            if self.model is None:
                raise ValueError("未提供模型实例，无法进行预测")
            
            # 进行预测
            y_pred = self.model.predict(X)
            
            self.logger.info(f"预测完成，预测样本数: {len(y_pred)}")
            
            # 如果启用了预测结果保存
            if self.config.get('save_predictions', True):
                self._save_predictions(y_pred)
            
            return y_pred
        except Exception as e:
            self.logger.error(f"预测时发生异常: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                task_type: str = 'regression', 
                **eval_params) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            X: 特征数据
            y: 目标数据
            task_type: 任务类型，'regression'或'classification'
            **eval_params: 传递给评估器的额外参数
        
        Returns:
            评估指标字典
        """
        try:
            self.logger.info(f"评估模型性能，任务类型: {task_type}")
            
            # 检查是否有模型
            if self.model is None:
                raise ValueError("未提供模型实例，无法进行评估")
            
            # 如果有模型评估器，使用模型评估器进行评估
            if self.model_evaluator is not None:
                self.evaluation_results = self.model_evaluator.evaluate(
                    self.model, X, y, task_type=task_type, **eval_params
                )
            else:
                # 直接计算评估指标
                y_pred = self.predict(X)
                self.evaluation_results = self._calculate_metrics(y, y_pred, task_type)
            
            self.logger.info(f"模型评估完成，评估结果: {self.evaluation_results}")
            
            # 保存评估结果
            if self.config.get('save_predictions', True):
                self._save_evaluation_results()
            
            # 绘制评估结果
            if self.config.get('save_figures', True):
                self.plot_evaluation_results(y, self.predict(X), task_type)
            
            return self.evaluation_results
        except Exception as e:
            self.logger.error(f"评估模型时发生异常: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          task_type: str = 'regression') -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            task_type: 任务类型，'regression'或'classification'
        
        Returns:
            评估指标字典
        """
        metrics_dict = {}
        
        if task_type == 'regression':
            # 回归指标
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            metrics_dict['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics_dict['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics_dict['rmse'] = float(np.sqrt(metrics_dict['mse']))
            metrics_dict['r2'] = float(r2_score(y_true, y_pred))
            
            # 平均绝对百分比误差 (MAPE)
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = 0
            metrics_dict['mape'] = float(mape)
        
        else:
            # 分类指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # 确保预测值是一维的
            if len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)
            
            metrics_dict['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics_dict['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics_dict['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics_dict['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        return metrics_dict
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """保存模型
        
        Args:
            model_path: 模型保存路径，如果为None则使用默认路径
        
        Returns:
            实际的模型保存路径
        """
        try:
            self.logger.info("保存模型")
            
            # 检查模型是否存在
            if self.model is None:
                raise ValueError("未提供模型实例，无法保存模型")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if model_path is None:
                model_dir = self.config.get('models_dir', './models')
                model_path = os.path.join(model_dir, f'model_{timestamp}')
            
            # 创建保存目录
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            
            # 尝试使用模型提供的save方法
            if hasattr(self.model, 'save'):
                try:
                    self.model.save(model_path)
                    self.logger.info(f"模型已保存到: {model_path}")
                    return model_path
                except Exception as e:
                    self.logger.warning(f"使用模型自带的save方法保存失败: {str(e)}，尝试使用joblib保存")
            
            # 使用joblib保存
            joblib_path = f"{model_path}.joblib"
            joblib.dump(self.model, joblib_path)
            self.logger.info(f"模型已保存到: {joblib_path}")
            
            # 保存流水线配置
            config_path = f"{model_path}_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.logger.info(f"流水线配置已保存到: {config_path}")
            
            # 如果有数据处理器、特征选择器等，也保存它们
            if self.data_processor is not None:
                try:
                    dp_path = f"{model_path}_data_processor.joblib"
                    joblib.dump(self.data_processor, dp_path)
                    self.logger.info(f"数据处理器已保存到: {dp_path}")
                except Exception as e:
                    self.logger.warning(f"保存数据处理器失败: {str(e)}")
            
            if self.feature_selector is not None:
                try:
                    fs_path = f"{model_path}_feature_selector.joblib"
                    joblib.dump(self.feature_selector, fs_path)
                    self.logger.info(f"特征选择器已保存到: {fs_path}")
                except Exception as e:
                    self.logger.warning(f"保存特征选择器失败: {str(e)}")
            
            return joblib_path
        except Exception as e:
            self.logger.error(f"保存模型时发生异常: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载模型: {model_path}")
            
            # 尝试加载流水线配置
            config_path = model_path.replace('.joblib', '_config.json').replace('.pkl', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"流水线配置已从{config_path}加载")
            
            # 尝试直接加载模型（适用于Keras模型）
            try:
                if os.path.isdir(model_path):
                    # 尝试作为目录加载（Keras模型）
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                    self.logger.info("成功加载Keras模型")
                    self.is_trained = True
                    return True
            except Exception as e:
                self.logger.warning(f"尝试作为Keras模型加载失败: {str(e)}")
            
            # 尝试使用joblib加载
            if os.path.exists(model_path) or os.path.exists(f"{model_path}.joblib"):
                if not os.path.exists(model_path):
                    model_path = f"{model_path}.joblib"
                
                self.model = joblib.load(model_path)
                self.logger.info(f"模型已从{model_path}加载")
                self.is_trained = True
            else:
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 尝试加载数据处理器
            dp_path = model_path.replace('.joblib', '_data_processor.joblib')
            if os.path.exists(dp_path):
                try:
                    self.data_processor = joblib.load(dp_path)
                    self.logger.info(f"数据处理器已从{dp_path}加载")
                except Exception as e:
                    self.logger.warning(f"加载数据处理器失败: {str(e)}")
            
            # 尝试加载特征选择器
            fs_path = model_path.replace('.joblib', '_feature_selector.joblib')
            if os.path.exists(fs_path):
                try:
                    self.feature_selector = joblib.load(fs_path)
                    self.logger.info(f"特征选择器已从{fs_path}加载")
                except Exception as e:
                    self.logger.warning(f"加载特征选择器失败: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"加载模型时发生异常: {str(e)}")
            return False
    
    def _save_training_history(self) -> None:
        """保存训练历史"""
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            history_dir = os.path.join(self.config.get('results_dir', './results'), 'training_history')
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            
            history_path = os.path.join(history_dir, f'training_history_{timestamp}.json')
            
            # 转换训练历史为可序列化的格式
            history_dict = {}
            
            if hasattr(self.training_history, 'history'):
                # Keras模型的训练历史
                for key, values in self.training_history.history.items():
                    history_dict[key] = [float(v) for v in values]
            elif isinstance(self.training_history, dict):
                # 字典格式的训练历史
                for key, values in self.training_history.items():
                    if isinstance(values, (list, np.ndarray)):
                        history_dict[key] = [float(v) for v in values]
                    else:
                        history_dict[key] = float(values)
            else:
                self.logger.warning("无法识别训练历史格式，跳过保存")
                return
            
            # 保存到文件
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"训练历史已保存到: {history_path}")
        except Exception as e:
            self.logger.error(f"保存训练历史时发生异常: {str(e)}")
    
    def _save_predictions(self, y_pred: np.ndarray) -> None:
        """保存预测结果"""
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            pred_dir = os.path.join(self.config.get('results_dir', './results'), 'predictions')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            
            pred_path = os.path.join(pred_dir, f'predictions_{timestamp}.npy')
            
            # 保存预测结果
            np.save(pred_path, y_pred)
            
            self.logger.info(f"预测结果已保存到: {pred_path}")
        except Exception as e:
            self.logger.error(f"保存预测结果时发生异常: {str(e)}")
    
    def _save_evaluation_results(self) -> None:
        """保存评估结果"""
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            eval_dir = os.path.join(self.config.get('results_dir', './results'), 'evaluation')
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            
            eval_path = os.path.join(eval_dir, f'evaluation_{timestamp}.json')
            
            # 转换评估结果为可序列化的格式
            serializable_dict = {}
            for key, value in self.evaluation_results.items():
                if isinstance(value, np.ndarray):
                    serializable_dict[key] = value.tolist()
                elif isinstance(value, np.generic):
                    serializable_dict[key] = np.asscalar(value)
                else:
                    serializable_dict[key] = value
            
            # 保存到文件
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"评估结果已保存到: {eval_path}")
        except Exception as e:
            self.logger.error(f"保存评估结果时发生异常: {str(e)}")
    
    def plot_training_history(self) -> None:
        """绘制训练历史"""
        try:
            self.logger.info("绘制训练历史")
            
            # 检查是否有训练历史
            if self.training_history is None:
                self.logger.warning("没有训练历史可绘制")
                return
            
            # 获取训练历史数据
            if hasattr(self.training_history, 'history'):
                history = self.training_history.history
            elif isinstance(self.training_history, dict):
                history = self.training_history
            else:
                self.logger.warning("无法识别训练历史格式，跳过绘图")
                return
            
            # 找出所有的指标
            metrics = set()
            for key in history.keys():
                # 去掉可能的前缀（如'val_'）
                metric = key.replace('val_', '')
                metrics.add(metric)
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            fig_dir = os.path.join(self.config.get('figures_dir', './figures'), 'training')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            
            # 为每个指标绘制图形
            for metric in metrics:
                # 检查是否有训练和验证的指标
                if metric in history:
                    plt.figure(figsize=(10, 6))
                    
                    # 绘制训练指标
                    plt.plot(history[metric], label='训练')
                    
                    # 绘制验证指标（如果存在）
                    val_key = f'val_{metric}'
                    if val_key in history:
                        plt.plot(history[val_key], label='验证')
                    
                    # 设置图表属性
                    plt.title(f'训练过程中的{metric}变化')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # 保存图像
                    fig_path = os.path.join(fig_dir, f'training_{metric}_{timestamp}.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"训练历史图像已保存到: {fig_path}")
                    
                    # 关闭图形，释放内存
                    plt.close()
        except Exception as e:
            self.logger.error(f"绘制训练历史时发生异常: {str(e)}")
    
    def plot_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               task_type: str = 'regression') -> None:
        """绘制评估结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            task_type: 任务类型，'regression'或'classification'
        """
        try:
            self.logger.info(f"绘制评估结果，任务类型: {task_type}")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            fig_dir = os.path.join(self.config.get('figures_dir', './figures'), 'evaluation')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            
            if task_type == 'regression':
                # 回归任务的评估可视化
                
                # 1. 真实值 vs 预测值散点图
                plt.figure(figsize=(10, 8))
                plt.scatter(y_true, y_pred, alpha=0.5)
                
                # 添加理想线
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                plt.title('真实值 vs 预测值')
                plt.xlabel('真实值')
                plt.ylabel('预测值')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                fig_path = os.path.join(fig_dir, f'true_vs_pred_{timestamp}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"真实值vs预测值图像已保存到: {fig_path}")
                plt.close()
                
                # 2. 残差图
                plt.figure(figsize=(10, 6))
                residuals = y_true - y_pred
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--', lw=2)
                
                plt.title('残差图')
                plt.xlabel('预测值')
                plt.ylabel('残差 (真实值 - 预测值)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                fig_path = os.path.join(fig_dir, f'residuals_{timestamp}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"残差图已保存到: {fig_path}")
                plt.close()
                
                # 3. 残差分布直方图
                plt.figure(figsize=(10, 6))
                plt.hist(residuals, bins=30, alpha=0.7)
                
                plt.title('残差分布')
                plt.xlabel('残差')
                plt.ylabel('频率')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                fig_path = os.path.join(fig_dir, f'residuals_distribution_{timestamp}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"残差分布图已保存到: {fig_path}")
                plt.close()
                
                # 4. 预测值和真实值的时间序列对比图（如果数据是时间序列）
                if len(y_true) > 100:  # 仅当数据量足够大时绘制
                    plt.figure(figsize=(15, 8))
                    plt.plot(range(len(y_true)), y_true, label='真实值', linewidth=2)
                    plt.plot(range(len(y_pred)), y_pred, label='预测值', linewidth=2, alpha=0.7)
                    
                    plt.title('真实值和预测值的时间序列对比')
                    plt.xlabel('样本索引')
                    plt.ylabel('值')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    fig_path = os.path.join(fig_dir, f'timeseries_comparison_{timestamp}.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"时间序列对比图已保存到: {fig_path}")
                    plt.close()
                
            else:
                # 分类任务的评估可视化
                
                # 确保预测值是一维的
                if len(y_pred.shape) > 1:
                    y_pred_labels = np.argmax(y_pred, axis=1)
                else:
                    y_pred_labels = y_pred
                
                # 1. 混淆矩阵
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                
                cm = confusion_matrix(y_true, y_pred_labels)
                plt.figure(figsize=(10, 8))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                
                plt.title('混淆矩阵')
                plt.grid(False)
                
                fig_path = os.path.join(fig_dir, f'confusion_matrix_{timestamp}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"混淆矩阵图像已保存到: {fig_path}")
                plt.close()
                
                # 2. ROC曲线（如果支持概率预测）
                try:
                    from sklearn.metrics import roc_curve, auc
                    
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        # 多类别分类
                        # 计算每个类别的ROC曲线
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        
                        # 获取唯一类别
                        classes = np.unique(y_true)
                        
                        plt.figure(figsize=(10, 8))
                        
                        for i, cls in enumerate(classes):
                            # 将多类别问题转换为二分类问题
                            y_true_binary = (y_true == cls).astype(int)
                            y_pred_prob = y_pred[:, i]
                            
                            # 计算ROC曲线
                            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_prob)
                            roc_auc[i] = auc(fpr[i], tpr[i])
                            
                            # 绘制ROC曲线
                            plt.plot(fpr[i], tpr[i], lw=2, 
                                    label=f'类别 {cls} (AUC = {roc_auc[i]:.2f})')
                        
                        # 添加随机猜测线
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('假阳性率')
                        plt.ylabel('真阳性率')
                        plt.title('多类别ROC曲线')
                        plt.legend(loc="lower right")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        fig_path = os.path.join(fig_dir, f'roc_curve_{timestamp}.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                        self.logger.info(f"ROC曲线图像已保存到: {fig_path}")
                        plt.close()
                    elif len(np.unique(y_true)) == 2:
                        # 二分类问题
                        if len(y_pred.shape) > 1:
                            y_pred_prob = y_pred[:, 1]
                        else:
                            # 如果没有概率预测，使用预测标签作为概率（仅作参考）
                            y_pred_prob = y_pred_labels
                        
                        # 计算ROC曲线
                        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.figure(figsize=(10, 8))
                        plt.plot(fpr, tpr, lw=2, 
                                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('假阳性率')
                        plt.ylabel('真阳性率')
                        plt.title('ROC曲线')
                        plt.legend(loc="lower right")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        fig_path = os.path.join(fig_dir, f'roc_curve_{timestamp}.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                        self.logger.info(f"ROC曲线图像已保存到: {fig_path}")
                        plt.close()
                except Exception as e:
                    self.logger.warning(f"绘制ROC曲线时发生异常: {str(e)}")
                
                # 3. 精确率-召回率曲线
                try:
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    
                    if len(y_pred.shape) > 1:
                        if y_pred.shape[1] > 1:
                            # 多类别分类
                            precision = dict()
                            recall = dict()
                            average_precision = dict()
                            
                            # 获取唯一类别
                            classes = np.unique(y_true)
                            
                            plt.figure(figsize=(10, 8))
                            
                            for i, cls in enumerate(classes):
                                # 将多类别问题转换为二分类问题
                                y_true_binary = (y_true == cls).astype(int)
                                y_pred_prob = y_pred[:, i]
                                
                                # 计算精确率-召回率曲线
                                precision[i], recall[i], _ = precision_recall_curve(
                                    y_true_binary, y_pred_prob
                                )
                                average_precision[i] = average_precision_score(
                                    y_true_binary, y_pred_prob
                                )
                                
                                # 绘制精确率-召回率曲线
                                plt.plot(recall[i], precision[i], lw=2, 
                                        label=f'类别 {cls} (AP = {average_precision[i]:.2f})')
                        else:
                            # 二分类问题
                            if len(np.unique(y_true)) == 2:
                                y_pred_prob = y_pred[:, 0]
                            else:
                                y_pred_prob = y_pred_labels
                            
                            # 计算精确率-召回率曲线
                            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
                            average_precision = average_precision_score(y_true, y_pred_prob)
                            
                            plt.figure(figsize=(10, 8))
                            plt.plot(recall, precision, lw=2, 
                                    label=f'精确率-召回率曲线 (AP = {average_precision:.2f})')
                            
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('召回率')
                        plt.ylabel('精确率')
                        plt.title('精确率-召回率曲线')
                        plt.legend(loc="lower left")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        fig_path = os.path.join(fig_dir, f'precision_recall_curve_{timestamp}.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                        self.logger.info(f"精确率-召回率曲线图像已保存到: {fig_path}")
                        plt.close()
                except Exception as e:
                    self.logger.warning(f"绘制精确率-召回率曲线时发生异常: {str(e)}")
        except Exception as e:
            self.logger.error(f"绘制评估结果时发生异常: {str(e)}")
    
    def full_pipeline(self, data_path: str, 
                     feature_columns: Optional[List[str]] = None, 
                     target_column: str = 'close', 
                     date_column: str = 'date', 
                     train_ratio: float = 0.7, 
                     valid_ratio: float = 0.15, 
                     task_type: str = 'regression', 
                     **kwargs) -> Dict[str, object]:
        """运行完整的预测流水线
        
        Args:
            data_path: 数据文件路径
            feature_columns: 特征列名列表，如果为None则使用所有非目标列
            target_column: 目标列名
            date_column: 日期列名
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            task_type: 任务类型，'regression'或'classification'
            **kwargs: 传递给各阶段的额外参数
        
        Returns:
            包含流水线结果的字典
        """
        try:
            self.logger.info("开始运行完整的预测流水线")
            
            # 记录开始时间
            start_time = time.time()
            
            # 加载数据
            df, X, y = self.load_data(
                data_path, feature_columns, target_column, date_column
            )
            
            # 划分训练集、验证集和测试集
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data(
                X, y, train_ratio, valid_ratio
            )
            
            # 预处理训练数据
            X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train, is_train=True)
            
            # 预处理验证数据
            if X_valid is not None and y_valid is not None:
                X_valid_processed, y_valid_processed = self.preprocess_data(X_valid, y_valid, is_train=False)
            else:
                X_valid_processed, y_valid_processed = None, None
            
            # 预处理测试数据
            X_test_processed, y_test_processed = self.preprocess_data(X_test, y_test, is_train=False)
            
            # 特征选择
            X_train_selected = self.select_features(X_train_processed, y_train_processed, is_train=True)
            
            if X_valid_processed is not None:
                X_valid_selected = self.select_features(X_valid_processed, y_valid_processed, is_train=False)
            else:
                X_valid_selected = None
            
            X_test_selected = self.select_features(X_test_processed, y_test_processed, is_train=False)
            
            # 训练模型
            self.train(
                X_train_selected, y_train_processed, 
                X_valid_selected, y_valid_processed,
                **kwargs
            )
            
            # 评估模型
            evaluation_results = self.evaluate(
                X_test_selected, y_test_processed, 
                task_type=task_type, 
                **kwargs
            )
            
            # 进行预测
            y_pred = self.predict(X_test_selected)
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(f"完整预测流水线运行完成，总耗时: {total_time:.2f}秒")
            
            # 返回结果
            results = {
                'model': self.model,
                'evaluation_results': evaluation_results,
                'predictions': y_pred,
                'y_test': y_test_processed,
                'X_test': X_test_selected,
                'training_history': self.training_history,
                'total_time': total_time
            }
            
            return results
        except Exception as e:
            self.logger.error(f"运行完整预测流水线时发生异常: {str(e)}")
            raise
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray, 
                         retrain_ratio: float = 0.1, 
                         **update_params) -> object:
        """增量更新模型
        
        Args:
            X_new: 新的特征数据
            y_new: 新的目标数据
            retrain_ratio: 用于重新训练的新数据比例
            **update_params: 传递给训练方法的额外参数
        
        Returns:
            更新后的模型
        """
        try:
            self.logger.info(f"增量更新模型，重新训练比例: {retrain_ratio}")
            
            # 检查模型是否已训练
            if not self.is_trained:
                self.logger.warning("模型尚未训练，将使用新数据进行首次训练")
                return self.train(X_new, y_new, **update_params)
            
            # 预处理新数据
            X_new_processed, y_new_processed = self.preprocess_data(X_new, y_new, is_train=False)
            
            # 特征选择
            X_new_selected = self.select_features(X_new_processed, y_new_processed, is_train=False)
            
            # 确定用于重新训练的数据量
            n_samples = len(X_new_selected)
            n_retrain = max(1, int(n_samples * retrain_ratio))
            
            # 选择最新的n_retrain个样本用于重新训练
            X_retrain = X_new_selected[-n_retrain:]
            y_retrain = y_new_processed[-n_retrain:]
            
            # 检查模型是否支持部分拟合
            if hasattr(self.model, 'partial_fit'):
                # 使用partial_fit进行增量更新
                try:
                    self.logger.info("使用partial_fit方法进行增量更新")
                    self.model.partial_fit(X_retrain, y_retrain, **update_params)
                except Exception as e:
                    self.logger.warning(f"使用partial_fit方法失败: {str(e)}，尝试使用fit方法")
                    # 回退到使用fit方法
                    self.model.fit(X_retrain, y_retrain, **update_params)
            else:
                # 直接使用fit方法
                self.logger.info("模型不支持partial_fit，使用fit方法进行重新训练")
                self.model.fit(X_retrain, y_retrain, **update_params)
            
            self.logger.info("模型增量更新完成")
            
            return self.model
        except Exception as e:
            self.logger.error(f"增量更新模型时发生异常: {str(e)}")
            raise
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, 
                      task_type: str = 'regression', 
                      **kwargs) -> Dict[str, float]:
        """交叉验证
        
        Args:
            X: 特征数据
            y: 目标数据
            cv: 交叉验证的折数
            task_type: 任务类型，'regression'或'classification'
            **kwargs: 传递给训练和评估方法的额外参数
        
        Returns:
            交叉验证的评估指标结果
        """
        try:
            self.logger.info(f"进行交叉验证，折数: {cv}")
            
            from sklearn.model_selection import KFold
            
            # 创建K折交叉验证分割器
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.config.get('random_state', 42))
            
            # 存储每折的评估结果
            fold_results = []
            
            # 进行交叉验证
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                self.logger.info(f"开始交叉验证第{fold+1}/{cv}折")
                
                # 分割数据
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 预处理数据
                X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train, is_train=True)
                X_val_processed, y_val_processed = self.preprocess_data(X_val, y_val, is_train=False)
                
                # 特征选择
                X_train_selected = self.select_features(X_train_processed, y_train_processed, is_train=True)
                X_val_selected = self.select_features(X_val_processed, y_val_processed, is_train=False)
                
                # 需要重新初始化模型以避免前面折的影响
                if hasattr(self.model, 'reset'):
                    self.model.reset()
                else:
                    # 如果模型没有reset方法，尝试重新创建模型
                    import copy
                    original_model = copy.deepcopy(self.model)
                    
                # 训练模型
                self.train(X_train_selected, y_train_processed, **kwargs)
                
                # 评估模型
                fold_result = self.evaluate(X_val_selected, y_val_processed, task_type=task_type, **kwargs)
                fold_results.append(fold_result)
                
                self.logger.info(f"第{fold+1}折评估结果: {fold_result}")
                
                # 恢复原始模型
                if not hasattr(self.model, 'reset'):
                    self.model = original_model
            
            # 计算平均评估指标
            avg_results = {}
            if fold_results:
                # 获取所有指标名称
                metric_names = fold_results[0].keys()
                
                # 计算每个指标的平均值
                for metric in metric_names:
                    # 检查指标是否是数值类型
                    if all(isinstance(fold_result.get(metric, 0), (int, float, np.number)) for fold_result in fold_results):
                        avg_results[f'avg_{metric}'] = float(np.mean([fold_result.get(metric, 0) for fold_result in fold_results]))
                        avg_results[f'std_{metric}'] = float(np.std([fold_result.get(metric, 0) for fold_result in fold_results]))
            
            self.logger.info(f"交叉验证完成，平均评估结果: {avg_results}")
            
            # 保存交叉验证结果
            if self.config.get('save_predictions', True):
                self._save_cross_validation_results(fold_results, avg_results)
            
            return avg_results
        except Exception as e:
            self.logger.error(f"进行交叉验证时发生异常: {str(e)}")
            raise
    
    def _save_cross_validation_results(self, fold_results: List[Dict], avg_results: Dict) -> None:
        """保存交叉验证结果
        
        Args:
            fold_results: 每折的评估结果列表
            avg_results: 平均评估结果字典
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            cv_dir = os.path.join(self.config.get('results_dir', './results'), 'cross_validation')
            if not os.path.exists(cv_dir):
                os.makedirs(cv_dir)
            
            cv_path = os.path.join(cv_dir, f'cross_validation_{timestamp}.json')
            
            # 构建保存的数据结构
            cv_data = {
                'fold_results': fold_results,
                'average_results': avg_results,
                'config': self.config,
                'timestamp': timestamp
            }
            
            # 转换为可序列化的格式
            serializable_data = {
                'fold_results': [],
                'average_results': {},
                'config': cv_data['config'],
                'timestamp': cv_data['timestamp']
            }
            
            # 处理每折的结果
            for fold_result in cv_data['fold_results']:
                serializable_fold = {}
                for key, value in fold_result.items():
                    if isinstance(value, np.ndarray):
                        serializable_fold[key] = value.tolist()
                    elif isinstance(value, np.generic):
                        serializable_fold[key] = np.asscalar(value)
                    else:
                        serializable_fold[key] = value
                serializable_data['fold_results'].append(serializable_fold)
            
            # 处理平均结果
            for key, value in cv_data['average_results'].items():
                if isinstance(value, np.ndarray):
                    serializable_data['average_results'][key] = value.tolist()
                elif isinstance(value, np.generic):
                    serializable_data['average_results'][key] = np.asscalar(value)
                else:
                    serializable_data['average_results'][key] = value
            
            # 保存到文件
            with open(cv_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"交叉验证结果已保存到: {cv_path}")
        except Exception as e:
            self.logger.error(f"保存交叉验证结果时发生异常: {str(e)}")
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            param_grid: Dict, 
                            cv: int = 5, 
                            scoring: str = 'neg_mean_squared_error', 
                            n_jobs: int = -1, 
                            **kwargs) -> Dict:
        """超参数调优
        
        Args:
            X: 特征数据
            y: 目标数据
            param_grid: 超参数网格
            cv: 交叉验证的折数
            scoring: 评估指标
            n_jobs: 并行计算的任务数
            **kwargs: 传递给GridSearchCV的额外参数
        
        Returns:
            最佳超参数配置
        """
        try:
            self.logger.info("进行超参数调优")
            
            from sklearn.model_selection import GridSearchCV
            
            # 预处理数据
            X_processed, y_processed = self.preprocess_data(X, y, is_train=True)
            
            # 特征选择
            X_selected = self.select_features(X_processed, y_processed, is_train=True)
            
            # 创建GridSearchCV对象
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=self.config.get('verbose', 1),
                **kwargs
            )
            
            # 执行超参数搜索
            grid_search.fit(X_selected, y_processed)
            
            # 获取最佳参数和最佳得分
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            self.logger.info(f"超参数调优完成，最佳参数: {best_params}，最佳得分: {best_score}")
            
            # 更新模型为最佳模型
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # 保存超参数调优结果
            if self.config.get('save_predictions', True):
                self._save_hyperparameter_results(grid_search)
            
            return best_params
        except Exception as e:
            self.logger.error(f"进行超参数调优时发生异常: {str(e)}")
            raise
    
    def _save_hyperparameter_results(self, grid_search: object) -> None:
        """保存超参数调优结果
        
        Args:
            grid_search: GridSearchCV对象
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            hp_dir = os.path.join(self.config.get('results_dir', './results'), 'hyperparameters')
            if not os.path.exists(hp_dir):
                os.makedirs(hp_dir)
            
            hp_path = os.path.join(hp_dir, f'hyperparameter_tuning_{timestamp}.json')
            
            # 构建保存的数据结构
            hp_data = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': {},
                'param_grid': grid_search.param_grid,
                'config': self.config,
                'timestamp': timestamp
            }
            
            # 处理交叉验证结果
            if hasattr(grid_search, 'cv_results_'):
                for key, value in grid_search.cv_results_.items():
                    if isinstance(value, np.ndarray):
                        hp_data['cv_results'][key] = value.tolist()
                    else:
                        hp_data['cv_results'][key] = value
            
            # 保存到文件
            with open(hp_path, 'w', encoding='utf-8') as f:
                json.dump(hp_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"超参数调优结果已保存到: {hp_path}")
        except Exception as e:
            self.logger.error(f"保存超参数调优结果时发生异常: {str(e)}")
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None, 
                             top_n: int = 10) -> Dict[str, float]:
        """获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            top_n: 显示前N个重要的特征
        
        Returns:
            特征重要性字典
        """
        try:
            self.logger.info(f"获取特征重要性，显示前{top_n}个特征")
            
            # 检查模型是否已训练
            if not self.is_trained:
                self.logger.warning("模型尚未训练，无法获取特征重要性")
                return {}
            
            # 检查模型是否支持特征重要性
            if hasattr(self.model, 'feature_importances_'):
                # 树模型的特征重要性
                importances = self.model.feature_importances_
                
            elif hasattr(self.model, 'coef_'):
                # 线性模型的系数（绝对值作为重要性）
                coef = self.model.coef_
                if len(coef.shape) > 1:
                    # 多输出模型，取绝对值的平均值
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
            
            else:
                self.logger.warning("模型不支持获取特征重要性")
                return {}
            
            # 创建特征重要性字典
            feature_importance = {}
            
            if feature_names is not None and len(feature_names) == len(importances):
                # 使用提供的特征名称
                for i, name in enumerate(feature_names):
                    feature_importance[name] = float(importances[i])
            else:
                # 使用索引作为特征名称
                for i in range(len(importances)):
                    feature_importance[f'feature_{i}'] = float(importances[i])
            
            # 按重要性排序
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 取前N个重要的特征
            top_importance = dict(sorted_importance[:top_n])
            
            self.logger.info(f"特征重要性（前{top_n}个）: {top_importance}")
            
            # 绘制特征重要性图
            if self.config.get('save_figures', True):
                self.plot_feature_importance(top_importance)
            
            return top_importance
        except Exception as e:
            self.logger.error(f"获取特征重要性时发生异常: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性字典
        """
        try:
            self.logger.info("绘制特征重要性图")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            fig_dir = os.path.join(self.config.get('figures_dir', './figures'), 'features')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            
            # 排序特征重要性
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_importance)
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 绘制条形图
            bars = plt.barh(range(len(features)), importances, align='center')
            
            # 设置坐标轴标签
            plt.yticks(range(len(features)), features)
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.title('特征重要性排名')
            
            # 在条形图上显示数值
            for bar, importance in zip(bars, importances):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2., 
                        f'{importance:.4f}', ha='left', va='center')
            
            plt.grid(True, linestyle='--', alpha=0.7, axis='x')
            plt.tight_layout()
            
            # 保存图像
            fig_path = os.path.join(fig_dir, f'feature_importance_{timestamp}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"特征重要性图像已保存到: {fig_path}")
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制特征重要性图时发生异常: {str(e)}")
    
    def generate_predict_report(self, df: pd.DataFrame, y_pred: np.ndarray, 
                              y_true: Optional[np.ndarray] = None, 
                              date_column: str = 'date', 
                              target_column: str = 'close') -> str:
        """生成预测报告
        
        Args:
            df: 包含原始数据的DataFrame
            y_pred: 预测结果
            y_true: 真实值（可选）
            date_column: 日期列名
            target_column: 目标列名
        
        Returns:
            报告文件路径
        """
        try:
            self.logger.info("生成预测报告")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            report_dir = os.path.join(self.config.get('results_dir', './results'), 'reports')
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            report_path = os.path.join(report_dir, f'prediction_report_{timestamp}.html')
            
            # 创建报告内容
            report_content = []
            
            # 添加报告头部
            report_content.append('<!DOCTYPE html>')
            report_content.append('<html>')
            report_content.append('<head>')
            report_content.append('<meta charset="UTF-8">')
            report_content.append('<title>预测报告</title>')
            report_content.append('<style>')
            report_content.append('body { font-family: Arial, sans-serif; margin: 20px; }')
            report_content.append('h1, h2 { color: #333; }')
            report_content.append('.container { max-width: 1200px; margin: 0 auto; }')
            report_content.append('.metrics { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }')
            report_content.append('.metric-card { background: #f5f5f5; padding: 20px; border-radius: 5px; flex: 1; min-width: 200px; }')
            report_content.append('.metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }')
            report_content.append('.metric-name { color: #7f8c8d; }')
            report_content.append('table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
            report_content.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
            report_content.append('th { background-color: #f2f2f2; }')
            report_content.append('tr:nth-child(even) { background-color: #f9f9f9; }')
            report_content.append('img { max-width: 100%; height: auto; margin: 20px 0; }')
            report_content.append('</style>')
            report_content.append('</head>')
            report_content.append('<body>')
            report_content.append('<div class="container">')
            report_content.append(f'<h1>预测报告 - {timestamp}</h1>')
            
            # 添加基本信息
            report_content.append('<h2>基本信息</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>项目</th><th>值</th></tr>')
            report_content.append(f'<tr><td>模型类型</td><td>{type(self.model).__name__}</td></tr>')
            report_content.append(f'<tr><td>预测样本数</td><td>{len(y_pred)}</td></tr>')
            report_content.append(f'<tr><td>生成时间</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>')
            report_content.append('</table>')
            
            # 添加评估指标
            if y_true is not None:
                report_content.append('<h2>评估指标</h2>')
                metrics = self._calculate_metrics(y_true, y_pred)
                
                report_content.append('<div class="metrics">')
                for metric_name, metric_value in metrics.items():
                    report_content.append('<div class="metric-card">')
                    report_content.append(f'<div class="metric-name">{metric_name.upper()}</div>')
                    report_content.append(f'<div class="metric-value">{metric_value:.4f}</div>')
                    report_content.append('</div>')
                report_content.append('</div>')
            
            # 添加预测结果示例
            report_content.append('<h2>预测结果示例（前10条）</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>索引</th>')
            
            if date_column in df.columns:
                report_content.append(f'<th>{date_column}</th>')
            
            if y_true is not None and target_column in df.columns:
                report_content.append(f'<th>{target_column}（真实值）</th>')
            
            report_content.append('<th>预测值</th>')
            
            if y_true is not None:
                report_content.append('<th>误差</th>')
                report_content.append('<th>相对误差(%)</th>')
            
            report_content.append('</tr>')
            
            # 添加前10条预测结果
            sample_size = min(10, len(y_pred))
            for i in range(sample_size):
                report_content.append('<tr>')
                report_content.append(f'<td>{i}</td>')
                
                if date_column in df.columns:
                    date_value = df.iloc[i][date_column] if i < len(df) else 'N/A'
                    report_content.append(f'<td>{date_value}</td>')
                
                true_value = None
                if y_true is not None and target_column in df.columns and i < len(df):
                    true_value = df.iloc[i][target_column]
                    report_content.append(f'<td>{true_value:.4f}</td>')
                elif y_true is not None and i < len(y_true):
                    true_value = y_true[i]
                    report_content.append(f'<td>{true_value:.4f}</td>')
                
                pred_value = y_pred[i]
                if isinstance(pred_value, np.ndarray) and len(pred_value) == 1:
                    pred_value = pred_value[0]
                report_content.append(f'<td>{float(pred_value):.4f}</td>')
                
                if y_true is not None and true_value is not None:
                    error = float(pred_value) - float(true_value)
                    report_content.append(f'<td>{error:.4f}</td>')
                    
                    if float(true_value) != 0:
                        rel_error = (error / float(true_value)) * 100
                    else:
                        rel_error = 0
                    report_content.append(f'<td>{rel_error:.2f}</td>')
                
                report_content.append('</tr>')
            
            report_content.append('</table>')
            
            # 添加图表
            report_content.append('<h2>可视化结果</h2>')
            
            # 查找已生成的图表
            figures_dir = self.config.get('figures_dir', './figures')
            recent_figures = []
            
            # 查找最近的评估图表
            eval_fig_dir = os.path.join(figures_dir, 'evaluation')
            if os.path.exists(eval_fig_dir):
                files = sorted(os.listdir(eval_fig_dir), key=lambda x: os.path.getmtime(os.path.join(eval_fig_dir, x)), reverse=True)
                for file in files[:3]:  # 取最近3个图表
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        recent_figures.append(os.path.join(eval_fig_dir, file))
            
            # 查找最近的特征重要性图表
            feat_fig_dir = os.path.join(figures_dir, 'features')
            if os.path.exists(feat_fig_dir):
                files = sorted(os.listdir(feat_fig_dir), key=lambda x: os.path.getmtime(os.path.join(feat_fig_dir, x)), reverse=True)
                for file in files[:1]:  # 取最近1个图表
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        recent_figures.append(os.path.join(feat_fig_dir, file))
            
            # 添加图表到报告
            for fig_path in recent_figures:
                # 使用相对路径
                rel_path = os.path.relpath(fig_path, report_dir)
                report_content.append(f'<img src="{rel_path}" alt="{os.path.basename(fig_path)}">')
            
            # 添加配置信息
            report_content.append('<h2>配置信息</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>配置项</th><th>值</th></tr>')
            for key, value in self.config.items():
                report_content.append(f'<tr><td>{key}</td><td>{value}</td></tr>')
            report_content.append('</table>')
            
            # 添加报告尾部
            report_content.append('</div>')
            report_content.append('</body>')
            report_content.append('</html>')
            
            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            self.logger.info(f"预测报告已生成: {report_path}")
            
            return report_path
        except Exception as e:
            self.logger.error(f"生成预测报告时发生异常: {str(e)}")
            raise
    
    def save_pipeline(self, pipeline_path: Optional[str] = None) -> str:
        """保存整个预测流水线
        
        Args:
            pipeline_path: 流水线保存路径，如果为None则使用默认路径
        
        Returns:
            实际的流水线保存路径
        """
        try:
            self.logger.info("保存整个预测流水线")
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if pipeline_path is None:
                pipeline_dir = self.config.get('models_dir', './models')
                pipeline_path = os.path.join(pipeline_dir, f'pipeline_{timestamp}')
            
            # 创建保存目录
            if not os.path.exists(os.path.dirname(pipeline_path)):
                os.makedirs(os.path.dirname(pipeline_path))
            
            # 保存流水线组件
            pipeline_data = {
                'data_processor': self.data_processor,
                'feature_selector': self.feature_selector,
                'model': self.model,
                'model_trainer': self.model_trainer,
                'model_evaluator': self.model_evaluator,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'evaluation_results': self.evaluation_results,
                'timestamp': timestamp
            }
            
            # 使用joblib保存
            joblib_path = f"{pipeline_path}.joblib"
            joblib.dump(pipeline_data, joblib_path)
            
            self.logger.info(f"预测流水线已保存到: {joblib_path}")
            
            return joblib_path
        except Exception as e:
            self.logger.error(f"保存预测流水线时发生异常: {str(e)}")
            raise
    
    def load_pipeline(self, pipeline_path: str) -> bool:
        """加载预测流水线
        
        Args:
            pipeline_path: 流水线文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"加载预测流水线: {pipeline_path}")
            
            # 检查文件是否存在
            if not os.path.exists(pipeline_path):
                # 尝试添加.joblib后缀
                if not pipeline_path.endswith('.joblib'):
                    pipeline_path = f"{pipeline_path}.joblib"
                    if not os.path.exists(pipeline_path):
                        self.logger.error(f"流水线文件不存在: {pipeline_path}")
                        return False
            
            # 加载流水线数据
            pipeline_data = joblib.load(pipeline_path)
            
            # 恢复流水线组件
            if 'data_processor' in pipeline_data:
                self.data_processor = pipeline_data['data_processor']
            
            if 'feature_selector' in pipeline_data:
                self.feature_selector = pipeline_data['feature_selector']
            
            if 'model' in pipeline_data:
                self.model = pipeline_data['model']
            
            if 'model_trainer' in pipeline_data:
                self.model_trainer = pipeline_data['model_trainer']
            
            if 'model_evaluator' in pipeline_data:
                self.model_evaluator = pipeline_data['model_evaluator']
            
            if 'config' in pipeline_data:
                self.config = pipeline_data['config']
                # 更新日志目录
                self.logger = self._init_logger()
            
            if 'is_trained' in pipeline_data:
                self.is_trained = pipeline_data['is_trained']
            
            if 'training_history' in pipeline_data:
                self.training_history = pipeline_data['training_history']
            
            if 'evaluation_results' in pipeline_data:
                self.evaluation_results = pipeline_data['evaluation_results']
            
            self.logger.info("预测流水线加载完成")
            
            # 创建必要的目录
            self._create_directories()
            
            return True
        except Exception as e:
            self.logger.error(f"加载预测流水线时发生异常: {str(e)}")
            return False

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['PredictPipeline']