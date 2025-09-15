"""深度学习模型训练器模块，实现模型训练、验证和早停等功能"""

import tensorflow as tf
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ModelTrainer:
    """深度学习模型训练器类，提供模型训练、验证和早停等功能"""
    
    def __init__(self, model: tf.keras.models.Model, config: Optional[Dict] = None):
        """初始化模型训练器
        
        Args:
            model: 要训练的Keras模型
            config: 训练配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'decay_rate': 0.9,
            'decay_steps': 1000,
            'early_stopping_patience': 10,
            'early_stopping_monitor': 'val_loss',
            'reduce_lr_patience': 5,
            'reduce_lr_factor': 0.5,
            'reduce_lr_min_lr': 1e-6,
            'checkpoint_dir': './models/checkpoints',
            'log_dir': './logs',
            'tensorboard_log_dir': './logs/tensorboard',
            'model_save_dir': './models',
            'save_best_only': True,
            'monitor_metric': 'val_loss',
            'monitor_mode': 'min',
            'use_class_weights': False,
            'class_weights': None,
            'use_data_augmentation': False,
            'augmentation_params': {
                'rotation_range': 0,
                'width_shift_range': 0.1,
                'height_shift_range': 0,
                'shear_range': 0,
                'zoom_range': 0.1,
                'horizontal_flip': False,
                'vertical_flip': False
            },
            'callbacks': ['early_stopping', 'model_checkpoint', 'reduce_lr', 'tensorboard']
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 设置模型
        self.model = model
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 初始化训练历史
        self.history = None
        
        # 创建必要的目录
        self._create_directories()
        
        self.logger.info("模型训练器初始化完成")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('ModelTrainer')
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
            log_file = os.path.join(log_dir, 'model_training.log')
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
            self.config.get('checkpoint_dir', './models/checkpoints'),
            self.config.get('log_dir', './logs'),
            self.config.get('tensorboard_log_dir', './logs/tensorboard'),
            self.config.get('model_save_dir', './models')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.logger.info(f"创建目录: {directory}")
                except Exception as e:
                    self.logger.error(f"创建目录{directory}时发生异常: {str(e)}")
    
    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """获取训练回调函数列表
        
        Returns:
            Keras回调函数列表
        """
        callbacks = []
        
        # 获取配置
        callbacks_config = self.config.get('callbacks', [])
        checkpoint_dir = self.config.get('checkpoint_dir', './models/checkpoints')
        tensorboard_log_dir = self.config.get('tensorboard_log_dir', './logs/tensorboard')
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        early_stopping_monitor = self.config.get('early_stopping_monitor', 'val_loss')
        reduce_lr_patience = self.config.get('reduce_lr_patience', 5)
        reduce_lr_factor = self.config.get('reduce_lr_factor', 0.5)
        reduce_lr_min_lr = self.config.get('reduce_lr_min_lr', 1e-6)
        save_best_only = self.config.get('save_best_only', True)
        monitor_metric = self.config.get('monitor_metric', 'val_loss')
        monitor_mode = self.config.get('monitor_mode', 'min')
        
        # 生成时间戳用于文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 早停机制
        if 'early_stopping' in callbacks_config:
            self.logger.info(f"添加早停机制，耐心值: {early_stopping_patience}")
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # 模型检查点
        if 'model_checkpoint' in callbacks_config:
            self.logger.info(f"添加模型检查点，保存最佳模型: {save_best_only}")
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"model_checkpoint_{timestamp}_epoch_{{epoch:02d}}.keras"
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor_metric,
                save_best_only=save_best_only,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(model_checkpoint)
        
        # 学习率调度器
        if 'reduce_lr' in callbacks_config:
            self.logger.info(f"添加学习率调度器，耐心值: {reduce_lr_patience}")
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=reduce_lr_min_lr,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # TensorBoard
        if 'tensorboard' in callbacks_config:
            self.logger.info("添加TensorBoard回调")
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_log_dir, timestamp),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=2
            )
            callbacks.append(tensorboard)
        
        # CSV日志记录器
        if 'csv_logger' in callbacks_config:
            self.logger.info("添加CSV日志记录器")
            csv_logger = tf.keras.callbacks.CSVLogger(
                os.path.join(self.config.get('log_dir', './logs'), f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            )
            callbacks.append(csv_logger)
        
        return callbacks
    
    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """获取优化器
        
        Returns:
            Keras优化器
        """
        # 获取配置
        learning_rate = self.config.get('learning_rate', 0.001)
        decay_rate = self.config.get('decay_rate', 0.9)
        decay_steps = self.config.get('decay_steps', 1000)
        
        # 定义学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        # 创建优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.logger.info(f"使用Adam优化器，初始学习率: {learning_rate}")
        
        return optimizer
    
    def compile_model(self, loss: Union[str, tf.keras.losses.Loss] = None, 
                    metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None) -> None:
        """编译模型
        
        Args:
            loss: 损失函数，如果为None则使用默认的Huber损失
            metrics: 评估指标列表，如果为None则使用默认指标
        """
        try:
            self.logger.info("编译模型")
            
            # 设置默认损失函数
            if loss is None:
                loss = tf.keras.losses.Huber()
                self.logger.info("使用默认的Huber损失函数")
            
            # 设置默认评估指标
            if metrics is None:
                metrics = [
                    'mae',  # 平均绝对误差
                    'mse',  # 均方误差
                    tf.keras.metrics.R2Score()  # R2分数
                ]
                self.logger.info("使用默认评估指标: MAE, MSE, R2")
            
            # 获取优化器
            optimizer = self._get_optimizer()
            
            # 编译模型
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            self.logger.info("模型编译完成")
        except Exception as e:
            self.logger.error(f"编译模型时发生异常: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
             batch_size: Optional[int] = None, epochs: Optional[int] = None) -> Dict:
        """训练模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标数据
            X_val: 验证特征数据，如果为None则不进行验证
            y_val: 验证目标数据
            batch_size: 批次大小，如果为None则使用配置中的值
            epochs: 训练轮数，如果为None则使用配置中的值
        
        Returns:
            训练历史记录字典
        """
        try:
            # 获取批次大小和训练轮数
            if batch_size is None:
                batch_size = self.config.get('batch_size', 32)
            
            if epochs is None:
                epochs = self.config.get('epochs', 100)
            
            self.logger.info(f"开始训练模型，批次大小: {batch_size}，轮数: {epochs}")
            
            # 检查数据形状
            self.logger.info(f"训练数据形状 - X: {X_train.shape}, y: {y_train.shape}")
            if X_val is not None and y_val is not None:
                self.logger.info(f"验证数据形状 - X: {X_val.shape}, y: {y_val.shape}")
            
            # 准备验证数据
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # 获取回调函数
            callbacks = self._get_callbacks()
            
            # 获取类权重（如果启用）
            class_weight = None
            if self.config.get('use_class_weights', False) and self.config.get('class_weights') is not None:
                class_weight = self.config.get('class_weights')
                self.logger.info(f"使用类权重: {class_weight}")
            
            # 开始训练
            start_time = datetime.now()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
            
            # 计算训练时间
            end_time = datetime.now()
            training_time = end_time - start_time
            
            self.logger.info(f"模型训练完成，耗时: {training_time}")
            
            # 保存训练历史
            self._save_training_history()
            
            # 绘制训练历史
            self.plot_training_history()
            
            return self.history.history
        except Exception as e:
            self.logger.error(f"训练模型时发生异常: {str(e)}")
            raise
    
    def train_with_generator(self, train_generator: tf.keras.utils.Sequence, 
                           val_generator: Optional[tf.keras.utils.Sequence] = None, 
                           epochs: Optional[int] = None, 
                           max_queue_size: int = 10, 
                           workers: int = 1, 
                           use_multiprocessing: bool = False) -> Dict:
        """使用数据生成器训练模型
        
        Args:
            train_generator: 训练数据生成器
            val_generator: 验证数据生成器，如果为None则不进行验证
            epochs: 训练轮数，如果为None则使用配置中的值
            max_queue_size: 生成器队列的最大大小
            workers: 使用的线程数
            use_multiprocessing: 是否使用多进程
        
        Returns:
            训练历史记录字典
        """
        try:
            # 获取训练轮数
            if epochs is None:
                epochs = self.config.get('epochs', 100)
            
            self.logger.info(f"使用数据生成器训练模型，轮数: {epochs}")
            
            # 获取回调函数
            callbacks = self._get_callbacks()
            
            # 开始训练
            start_time = datetime.now()
            
            self.history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=1
            )
            
            # 计算训练时间
            end_time = datetime.now()
            training_time = end_time - start_time
            
            self.logger.info(f"模型训练完成，耗时: {training_time}")
            
            # 保存训练历史
            self._save_training_history()
            
            # 绘制训练历史
            self.plot_training_history()
            
            return self.history.history
        except Exception as e:
            self.logger.error(f"使用数据生成器训练模型时发生异常: {str(e)}")
            raise
    
    def _save_training_history(self) -> None:
        """保存训练历史到文件"""
        try:
            if self.history is None:
                self.logger.warning("没有训练历史，无法保存")
                return
            
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON文件
            log_dir = self.config.get('log_dir', './logs')
            history_file = os.path.join(log_dir, f'training_history_{timestamp}.json')
            
            # 转换numpy数组为Python列表以便JSON序列化
            history_dict = self.history.history
            serializable_history = {}
            for key, value in history_dict.items():
                if isinstance(value, np.ndarray):
                    serializable_history[key] = value.tolist()
                else:
                    serializable_history[key] = value
            
            # 保存到文件
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"训练历史已保存到: {history_file}")
        except Exception as e:
            self.logger.error(f"保存训练历史时发生异常: {str(e)}")
    
    def plot_training_history(self, metrics: Optional[List[str]] = None, 
                            save_path: Optional[str] = None) -> None:
        """绘制训练历史曲线
        
        Args:
            metrics: 要绘制的指标列表，如果为None则绘制所有指标
            save_path: 图像保存路径，如果为None则不保存
        """
        try:
            if self.history is None:
                self.logger.warning("没有训练历史，无法绘制")
                return
            
            # 获取要绘制的指标
            if metrics is None:
                metrics = list(self.history.history.keys())
            
            self.logger.info(f"绘制训练历史曲线，指标: {metrics}")
            
            # 计算需要的子图数量
            num_metrics = len(metrics)
            rows = (num_metrics + 1) // 2  # 每两行显示两个指标
            cols = min(num_metrics, 2)
            
            # 创建图形
            plt.figure(figsize=(12, rows * 5))
            
            # 绘制每个指标
            for i, metric in enumerate(metrics):
                plt.subplot(rows, cols, i + 1)
                
                # 绘制训练指标
                plt.plot(self.history.history[metric], label=f'训练 {metric}')
                
                # 绘制验证指标（如果存在）
                val_metric = f'val_{metric}'
                if val_metric in self.history.history:
                    plt.plot(self.history.history[val_metric], label=f'验证 {metric}')
                
                plt.title(f'{metric} 曲线')
                plt.xlabel('轮数')
                plt.ylabel(metric)
                plt.legend()
            
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"训练历史图像已保存到: {save_path}")
            else:
                # 生成时间戳用于文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_dir = self.config.get('log_dir', './logs')
                default_save_path = os.path.join(log_dir, f'training_history_plot_{timestamp}.png')
                plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"训练历史图像已保存到默认路径: {default_save_path}")
                
                # 显示图像
                plt.show()
            
            # 关闭图形，释放内存
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制训练历史曲线时发生异常: {str(e)}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                batch_size: Optional[int] = None) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            X_test: 测试特征数据
            y_test: 测试目标数据
            batch_size: 批次大小，如果为None则使用配置中的值
        
        Returns:
            评估指标字典
        """
        try:
            # 获取批次大小
            if batch_size is None:
                batch_size = self.config.get('batch_size', 32)
            
            self.logger.info(f"评估模型性能，批次大小: {batch_size}")
            
            # 检查数据形状
            self.logger.info(f"测试数据形状 - X: {X_test.shape}, y: {y_test.shape}")
            
            # 执行评估
            eval_results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
            
            # 将评估结果转换为字典
            metrics_names = self.model.metrics_names
            metrics_dict = {metrics_names[i]: eval_results[i] for i in range(len(metrics_names))}
            
            # 计算额外的评估指标
            y_pred = self.model.predict(X_test, batch_size=batch_size)
            
            # 计算准确率（这里使用阈值0.5判断涨跌方向）
            if y_test.shape[0] > 1:
                y_test_direction = np.sign(y_test[1:] - y_test[:-1])
                y_pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
                direction_accuracy = np.mean(y_test_direction == y_pred_direction)
                metrics_dict['direction_accuracy'] = direction_accuracy
            
            # 计算均方根误差
            mse = metrics_dict.get('mse', np.mean((y_test - y_pred) ** 2))
            rmse = np.sqrt(mse)
            metrics_dict['rmse'] = rmse
            
            # 计算平均绝对百分比误差
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            metrics_dict['mape'] = mape
            
            # 计算对称平均绝对百分比误差
            smape = (100 / len(y_test)) * np.sum(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred)))
            metrics_dict['smape'] = smape
            
            # 计算预测偏差
            bias = np.mean(y_pred - y_test)
            metrics_dict['bias'] = bias
            
            # 打印评估指标
            self.logger.info(f"模型评估结果: {metrics_dict}")
            
            # 保存评估结果
            self._save_evaluation_results(metrics_dict)
            
            return metrics_dict
        except Exception as e:
            self.logger.error(f"评估模型性能时发生异常: {str(e)}")
            raise
    
    def _save_evaluation_results(self, metrics_dict: Dict[str, float]) -> None:
        """保存评估结果到文件
        
        Args:
            metrics_dict: 评估指标字典
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON文件
            log_dir = self.config.get('log_dir', './logs')
            eval_file = os.path.join(log_dir, f'model_evaluation_{timestamp}.json')
            
            # 保存到文件
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"评估结果已保存到: {eval_file}")
        except Exception as e:
            self.logger.error(f"保存评估结果时发生异常: {str(e)}")
    
    def save_model(self, model_path: Optional[str] = None, 
                  include_optimizer: bool = True, 
                  save_format: str = 'keras') -> str:
        """保存模型到文件
        
        Args:
            model_path: 模型保存路径，如果为None则使用默认路径
            include_optimizer: 是否包含优化器状态
            save_format: 保存格式，支持'keras'和'h5'
        
        Returns:
            实际的模型保存路径
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置默认保存路径
            if model_path is None:
                model_save_dir = self.config.get('model_save_dir', './models')
                model_path = os.path.join(model_save_dir, f'model_{timestamp}.{save_format}')
            
            self.logger.info(f"保存模型到: {model_path}")
            
            # 保存模型
            self.model.save(
                model_path,
                include_optimizer=include_optimizer,
                save_format=save_format
            )
            
            self.logger.info("模型保存完成")
            
            return model_path
        except Exception as e:
            self.logger.error(f"保存模型时发生异常: {str(e)}")
            raise
    
    def save_best_model(self, metrics_dict: Optional[Dict[str, float]] = None) -> str:
        """保存最佳模型
        
        Args:
            metrics_dict: 评估指标字典，如果为None则使用默认指标
        
        Returns:
            最佳模型的保存路径
        """
        try:
            # 生成时间戳用于文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 构建文件名，包含关键指标
            model_save_dir = self.config.get('model_save_dir', './models')
            
            if metrics_dict is None:
                # 如果没有评估指标，使用简单的文件名
                best_model_path = os.path.join(model_save_dir, f'best_model_{timestamp}.keras')
            else:
                # 获取关键指标
                monitor_metric = self.config.get('monitor_metric', 'val_loss')
                if monitor_metric in metrics_dict:
                    metric_value = metrics_dict[monitor_metric]
                    # 格式化指标值，保留4位小数
                    formatted_metric = f"{metric_value:.4f}"
                else:
                    formatted_metric = "unknown"
                    
                # 构建包含指标的文件名
                best_model_path = os.path.join(
                    model_save_dir,
                    f'best_model_{timestamp}_{monitor_metric}_{formatted_metric}.keras'
                )
            
            self.logger.info(f"保存最佳模型到: {best_model_path}")
            
            # 保存模型
            self.model.save(best_model_path)
            
            # 保存配置文件
            config_path = best_model_path.replace('.keras', '.json').replace('.h5', '.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            # 如果有评估指标，保存评估结果
            if metrics_dict is not None:
                eval_path = best_model_path.replace('.keras', '_evaluation.json').replace('.h5', '_evaluation.json')
                with open(eval_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info("最佳模型保存完成")
            
            return best_model_path
        except Exception as e:
            self.logger.error(f"保存最佳模型时发生异常: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> bool:
        """从文件加载模型
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载模型: {model_path}")
            
            # 加载模型
            self.model = tf.keras.models.load_model(model_path)
            
            # 尝试加载配置文件
            config_path = model_path.replace('.keras', '.json').replace('.h5', '.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                
                self.logger.info(f"模型配置已从{config_path}加载")
            
            self.logger.info("模型加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载模型时发生异常: {str(e)}")
            return False
    
    def get_model_summary(self) -> str:
        """获取模型摘要
        
        Returns:
            模型摘要字符串
        """
        try:
            self.logger.info("获取模型摘要")
            
            # 使用临时缓冲区捕获模型摘要
            import io
            buffer = io.StringIO()
            self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            summary_str = buffer.getvalue()
            buffer.close()
            
            # 打印模型摘要到日志
            self.logger.info(f"模型摘要:\n{summary_str}")
            
            # 保存模型摘要到文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = self.config.get('log_dir', './logs')
            summary_file = os.path.join(log_dir, f'model_summary_{timestamp}.txt')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_str)
            
            self.logger.info(f"模型摘要已保存到: {summary_file}")
            
            return summary_str
        except Exception as e:
            self.logger.error(f"获取模型摘要时发生异常: {str(e)}")
            raise
    
    def plot_model_architecture(self, save_path: Optional[str] = None, 
                              show_shapes: bool = True, 
                              show_layer_names: bool = True, 
                              rankdir: str = 'TB') -> None:
        """绘制模型架构图
        
        Args:
            save_path: 图像保存路径，如果为None则不保存
            show_shapes: 是否显示输入输出形状
            show_layer_names: 是否显示层名称
            rankdir: 绘图方向，'TB'表示从上到下，'LR'表示从左到右
        """
        try:
            self.logger.info("绘制模型架构图")
            
            # 使用tf.keras.utils.plot_model绘制模型架构
            if save_path is None:
                # 生成时间戳用于文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_dir = self.config.get('log_dir', './logs')
                save_path = os.path.join(log_dir, f'model_architecture_{timestamp}.png')
            
            # 绘制模型架构
            tf.keras.utils.plot_model(
                self.model,
                to_file=save_path,
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                rankdir=rankdir,
                dpi=300
            )
            
            self.logger.info(f"模型架构图已保存到: {save_path}")
        except Exception as e:
            self.logger.error(f"绘制模型架构图时发生异常: {str(e)}")
            raise
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5, 
                     batch_size: Optional[int] = None, 
                     epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """执行交叉验证
        
        Args:
            X: 特征数据
            y: 目标数据
            n_folds: 交叉验证的折数
            batch_size: 批次大小，如果为None则使用配置中的值
            epochs: 训练轮数，如果为None则使用配置中的值
        
        Returns:
            每折的评估指标列表字典
        """
        try:
            from sklearn.model_selection import KFold
            
            # 获取批次大小和训练轮数
            if batch_size is None:
                batch_size = self.config.get('batch_size', 32)
            
            if epochs is None:
                epochs = self.config.get('epochs', 100)
            
            self.logger.info(f"执行交叉验证，折数: {n_folds}，批次大小: {batch_size}，轮数: {epochs}")
            
            # 初始化KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            # 存储每折的评估结果
            fold_results = {}
            
            # 执行交叉验证
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                self.logger.info(f"开始交叉验证第{fold+1}/{n_folds}折")
                
                # 分割数据
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 克隆模型以避免折叠间的影响
                from tensorflow.keras.models import clone_model
                fold_model = clone_model(self.model)
                
                # 编译模型
                optimizer = self._get_optimizer()
                loss = self.model.loss
                metrics = self.model.metrics
                fold_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                # 备份原始模型并使用当前折叠模型
                original_model = self.model
                self.model = fold_model
                
                try:
                    # 训练模型
                    self.train(X_train, y_train, X_val, y_val, batch_size, epochs)
                    
                    # 评估模型
                    fold_metrics = self.evaluate(X_val, y_val, batch_size)
                    
                    # 存储结果
                    for metric, value in fold_metrics.items():
                        if metric not in fold_results:
                            fold_results[metric] = []
                        fold_results[metric].append(value)
                        
                    self.logger.info(f"第{fold+1}折验证结果: {fold_metrics}")
                finally:
                    # 恢复原始模型
                    self.model = original_model
            
            # 计算平均指标
            avg_results = {}
            for metric, values in fold_results.items():
                avg_results[f'{metric}_avg'] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
                
            self.logger.info(f"交叉验证平均结果: {avg_results}")
            
            # 保存交叉验证结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = self.config.get('log_dir', './logs')
            cv_file = os.path.join(log_dir, f'cross_validation_results_{timestamp}.json')
            
            with open(cv_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'fold_results': fold_results,
                    'avg_results': avg_results,
                    'config': self.config
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"交叉验证结果已保存到: {cv_file}")
            
            return fold_results
        except Exception as e:
            self.logger.error(f"执行交叉验证时发生异常: {str(e)}")
            raise