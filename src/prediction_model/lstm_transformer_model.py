"""LSTM-Transformer融合架构的股价预测模型实现"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention, Add, Concatenate,
    Reshape, Bidirectional
)
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

class LSTMTransformerModel:
    """LSTM-Transformer融合架构的股价预测模型"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化LSTM-Transformer模型
        
        Args:
            config: 模型配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.default_config = {
            'input_shape': (60, 128),  # 时间步长 × 特征维度
            'lstm_layers': [
                {'units': 128, 'return_sequences': True, 'bidirectional': True},
                {'units': 64, 'return_sequences': False, 'bidirectional': False}
            ],
            'transformer_layers': 2,
            'num_heads': 4,
            'd_model': 128,
            'dff': 256,  # 前馈网络的隐藏层维度
            'dropout_rate': 0.3,
            'output_dim': 1,  # 输出维度，预测未来1天的收盘价
            'use_attention_on_lstm': True,  # 是否在LSTM输出上应用注意力机制
            'use_feature_attention': True,  # 是否使用特征级注意力
            'use_positional_encoding': True,  # 是否使用位置编码
            'time_horizon': 1,  # 预测的时间跨度（天数）
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'decay_rate': 0.9,
            'decay_steps': 1000
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config is not None else self.default_config
        
        # 初始化日志
        self.logger = self._init_logger()
        
        # 初始化模型
        self.model = None
        self.history = None
        
        # 创建模型
        self._build_model()
        
        self.logger.info("LSTM-Transformer模型初始化完成")
    
    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger('LSTMTransformerModel')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(console_handler)
        
        return logger
    
    def _positional_encoding(self, position: int, d_model: int) -> np.ndarray:
        """生成位置编码
        
        Args:
            position: 位置数量
            d_model: 模型维度
        
        Returns:
            位置编码矩阵
        """
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(
            10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
        )
        
        # 应用sin到偶数索引，cos到奇数索引
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _feature_attention(self, inputs: tf.Tensor) -> tf.Tensor:
        """特征级注意力机制
        
        Args:
            inputs: 输入张量 [batch_size, seq_len, feature_dim]
        
        Returns:
            注意力加权后的特征张量
        """
        # 获取输入维度
        _, seq_len, feature_dim = inputs.shape
        
        # 计算特征注意力权重
        attention_weights = Dense(feature_dim, activation='softmax')(inputs)
        
        # 应用注意力权重
        attended_features = tf.multiply(inputs, attention_weights)
        
        return attended_features
    
    def _transformer_encoder_layer(self, inputs: tf.Tensor, d_model: int, 
                                 num_heads: int, dff: int, 
                                 dropout_rate: float) -> tf.Tensor:
        """Transformer编码器层
        
        Args:
            inputs: 输入张量
            d_model: 模型维度
            num_heads: 注意力头数量
            dff: 前馈网络的隐藏层维度
            dropout_rate: Dropout比率
        
        Returns:
            编码器层的输出张量
        """
        # 多头自注意力机制
        attention_output = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout_rate
        )(inputs, inputs)
        
        # 添加与规范化
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        
        # 前馈网络
        ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        # 应用前馈网络
        ffn_output = ffn(attention_output)
        
        # 添加与规范化
        encoder_output = Add()([attention_output, ffn_output])
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output)
        
        return encoder_output
    
    def _build_model(self) -> None:
        """构建LSTM-Transformer融合模型"""
        try:
            self.logger.info("开始构建LSTM-Transformer融合模型")
            
            # 获取配置参数
            input_shape = self.config['input_shape']
            lstm_layers = self.config['lstm_layers']
            transformer_layers = self.config['transformer_layers']
            num_heads = self.config['num_heads']
            d_model = self.config['d_model']
            dff = self.config['dff']
            dropout_rate = self.config['dropout_rate']
            output_dim = self.config['output_dim']
            use_attention_on_lstm = self.config['use_attention_on_lstm']
            use_feature_attention = self.config['use_feature_attention']
            use_positional_encoding = self.config['use_positional_encoding']
            learning_rate = self.config['learning_rate']
            decay_rate = self.config['decay_rate']
            decay_steps = self.config['decay_steps']
            
            # 输入层
            inputs = Input(shape=input_shape, name='input_layer')
            x = inputs
            
            # 特征级注意力（如果启用）
            if use_feature_attention:
                self.logger.info("添加特征级注意力层")
                x = self._feature_attention(x)
            
            # 位置编码（如果启用）
            if use_positional_encoding:
                self.logger.info("添加位置编码")
                seq_len = input_shape[0]
                pos_encoding = self._positional_encoding(seq_len, input_shape[1])
                x = Add()([x, pos_encoding])
            
            # LSTM层
            self.logger.info(f"添加{len(lstm_layers)}个LSTM层")
            lstm_outputs = []
            
            for i, lstm_config in enumerate(lstm_layers):
                units = lstm_config['units']
                return_sequences = lstm_config['return_sequences']
                bidirectional = lstm_config['bidirectional']
                
                if bidirectional:
                    lstm_layer = Bidirectional(
                        LSTM(units, return_sequences=return_sequences, 
                             activation='relu', name=f'lstm_bi_{i+1}')
                    )
                else:
                    lstm_layer = LSTM(units, return_sequences=return_sequences, 
                                    activation='relu', name=f'lstm_{i+1}')
                
                x = lstm_layer(x)
                x = BatchNormalization(name=f'lstm_batch_norm_{i+1}')(x)
                x = Dropout(dropout_rate, name=f'lstm_dropout_{i+1}')(x)
                
                # 保存LSTM层的输出用于可能的融合
                if return_sequences:
                    lstm_outputs.append(x)
            
            # 如果最后一个LSTM层返回序列，并且启用了LSTM输出的注意力机制
            if lstm_layers[-1]['return_sequences'] and use_attention_on_lstm:
                self.logger.info("添加LSTM输出的注意力机制")
                
                # 确保维度匹配
                if x.shape[-1] != d_model:
                    x = Dense(d_model, activation='relu', name='lstm_to_transformer')(x)
                
                # Transformer编码器层
                for i in range(transformer_layers):
                    x = self._transformer_encoder_layer(
                        x, d_model, num_heads, dff, dropout_rate
                    )
                    x = Dropout(dropout_rate, name=f'transformer_dropout_{i+1}')(x)
            
            # 特征融合（如果有多个LSTM层返回序列）
            if len(lstm_outputs) > 1 and not lstm_layers[-1]['return_sequences']:
                self.logger.info("融合多个LSTM层的输出")
                # 调整最后一个LSTM层的输出维度以匹配其他层
                last_lstm_output = Reshape((1, x.shape[-1]))(x)
                # 合并所有LSTM输出
                x = Concatenate(axis=1)(lstm_outputs + [last_lstm_output])
                # 应用注意力机制
                x = MultiHeadAttention(
                    key_dim=x.shape[-1], num_heads=num_heads, dropout=dropout_rate
                )(x, x)
            
            # 全连接层用于预测
            self.logger.info("添加预测层")
            x = Dense(64, activation='relu', name='dense_1')(x)
            x = Dropout(dropout_rate, name='dense_dropout_1')(x)
            x = Dense(32, activation='relu', name='dense_2')(x)
            x = Dropout(dropout_rate, name='dense_dropout_2')(x)
            
            # 输出层
            outputs = Dense(output_dim, name='output_layer')(x)
            
            # 创建模型
            self.model = Model(inputs=inputs, outputs=outputs, name='LSTMTransformer')
            
            # 定义学习率调度器
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True
            )
            
            # 编译模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            # 使用Huber损失函数，对异常值更稳健
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.Huber(),
                metrics=['mae', 'mse', tf.keras.metrics.R2Score()]
            )
            
            # 打印模型摘要
            self.model.summary(print_fn=lambda x: self.logger.info(x))
            
            self.logger.info("LSTM-Transformer融合模型构建完成")
        except Exception as e:
            self.logger.error(f"构建模型时发生异常: {str(e)}")
            raise
    
    def load_config(self, config_path: str) -> bool:
        """从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载配置: {config_path}")
            
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # 更新配置
            self.config.update(new_config)
            
            # 重新构建模型
            self._build_model()
            
            self.logger.info("配置加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载配置时发生异常: {str(e)}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """保存配置到YAML文件
        
        Args:
            config_path: 配置文件保存路径
        
        Returns:
            是否保存成功
        """
        try:
            self.logger.info(f"保存配置到文件: {config_path}")
            
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info("配置保存完成")
            return True
        except Exception as e:
            self.logger.error(f"保存配置时发生异常: {str(e)}")
            return False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray, 
             batch_size: int = 32, epochs: int = 100) -> Dict:
        """训练模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标数据
            X_val: 验证特征数据
            y_val: 验证目标数据
            batch_size: 批次大小
            epochs: 训练轮数
        
        Returns:
            训练历史记录字典
        """
        try:
            self.logger.info(f"开始训练模型，批次大小: {batch_size}，轮数: {epochs}")
            
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法训练")
                raise ValueError("模型未构建")
            
            # 定义早停机制
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
            
            # 定义模型检查点
            checkpoint_path = f"model_checkpoint_epoch_{{epoch:02d}}.keras"
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            # 定义学习率调度器回调
            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # 训练模型
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint, lr_reducer],
                verbose=1
            )
            
            self.logger.info("模型训练完成")
            
            return self.history.history
        except Exception as e:
            self.logger.error(f"训练模型时发生异常: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测
        
        Args:
            X: 输入特征数据
        
        Returns:
            预测结果数组
        """
        try:
            self.logger.info(f"进行预测，输入数据形状: {X.shape}")
            
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法预测")
                raise ValueError("模型未构建")
            
            # 执行预测
            predictions = self.model.predict(X)
            
            self.logger.info(f"预测完成，输出数据形状: {predictions.shape}")
            
            return predictions
        except Exception as e:
            self.logger.error(f"预测时发生异常: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            X_test: 测试特征数据
            y_test: 测试目标数据
        
        Returns:
            评估指标字典
        """
        try:
            self.logger.info(f"评估模型性能，测试数据形状: {X_test.shape}")
            
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法评估")
                raise ValueError("模型未构建")
            
            # 执行评估
            loss, mae, mse, r2 = self.model.evaluate(X_test, y_test, verbose=1)
            
            # 计算额外的评估指标
            y_pred = self.model.predict(X_test)
            
            # 计算准确率（这里使用阈值0.5判断涨跌方向）
            y_test_direction = np.sign(y_test[1:] - y_test[:-1])
            y_pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            direction_accuracy = np.mean(y_test_direction == y_pred_direction)
            
            # 计算均方根误差
            rmse = np.sqrt(mse)
            
            # 计算平均绝对百分比误差
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # 汇总评估指标
            metrics = {
                'loss': loss,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
            self.logger.info(f"模型评估完成，指标: {metrics}")
            
            return metrics
        except Exception as e:
            self.logger.error(f"评估模型时发生异常: {str(e)}")
            raise
    
    def save_model(self, model_path: str) -> bool:
        """保存模型到文件
        
        Args:
            model_path: 模型保存路径
        
        Returns:
            是否保存成功
        """
        try:
            self.logger.info(f"保存模型到文件: {model_path}")
            
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法保存")
                return False
            
            # 保存完整模型
            self.model.save(model_path)
            
            # 保存模型配置
            config_path = model_path.replace('.keras', '.yaml').replace('.h5', '.yaml')
            self.save_config(config_path)
            
            self.logger.info("模型保存完成")
            return True
        except Exception as e:
            self.logger.error(f"保存模型时发生异常: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """从文件加载模型
        
        Args:
            model_path: 模型加载路径
        
        Returns:
            是否加载成功
        """
        try:
            self.logger.info(f"从文件加载模型: {model_path}")
            
            # 尝试加载完整模型
            self.model = tf.keras.models.load_model(model_path)
            
            # 尝试加载模型配置
            config_path = model_path.replace('.keras', '.yaml').replace('.h5', '.yaml')
            if tf.io.gfile.exists(config_path):
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            
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
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法获取摘要")
                return "模型未构建"
            
            # 获取模型摘要
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            
            return '\n'.join(summary)
        except Exception as e:
            self.logger.error(f"获取模型摘要时发生异常: {str(e)}")
            return f"获取摘要失败: {str(e)}"
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """绘制训练历史
        
        Args:
            save_path: 图像保存路径，如果为None则不保存
        """
        try:
            # 检查是否有训练历史
            if self.history is None:
                self.logger.error("没有训练历史，无法绘制")
                return
            
            import matplotlib.pyplot as plt
            
            # 创建图形
            plt.figure(figsize=(12, 10))
            
            # 绘制损失曲线
            plt.subplot(2, 2, 1)
            plt.plot(self.history.history['loss'], label='训练损失')
            plt.plot(self.history.history['val_loss'], label='验证损失')
            plt.title('损失曲线')
            plt.xlabel('轮数')
            plt.ylabel('损失值')
            plt.legend()
            
            # 绘制MAE曲线
            plt.subplot(2, 2, 2)
            plt.plot(self.history.history['mae'], label='训练MAE')
            plt.plot(self.history.history['val_mae'], label='验证MAE')
            plt.title('MAE曲线')
            plt.xlabel('轮数')
            plt.ylabel('MAE值')
            plt.legend()
            
            # 绘制MSE曲线
            plt.subplot(2, 2, 3)
            plt.plot(self.history.history['mse'], label='训练MSE')
            plt.plot(self.history.history['val_mse'], label='验证MSE')
            plt.title('MSE曲线')
            plt.xlabel('轮数')
            plt.ylabel('MSE值')
            plt.legend()
            
            # 绘制R2曲线
            if 'r2' in self.history.history and 'val_r2' in self.history.history:
                plt.subplot(2, 2, 4)
                plt.plot(self.history.history['r2'], label='训练R2')
                plt.plot(self.history.history['val_r2'], label='验证R2')
                plt.title('R2曲线')
                plt.xlabel('轮数')
                plt.ylabel('R2值')
                plt.legend()
            
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"训练历史图像已保存到: {save_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"绘制训练历史时发生异常: {str(e)}")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray, 
                           param_grid: Optional[Dict] = None, 
                           n_trials: int = 10) -> Dict:
        """超参数调优
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标数据
            X_val: 验证特征数据
            y_val: 验证目标数据
            param_grid: 超参数网格
            n_trials: 试验次数
        
        Returns:
            最佳超参数配置字典
        """
        try:
            self.logger.info(f"开始超参数调优，试验次数: {n_trials}")
            
            # 导入必要的库
            import optuna
            
            # 设置默认超参数网格
            if param_grid is None:
                param_grid = {
                    'lstm_units': [32, 64, 128, 256],
                    'transformer_layers': [1, 2, 3],
                    'num_heads': [2, 4, 8],
                    'dff': [64, 128, 256, 512],
                    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
                    'learning_rate': [0.0001, 0.0005, 0.001, 0.005]
                }
            
            # 定义目标函数
            def objective(trial):
                # 采样超参数
                lstm_units = trial.suggest_categorical('lstm_units', param_grid['lstm_units'])
                transformer_layers = trial.suggest_categorical('transformer_layers', param_grid['transformer_layers'])
                num_heads = trial.suggest_categorical('num_heads', param_grid['num_heads'])
                dff = trial.suggest_categorical('dff', param_grid['dff'])
                dropout_rate = trial.suggest_categorical('dropout_rate', param_grid['dropout_rate'])
                learning_rate = trial.suggest_categorical('learning_rate', param_grid['learning_rate'])
                
                # 更新配置
                self.config['lstm_layers'] = [
                    {'units': lstm_units, 'return_sequences': True, 'bidirectional': True},
                    {'units': lstm_units // 2, 'return_sequences': False, 'bidirectional': False}
                ]
                self.config['transformer_layers'] = transformer_layers
                self.config['num_heads'] = num_heads
                self.config['dff'] = dff
                self.config['dropout_rate'] = dropout_rate
                self.config['learning_rate'] = learning_rate
                
                # 重建模型
                self._build_model()
                
                # 训练模型
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
                    verbose=0
                )
                
                # 返回验证损失作为优化目标
                return history.history['val_loss'][-1]
            
            # 创建Optuna优化器
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # 获取最佳超参数
            best_params = study.best_params
            
            self.logger.info(f"超参数调优完成，最佳参数: {best_params}")
            
            # 更新模型配置为最佳参数
            self.config['lstm_layers'] = [
                {'units': best_params['lstm_units'], 'return_sequences': True, 'bidirectional': True},
                {'units': best_params['lstm_units'] // 2, 'return_sequences': False, 'bidirectional': False}
            ]
            self.config['transformer_layers'] = best_params['transformer_layers']
            self.config['num_heads'] = best_params['num_heads']
            self.config['dff'] = best_params['dff']
            self.config['dropout_rate'] = best_params['dropout_rate']
            self.config['learning_rate'] = best_params['learning_rate']
            
            # 重建模型
            self._build_model()
            
            return best_params
        except Exception as e:
            self.logger.error(f"超参数调优时发生异常: {str(e)}")
            raise
    
    def explain_predictions(self, X: np.ndarray, sample_indices: List[int] = None, 
                          method: str = 'shap') -> Dict:
        """解释模型预测结果
        
        Args:
            X: 输入特征数据
            sample_indices: 要解释的样本索引列表
            method: 解释方法，支持'shap'和'lime'
        
        Returns:
            解释结果字典
        """
        try:
            self.logger.info(f"解释模型预测结果，方法: {method}")
            
            # 检查模型是否已构建
            if self.model is None:
                self.logger.error("模型未构建，无法解释")
                raise ValueError("模型未构建")
            
            # 选择样本
            if sample_indices is None:
                sample_indices = [0, 1, 2]  # 默认选择前3个样本
            
            # 根据方法进行解释
            if method.lower() == 'shap':
                import shap
                
                # 创建SHAP解释器
                explainer = shap.DeepExplainer(self.model, X[:100])  # 使用前100个样本作为背景数据
                
                # 选择要解释的样本
                samples = X[sample_indices]
                
                # 计算SHAP值
                shap_values = explainer.shap_values(samples)
                
                # 汇总解释结果
                explanations = {
                    'shap_values': shap_values,
                    'samples': samples,
                    'base_values': explainer.expected_value
                }
                
                # 生成SHAP总结图
                plt.figure()
                shap.summary_plot(shap_values[0], samples)
                
                self.logger.info("SHAP解释完成")
                
                return explanations
            elif method.lower() == 'lime':
                from lime import lime_tabular
                
                # 准备数据
                X_flat = X.reshape(X.shape[0], -1)  # 展平数据以适应LIME
                
                # 创建LIME解释器
                explainer = lime_tabular.LimeTabularExplainer(
                    X_flat, 
                    feature_names=[f'feature_{i}' for i in range(X_flat.shape[1])],
                    class_names=['price'],
                    mode='regression'
                )
                
                explanations = {}
                
                # 对每个样本进行解释
                for idx in sample_indices:
                    # 解释单个样本
                    exp = explainer.explain_instance(
                        X_flat[idx], 
                        lambda x: self.model.predict(x.reshape(x.shape[0], X.shape[1], X.shape[2])),
                        num_features=10
                    )
                    
                    explanations[idx] = exp
                
                self.logger.info("LIME解释完成")
                
                return explanations
            else:
                self.logger.error(f"不支持的解释方法: {method}")
                raise ValueError(f"不支持的解释方法: {method}")
        except Exception as e:
            self.logger.error(f"解释模型预测结果时发生异常: {str(e)}")
            raise