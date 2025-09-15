"""特征选择与降维模块，负责从高维特征中选择最重要的特征"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple
from sklearn.feature_selection import (SelectKBest, SelectFromModel, 
                                     f_classif, mutual_info_classif, 
                                     RFE, VarianceThreshold)
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class FeatureSelection:
    """特征选择与降维类，封装了特征选择和降维的核心功能"""
    
    def __init__(self):
        """初始化特征选择与降维类"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("特征选择与降维模块初始化完成")
        self.selected_features = None
        self.reduction_model = None
        self.scaler = None
        self.feature_importance = None
    
    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """移除低方差特征
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            threshold: 方差阈值，方差低于此值的特征将被移除
        
        Returns:
            移除低方差特征后的特征矩阵
        """
        try:
            self.logger.info(f"开始移除低方差特征，阈值: {threshold}")
            
            # 初始化方差阈值选择器
            selector = VarianceThreshold(threshold=threshold)
            
            # 拟合数据
            selector.fit(X)
            
            # 获取保留的特征索引
            selected_indices = np.where(selector.variances_ > threshold)[0]
            selected_columns = X.columns[selected_indices]
            
            # 筛选特征
            X_reduced = X[selected_columns].copy()
            
            self.logger.info(f"移除低方差特征完成，原始特征数: {X.shape[1]}，保留特征数: {X_reduced.shape[1]}")
            
            return X_reduced
        except Exception as e:
            self.logger.error(f"移除低方差特征时发生异常: {str(e)}")
            return X
    
    def remove_high_correlation_features(self, X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """移除高相关性特征
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            threshold: 相关性阈值，相关性高于此值的特征对将移除其中一个
        
        Returns:
            移除高相关性特征后的特征矩阵
        """
        try:
            self.logger.info(f"开始移除高相关性特征，阈值: {threshold}")
            
            # 计算特征相关性矩阵
            corr_matrix = X.corr().abs()
            
            # 选择上三角矩阵
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # 找到相关性高于阈值的特征列名
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            # 移除高相关性特征
            X_reduced = X.drop(to_drop, axis=1)
            
            self.logger.info(f"移除高相关性特征完成，原始特征数: {X.shape[1]}，保留特征数: {X_reduced.shape[1]}")
            self.logger.debug(f"移除的特征: {to_drop}")
            
            return X_reduced
        except Exception as e:
            self.logger.error(f"移除高相关性特征时发生异常: {str(e)}")
            return X
    
    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 128, 
                             method: str = 'f_classif') -> pd.DataFrame:
        """使用SelectKBest选择最佳k个特征
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            k: 要选择的特征数量
            method: 评分方法，可选'f_classif', 'mutual_info_classif'
        
        Returns:
            选择的k个最佳特征的特征矩阵
        """
        try:
            self.logger.info(f"开始使用SelectKBest选择最佳特征，特征数: {k}，方法: {method}")
            
            # 选择评分函数
            if method == 'f_classif':
                score_func = f_classif
            elif method == 'mutual_info_classif':
                score_func = mutual_info_classif
            else:
                self.logger.error(f"不支持的评分方法: {method}")
                return X.iloc[:, :k]  # 默认返回前k个特征
            
            # 确保k不超过特征总数
            k = min(k, X.shape[1])
            
            # 初始化SelectKBest选择器
            selector = SelectKBest(score_func=score_func, k=k)
            
            # 拟合数据并转换
            selector.fit(X, y)
            
            # 获取选中的特征索引
            selected_indices = selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            
            # 获取特征评分
            scores = selector.scores_
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'score': scores
            }).sort_values('score', ascending=False)
            
            # 筛选特征
            X_selected = X[self.selected_features].copy()
            
            self.logger.info(f"SelectKBest特征选择完成，原始特征数: {X.shape[1]}，选择特征数: {X_selected.shape[1]}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"使用SelectKBest选择特征时发生异常: {str(e)}")
            return X.iloc[:, :min(k, X.shape[1])]
    
    def select_from_model(self, X: pd.DataFrame, y: pd.Series, k: int = 128, 
                        model_type: str = 'random_forest', **model_kwargs) -> pd.DataFrame:
        """使用SelectFromModel选择特征
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            k: 目标特征数量
            model_type: 基础模型类型，可选'random_forest', 'lasso', 'ridge', 'elasticnet'
            **model_kwargs: 传递给基础模型的参数
        
        Returns:
            选择的特征矩阵
        """
        try:
            self.logger.info(f"开始使用SelectFromModel选择特征，目标特征数: {k}，模型类型: {model_type}")
            
            # 选择基础模型
            if model_type == 'random_forest':
                base_model = RandomForestRegressor(n_estimators=100, **model_kwargs)
            elif model_type == 'lasso':
                base_model = Lasso(alpha=0.01, **model_kwargs)
            elif model_type == 'ridge':
                base_model = Ridge(alpha=1.0, **model_kwargs)
            elif model_type == 'elasticnet':
                base_model = ElasticNet(alpha=0.01, l1_ratio=0.5, **model_kwargs)
            else:
                self.logger.error(f"不支持的模型类型: {model_type}")
                return X.iloc[:, :min(k, X.shape[1])]
            
            # 计算所需的阈值来选择k个特征
            base_model.fit(X, y)
            
            # 获取特征重要性
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
            elif hasattr(base_model, 'coef_'):
                importances = np.abs(base_model.coef_)
            else:
                self.logger.error("基础模型没有特征重要性属性")
                return X.iloc[:, :min(k, X.shape[1])]
            
            # 对特征重要性排序
            sorted_importances = np.sort(importances)[::-1]
            
            # 确保k不超过特征总数
            k = min(k, X.shape[1])
            
            # 设置阈值以选择前k个特征
            threshold = sorted_importances[k-1] if k > 0 else 0
            
            # 初始化SelectFromModel选择器
            selector = SelectFromModel(base_model, threshold=threshold, prefit=True)
            
            # 获取选中的特征索引
            selected_indices = selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            
            # 创建特征重要性DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 筛选特征
            X_selected = X[self.selected_features].copy()
            
            self.logger.info(f"SelectFromModel特征选择完成，原始特征数: {X.shape[1]}，选择特征数: {X_selected.shape[1]}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"使用SelectFromModel选择特征时发生异常: {str(e)}")
            return X.iloc[:, :min(k, X.shape[1])]
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, k: int = 128, 
                                    model_type: str = 'linear', **model_kwargs) -> pd.DataFrame:
        """使用递归特征消除(RFE)选择特征
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            k: 要选择的特征数量
            model_type: 基础模型类型，可选'linear', 'random_forest'
            **model_kwargs: 传递给基础模型的参数
        
        Returns:
            选择的特征矩阵
        """
        try:
            self.logger.info(f"开始使用递归特征消除选择特征，目标特征数: {k}，模型类型: {model_type}")
            
            # 选择基础模型
            if model_type == 'linear':
                base_model = LinearRegression(**model_kwargs)
            elif model_type == 'random_forest':
                base_model = RandomForestRegressor(n_estimators=100, **model_kwargs)
            else:
                self.logger.error(f"不支持的模型类型: {model_type}")
                return X.iloc[:, :min(k, X.shape[1])]
            
            # 确保k不超过特征总数
            k = min(k, X.shape[1])
            
            # 初始化RFE选择器
            selector = RFE(estimator=base_model, n_features_to_select=k, step=1)
            
            # 拟合数据并转换
            selector.fit(X, y)
            
            # 获取选中的特征索引
            selected_indices = selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            
            # 创建特征排名DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'rank': selector.ranking_
            }).sort_values('rank')
            
            # 筛选特征
            X_selected = X[self.selected_features].copy()
            
            self.logger.info(f"递归特征消除选择完成，原始特征数: {X.shape[1]}，选择特征数: {X_selected.shape[1]}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"使用递归特征消除选择特征时发生异常: {str(e)}")
            return X.iloc[:, :min(k, X.shape[1])]
    
    def pca_dimension_reduction(self, X: pd.DataFrame, n_components: int = 128, 
                              whiten: bool = False) -> pd.DataFrame:
        """使用主成分分析(PCA)进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            whiten: 是否对主成分进行白化处理
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用PCA进行降维，目标维度: {n_components}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 初始化PCA模型
            self.reduction_model = PCA(n_components=n_components, whiten=whiten)
            
            # 拟合数据并转换
            X_reduced = self.reduction_model.fit_transform(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'pca_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            # 计算解释方差比例
            explained_variance_ratio = self.reduction_model.explained_variance_ratio_
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            
            self.logger.info(f"PCA降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            self.logger.info(f"累计解释方差比例: {cumulative_explained_variance[-1]:.4f}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用PCA进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def kernel_pca_dimension_reduction(self, X: pd.DataFrame, n_components: int = 128, 
                                     kernel: str = 'rbf', gamma: float = None) -> pd.DataFrame:
        """使用核主成分分析(Kernel PCA)进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            kernel: 核函数类型，可选'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
            gamma: rbf、poly和sigmoid核的核系数
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用Kernel PCA进行降维，目标维度: {n_components}，核函数: {kernel}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 初始化Kernel PCA模型
            self.reduction_model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
            
            # 拟合数据并转换
            X_reduced = self.reduction_model.fit_transform(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'kpca_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            self.logger.info(f"Kernel PCA降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用Kernel PCA进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def ica_dimension_reduction(self, X: pd.DataFrame, n_components: int = 128, 
                             algorithm: str = 'fastica') -> pd.DataFrame:
        """使用独立成分分析(ICA)进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            algorithm: 算法类型，可选'fastica', 'parallel'
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用ICA进行降维，目标维度: {n_components}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 初始化ICA模型
            self.reduction_model = FastICA(n_components=n_components, algorithm=algorithm)
            
            # 拟合数据并转换
            X_reduced = self.reduction_model.fit_transform(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'ica_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            self.logger.info(f"ICA降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用ICA进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def sparse_pca_dimension_reduction(self, X: pd.DataFrame, n_components: int = 128, 
                                     alpha: float = 1.0) -> pd.DataFrame:
        """使用稀疏主成分分析(Sparse PCA)进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            alpha: 控制稀疏性的参数
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用Sparse PCA进行降维，目标维度: {n_components}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 初始化Sparse PCA模型
            self.reduction_model = SparsePCA(n_components=n_components, alpha=alpha)
            
            # 拟合数据并转换
            X_reduced = self.reduction_model.fit_transform(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'sparse_pca_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            self.logger.info(f"Sparse PCA降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用Sparse PCA进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def manifold_learning(self, X: pd.DataFrame, n_components: int = 2, 
                        method: str = 'tsne', **kwargs) -> pd.DataFrame:
        """使用流形学习方法进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            method: 流形学习方法，可选'tsne', 'isomap', 'lle'
            **kwargs: 传递给流形学习算法的参数
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用流形学习进行降维，目标维度: {n_components}，方法: {method}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 选择流形学习方法
            if method == 'tsne':
                model = TSNE(n_components=n_components, **kwargs)
            elif method == 'isomap':
                model = Isomap(n_components=n_components, **kwargs)
            elif method == 'lle':
                model = LocallyLinearEmbedding(n_components=n_components, **kwargs)
            else:
                self.logger.error(f"不支持的流形学习方法: {method}")
                return X.iloc[:, :min(n_components, X.shape[1])]
            
            # 拟合数据并转换
            X_reduced = model.fit_transform(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'{method}_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            self.logger.info(f"流形学习降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用流形学习进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def autoencoder_dimension_reduction(self, X: pd.DataFrame, n_components: int = 128, 
                                      encoding_dims: List[int] = None, 
                                      epochs: int = 100, batch_size: int = 32) -> pd.DataFrame:
        """使用自编码器进行降维
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            n_components: 降维后的维度数
            encoding_dims: 编码器隐藏层维度列表
            epochs: 训练轮数
            batch_size: 批次大小
        
        Returns:
            降维后的特征矩阵
        """
        try:
            self.logger.info(f"开始使用自编码器进行降维，目标维度: {n_components}")
            
            # 导入TensorFlow
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Model
                from tensorflow.keras.layers import Input, Dense, Dropout
                from tensorflow.keras.callbacks import EarlyStopping
            except ImportError:
                self.logger.error("TensorFlow未安装，无法使用自编码器进行降维")
                return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 数据标准化
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 定义自编码器架构
            input_dim = X.shape[1]
            input_layer = Input(shape=(input_dim,))
            
            # 编码器
            if encoding_dims is None:
                # 默认架构
                encoding_dims = [max(input_dim // 2, n_components * 2), max(input_dim // 4, n_components * 1.5)]
            
            encoded = input_layer
            for dim in encoding_dims:
                encoded = Dense(dim, activation='relu')(encoded)
                encoded = Dropout(0.2)(encoded)
            
            # 编码层（瓶颈层）
            encoded = Dense(n_components, activation='relu', name='encoder_output')(encoded)
            
            # 解码器
            decoded = encoded
            for dim in encoding_dims[::-1]:
                decoded = Dense(dim, activation='relu')(decoded)
                decoded = Dropout(0.2)(decoded)
            
            # 输出层
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            
            # 构建自编码器模型
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            
            # 构建编码器模型
            encoder = Model(inputs=input_layer, outputs=encoded)
            
            # 编译模型
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # 设置早停
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # 训练模型
            history = autoencoder.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # 使用编码器进行降维
            X_reduced = encoder.predict(X_scaled)
            
            # 转换为DataFrame
            component_names = [f'ae_component_{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
            # 保存模型
            self.reduction_model = encoder
            
            self.logger.info(f"自编码器降维完成，原始维度: {X.shape[1]}，降维后维度: {X_reduced_df.shape[1]}")
            self.logger.info(f"自编码器训练损失: {history.history['loss'][-1]:.6f}")
            
            return X_reduced_df
        except Exception as e:
            self.logger.error(f"使用自编码器进行降维时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def feature_clustering_selection(self, X: pd.DataFrame, y: pd.Series, n_components: int = 128, 
                                  n_clusters: int = None) -> pd.DataFrame:
        """使用特征聚类进行特征选择
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            n_components: 目标特征数量
            n_clusters: 聚类数量，默认为目标特征数量的一半
        
        Returns:
            选择的特征矩阵
        """
        try:
            self.logger.info(f"开始使用特征聚类进行特征选择，目标特征数: {n_components}")
            
            # 设置聚类数量
            if n_clusters is None:
                n_clusters = max(2, n_components // 2)
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 计算特征相关性矩阵
            corr_matrix = X.corr().abs()
            
            # 使用KMeans对特征进行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            feature_clusters = kmeans.fit_predict(corr_matrix)
            
            # 对每个聚类中的特征计算重要性（使用随机森林）
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            
            # 为每个聚类选择最重要的特征
            self.selected_features = []
            
            for cluster in range(n_clusters):
                # 获取当前聚类的特征索引
                cluster_features = np.where(feature_clusters == cluster)[0]
                
                if len(cluster_features) == 0:
                    continue
                
                # 获取当前聚类中特征的重要性
                cluster_importances = importances[cluster_features]
                
                # 对当前聚类中的特征按重要性排序
                sorted_indices = cluster_features[np.argsort(cluster_importances)[::-1]]
                
                # 选择当前聚类中最重要的特征
                self.selected_features.append(X.columns[sorted_indices[0]])
            
            # 如果选择的特征数少于目标特征数，从剩余特征中选择最重要的
            if len(self.selected_features) < n_components:
                # 计算已选择特征的索引
                selected_indices = [X.columns.get_loc(f) for f in self.selected_features]
                
                # 获取未选择的特征
                remaining_indices = [i for i in range(X.shape[1]) if i not in selected_indices]
                
                # 对未选择的特征按重要性排序
                remaining_importances = importances[remaining_indices]
                sorted_remaining_indices = remaining_indices[np.argsort(remaining_importances)[::-1]]
                
                # 选择额外的特征
                additional_features = X.columns[sorted_remaining_indices[:n_components - len(self.selected_features)]]
                self.selected_features.extend(additional_features)
            elif len(self.selected_features) > n_components:
                # 如果选择的特征数超过目标特征数，按重要性排序后选择前n_components个
                selected_importances = importances[[X.columns.get_loc(f) for f in self.selected_features]]
                sorted_selected_indices = np.argsort(selected_importances)[::-1][:n_components]
                self.selected_features = [self.selected_features[i] for i in sorted_selected_indices]
            
            # 创建特征重要性DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': importances[[X.columns.get_loc(f) for f in self.selected_features]]
            }).sort_values('importance', ascending=False)
            
            # 筛选特征
            X_selected = X[self.selected_features].copy()
            
            self.logger.info(f"特征聚类选择完成，原始特征数: {X.shape[1]}，选择特征数: {X_selected.shape[1]}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"使用特征聚类选择特征时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])]
    
    def two_stage_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_components: int = 128, 
                                  stage1_method: str = 'random_forest', 
                                  stage2_method: str = 'pca') -> pd.DataFrame:
        """两阶段特征选择方法
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            n_components: 最终特征数量
            stage1_method: 第一阶段特征选择方法，可选'random_forest', 'lasso', 'f_classif'
            stage2_method: 第二阶段降维方法，可选'pca', 'ica', 'sparse_pca'
        
        Returns:
            选择的特征矩阵
        """
        try:
            self.logger.info(f"开始两阶段特征选择，目标特征数: {n_components}")
            self.logger.info(f"第一阶段方法: {stage1_method}，第二阶段方法: {stage2_method}")
            
            # 确保n_components不超过特征总数
            n_components = min(n_components, X.shape[1])
            
            # 第一阶段：特征选择，保留2-3倍目标特征数
            stage1_k = min(n_components * 3, X.shape[1])
            
            self.logger.info(f"第一阶段特征选择，保留特征数: {stage1_k}")
            
            if stage1_method == 'random_forest':
                X_stage1 = self.select_from_model(X, y, k=stage1_k, model_type='random_forest')
            elif stage1_method == 'lasso':
                X_stage1 = self.select_from_model(X, y, k=stage1_k, model_type='lasso')
            elif stage1_method == 'f_classif':
                X_stage1 = self.select_k_best_features(X, y, k=stage1_k, method='f_classif')
            else:
                self.logger.error(f"不支持的第一阶段方法: {stage1_method}")
                return X.iloc[:, :n_components] if X.shape[1] > n_components else X
            
            # 如果第一阶段后特征数仍超过目标数，进行第二阶段降维
            if X_stage1.shape[1] > n_components:
                self.logger.info(f"第二阶段降维，目标维度: {n_components}")
                
                if stage2_method == 'pca':
                    X_final = self.pca_dimension_reduction(X_stage1, n_components=n_components)
                elif stage2_method == 'ica':
                    X_final = self.ica_dimension_reduction(X_stage1, n_components=n_components)
                elif stage2_method == 'sparse_pca':
                    X_final = self.sparse_pca_dimension_reduction(X_stage1, n_components=n_components)
                else:
                    self.logger.error(f"不支持的第二阶段方法: {stage2_method}")
                    return X_stage1.iloc[:, :n_components] if X_stage1.shape[1] > n_components else X_stage1
            else:
                X_final = X_stage1
            
            self.logger.info(f"两阶段特征选择完成，原始特征数: {X.shape[1]}，最终特征数: {X_final.shape[1]}")
            
            return X_final
        except Exception as e:
            self.logger.error(f"使用两阶段特征选择时发生异常: {str(e)}")
            return X.iloc[:, :n_components] if X.shape[1] > n_components else X
    
    def full_feature_selection_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                     n_components: int = 128, 
                                     remove_low_var: bool = True, 
                                     var_threshold: float = 0.01, 
                                     remove_high_corr: bool = True, 
                                     corr_threshold: float = 0.9, 
                                     method: str = 'two_stage', 
                                     **kwargs) -> pd.DataFrame:
        """完整的特征选择流水线
        
        Args:
            X: 特征矩阵，形状为[样本数, 特征数]
            y: 目标变量
            n_components: 最终特征数量
            remove_low_var: 是否移除低方差特征
            var_threshold: 低方差特征阈值
            remove_high_corr: 是否移除高相关性特征
            corr_threshold: 高相关性特征阈值
            method: 特征选择方法，可选'two_stage', 'select_from_model', 'rfe', 'clustering', 'autoencoder'
            **kwargs: 传递给具体特征选择方法的参数
        
        Returns:
            选择的特征矩阵
        """
        try:
            self.logger.info(f"开始完整的特征选择流水线，目标特征数: {n_components}")
            
            # 复制数据，避免修改原始数据
            X_processed = X.copy()
            
            # 步骤1：移除低方差特征
            if remove_low_var:
                X_processed = self.remove_low_variance_features(X_processed, threshold=var_threshold)
                
                # 如果移除低方差特征后特征数少于目标数，直接返回
                if X_processed.shape[1] <= n_components:
                    self.logger.warning("移除低方差特征后特征数少于目标数，直接返回")
                    return X_processed
            
            # 步骤2：移除高相关性特征
            if remove_high_corr:
                X_processed = self.remove_high_correlation_features(X_processed, threshold=corr_threshold)
                
                # 如果移除高相关性特征后特征数少于目标数，直接返回
                if X_processed.shape[1] <= n_components:
                    self.logger.warning("移除高相关性特征后特征数少于目标数，直接返回")
                    return X_processed
            
            # 步骤3：特征选择/降维
            if method == 'two_stage':
                X_selected = self.two_stage_feature_selection(X_processed, y, n_components=n_components, **kwargs)
            elif method == 'select_from_model':
                X_selected = self.select_from_model(X_processed, y, k=n_components, **kwargs)
            elif method == 'rfe':
                X_selected = self.recursive_feature_elimination(X_processed, y, k=n_components, **kwargs)
            elif method == 'clustering':
                X_selected = self.feature_clustering_selection(X_processed, y, n_components=n_components, **kwargs)
            elif method == 'autoencoder':
                X_selected = self.autoencoder_dimension_reduction(X_processed, n_components=n_components, **kwargs)
            else:
                self.logger.error(f"不支持的特征选择方法: {method}")
                X_selected = X_processed.iloc[:, :min(n_components, X_processed.shape[1])]
            
            self.logger.info(f"完整特征选择流水线完成，原始特征数: {X.shape[1]}，最终特征数: {X_selected.shape[1]}")
            
            return X_selected
        except Exception as e:
            self.logger.error(f"执行特征选择流水线时发生异常: {str(e)}")
            return X.iloc[:, :min(n_components, X.shape[1])] if X.shape[1] > n_components else X
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取特征重要性
        
        Returns:
            包含特征名称和重要性的DataFrame
        """
        return self.feature_importance
    
    def get_selected_features(self) -> Optional[List[str]]:
        """获取选中的特征名称
        
        Returns:
            选中的特征名称列表
        """
        return self.selected_features
    
    def save_feature_importance_plot(self, output_path: str, top_n: int = 20) -> None:
        """保存特征重要性可视化图表
        
        Args:
            output_path: 图表保存路径
            top_n: 显示前n个特征
        """
        try:
            if self.feature_importance is None:
                self.logger.error("没有特征重要性数据，无法生成图表")
                return
            
            self.logger.info(f"开始生成特征重要性图表，显示前{top_n}个特征")
            
            # 选择前n个特征
            top_features = self.feature_importance.head(top_n)
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 根据特征重要性DataFrame的列名确定绘图方式
            if 'importance' in top_features.columns:
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.xlabel('重要性')
            elif 'score' in top_features.columns:
                sns.barplot(x='score', y='feature', data=top_features)
                plt.xlabel('评分')
            elif 'rank' in top_features.columns:
                sns.barplot(x='rank', y='feature', data=top_features)
                plt.xlabel('排名')
            else:
                self.logger.error("特征重要性DataFrame缺少必要的列")
                return
            
            plt.ylabel('特征')
            plt.title(f'前{top_n}个重要特征')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            self.logger.info(f"特征重要性图表已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"生成特征重要性图表时发生异常: {str(e)}")
    
    def save_reduction_results(self, X_reduced: pd.DataFrame, output_path: str) -> None:
        """保存降维结果
        
        Args:
            X_reduced: 降维后的特征矩阵
            output_path: 输出文件路径
        """
        try:
            self.logger.info(f"开始保存降维结果到: {output_path}")
            
            # 保存为CSV文件
            X_reduced.to_csv(output_path, index=True)
            
            self.logger.info(f"降维结果保存完成")
        except Exception as e:
            self.logger.error(f"保存降维结果时发生异常: {str(e)}")