import os
import logging
import numpy as np
import pandas as pd
import joblib
import itertools
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HyperparameterTuner:
    """超参数调优器
    封装多种超参数优化方法，包括网格搜索、随机搜索和贝叶斯优化等
    """
    def __init__(self, 
                 model: Union[BaseEstimator, Model],
                 param_space: Dict[str, Any],
                 config: Optional[Dict[str, Any]] = None,
                 n_trials: int = 50,
                 scoring: Union[str, List[str], Callable] = 'neg_mean_squared_error',
                 cv_strategy: str = 'kfold',
                 n_splits: int = 5,
                 random_state: int = 42,
                 verbose: int = 1,
                 tuner_type: str = 'optuna_tpe',
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None):
        """初始化超参数调优器
        
        Args:
            model: 要调优的模型实例
            param_space: 超参数空间
            config: 配置字典
            n_trials: 试验次数（对于随机搜索和贝叶斯优化）
            scoring: 评分指标
            cv_strategy: 交叉验证策略，可选'kfold', 'timeseries', 'holdout'
            n_splits: 交叉验证折数
            random_state: 随机种子
            verbose: 详细程度
            tuner_type: 调优器类型，可选'grid', 'random', 'optuna_tpe', 'optuna_random', 'optuna_cmaes'
            study_name: Optuna研究名称
            storage: Optuna存储URL
        """
        self.model = model
        self.param_space = param_space
        self.config = config or {}
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.tuner_type = tuner_type
        self.study_name = study_name or f"hyperopt_{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        
        # 初始化结果相关属性
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = -np.inf
        self.best_estimator_: Optional[Union[BaseEstimator, Model]] = None
        self.cv_results_: Dict[str, Any] = {}
        self.trials_: List[Any] = []
        self.search_time_: float = 0.0
        
        # 创建必要的目录
        self._create_directories()
        
        # 初始化日志
        self._init_logger()
        
    def _init_logger(self):
        """初始化日志记录器"""
        log_dir = self.config.get('log_dir', './logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
        
        # 超参数调优结果目录
        hyperopt_dir = os.path.join(results_dir, 'hyperparameter_tuning')
        if not os.path.exists(hyperopt_dir):
            os.makedirs(hyperopt_dir)
        
        # 模型保存目录
        models_dir = self.config.get('models_dir', './models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def _get_cv_splitter(self, X: Any) -> Any:
        """获取交叉验证分割器
        
        Args:
            X: 特征数据
        
        Returns:
            交叉验证分割器实例
        """
        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            # 默认使用KFold
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    
    def _prepare_param_space(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """准备超参数空间，将Optuna特定的参数转换为标准参数空间
        
        Args:
            param_space: 原始超参数空间
        
        Returns:
            处理后的超参数空间
        """
        prepared_space = {}
        for param_name, param_config in param_space.items():
            # 处理字典形式的参数配置
            if isinstance(param_config, dict):
                # 例如 {'type': 'int', 'low': 10, 'high': 100, 'step': 10}
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    prepared_space[param_name] = list(range(
                        param_config['low'], 
                        param_config['high'] + 1, 
                        param_config.get('step', 1)
                    ))
                elif param_type == 'float':
                    # 对于浮点数，生成样本点
                    num_samples = param_config.get('samples', 10)
                    prepared_space[param_name] = np.linspace(
                        param_config['low'], 
                        param_config['high'], 
                        num_samples
                    )
                elif param_type == 'categorical':
                    prepared_space[param_name] = param_config.get('choices', [])
                else:
                    # 未知类型，保留原始值
                    prepared_space[param_name] = param_config
            else:
                # 非字典形式，直接使用
                prepared_space[param_name] = param_config
        
        return prepared_space
    
    def tune(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """执行超参数调优
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
        
        Returns:
            调优结果字典
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始超参数调优，使用方法: {self.tuner_type}")
            
            # 根据调优器类型选择不同的调优方法
            if self.tuner_type == 'grid':
                self._grid_search(X, y, **kwargs)
            elif self.tuner_type == 'random':
                self._random_search(X, y, **kwargs)
            elif self.tuner_type.startswith('optuna'):
                self._optuna_search(X, y, **kwargs)
            else:
                raise ValueError(f"不支持的调优器类型: {self.tuner_type}")
            
            # 计算调优时间
            self.search_time_ = time.time() - start_time
            logger.info(f"超参数调优完成，耗时: {self.search_time_:.2f}秒")
            
            # 保存调优结果
            results = self._save_results()
            
            return results
            
        except Exception as e:
            logger.error(f"超参数调优过程中发生异常: {str(e)}")
            raise
    
    def _grid_search(self, X: Any, y: Any, **kwargs) -> None:
        """执行网格搜索
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
        """
        logger.info("执行网格搜索")
        
        # 准备参数空间
        prepared_param_space = self._prepare_param_space(self.param_space)
        
        # 检查参数空间大小，避免计算量过大
        total_combinations = 1
        for param_values in prepared_param_space.values():
            if isinstance(param_values, (list, tuple, np.ndarray)):
                total_combinations *= len(param_values)
        
        if total_combinations > 1000 and self.verbose > 0:
            logger.warning(f"网格搜索参数组合数量过大: {total_combinations}，可能会耗费大量时间")
        
        # 创建GridSearchCV实例
        cv_splitter = self._get_cv_splitter(X)
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=prepared_param_space,
            scoring=self.scoring,
            cv=cv_splitter,
            n_jobs=kwargs.get('n_jobs', -1),
            verbose=self.verbose,
            refit=kwargs.get('refit', True),
            return_train_score=kwargs.get('return_train_score', True)
        )
        
        # 执行搜索
        grid_search.fit(X, y)
        
        # 保存结果
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_
    
    def _random_search(self, X: Any, y: Any, **kwargs) -> None:
        """执行随机搜索
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
        """
        logger.info("执行随机搜索")
        
        # 准备参数空间
        prepared_param_space = self.param_space.copy()
        
        # 创建RandomizedSearchCV实例
        cv_splitter = self._get_cv_splitter(X)
        
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=prepared_param_space,
            n_iter=self.n_trials,
            scoring=self.scoring,
            cv=cv_splitter,
            n_jobs=kwargs.get('n_jobs', -1),
            verbose=self.verbose,
            random_state=self.random_state,
            refit=kwargs.get('refit', True),
            return_train_score=kwargs.get('return_train_score', True)
        )
        
        # 执行搜索
        random_search.fit(X, y)
        
        # 保存结果
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.best_estimator_ = random_search.best_estimator_
        self.cv_results_ = random_search.cv_results_
    
    def _optuna_objective(self, trial: optuna.Trial, X: Any, y: Any, **kwargs) -> float:
        """Optuna优化目标函数
        
        Args:
            trial: Optuna试验对象
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
        
        Returns:
            目标分数
        """
        # 根据参数空间生成当前试验的参数
        params = {}
        for param_name, param_config in self.param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config.get('choices', [])
                    )
                elif param_type == 'uniform':
                    params[param_name] = trial.suggest_uniform(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                elif param_type == 'loguniform':
                    params[param_name] = trial.suggest_loguniform(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                else:
                    # 默认使用categorical
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        [param_config]
                    )
            else:
                # 非字典形式，直接使用
                params[param_name] = param_config
        
        # 创建模型实例
        if hasattr(self.model, 'set_params'):
            model = self.model.set_params(**params)
        else:
            # 对于Keras等模型，需要重新初始化
            model = self.model.__class__(**params)
        
        # 执行交叉验证
        cv_splitter = self._get_cv_splitter(X)
        scores = []
        
        for train_idx, val_idx in cv_splitter.split(X):
            # 分割数据
            if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
                X_train, X_val = X[train_idx], X[val_idx]
            else:
                # 处理其他类型的数据结构
                X_train = [X[i] for i in train_idx]
                X_val = [X[i] for i in val_idx]
            
            if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                y_train = [y[i] for i in train_idx]
                y_val = [y[i] for i in val_idx]
            
            # 训练模型
            try:
                model.fit(X_train, y_train, **kwargs)
                
                # 评估模型
                if hasattr(model, 'score'):
                    score = model.score(X_val, y_val)
                elif hasattr(model, 'evaluate'):
                    # Keras模型
                    score = model.evaluate(X_val, y_val, verbose=0)
                    # 处理多输出情况
                    if isinstance(score, list) or isinstance(score, tuple):
                        score = score[0]  # 取第一个指标
                else:
                    # 预测并计算评分
                    y_pred = model.predict(X_val)
                    score = self._calculate_score(y_val, y_pred)
                
                scores.append(score)
                
                # 早停检查
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            except Exception as e:
                logger.warning(f"参数组合 {params} 训练失败: {str(e)}")
                # 返回极低分数表示失败
                return -np.inf
        
        # 计算平均分数
        mean_score = np.mean(scores)
        
        # 记录参数和分数
        trial.set_user_attr('params', params)
        trial.set_user_attr('score', mean_score)
        
        return mean_score
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算评分指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            评分值
        """
        from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
        
        # 处理预测结果维度
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        
        # 选择评分函数
        if isinstance(self.scoring, str):
            if self.scoring == 'neg_mean_squared_error' or self.scoring == 'mse':
                score = -mean_squared_error(y_true, y_pred)
            elif self.scoring == 'neg_mean_absolute_error' or self.scoring == 'mae':
                score = -mean_absolute_error(y_true, y_pred)
            elif self.scoring == 'r2' or self.scoring == 'r2_score':
                score = r2_score(y_true, y_pred)
            else:
                # 默认使用MSE
                score = -mean_squared_error(y_true, y_pred)
        elif callable(self.scoring):
            score = self.scoring(y_true, y_pred)
        else:
            # 默认使用MSE
            score = -mean_squared_error(y_true, y_pred)
        
        return score
    
    def _optuna_search(self, X: Any, y: Any, **kwargs) -> None:
        """执行Optuna贝叶斯优化
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 额外参数
        """
        logger.info(f"执行Optuna优化，使用采样器: {self.tuner_type}")
        
        # 选择采样器
        if self.tuner_type == 'optuna_tpe':
            sampler = TPESampler(seed=self.random_state)
        elif self.tuner_type == 'optuna_random':
            sampler = RandomSampler(seed=self.random_state)
        elif self.tuner_type == 'optuna_cmaes':
            sampler = CmaEsSampler(seed=self.random_state)
        else:
            sampler = TPESampler(seed=self.random_state)
        
        # 选择剪枝器
        pruner = MedianPruner()
        
        # 创建Optuna研究
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            load_if_exists=False
        )
        
        # 执行优化
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y, **kwargs),
            n_trials=self.n_trials,
            n_jobs=kwargs.get('n_jobs', 1),
            show_progress_bar=self.verbose > 0
        )
        
        # 保存结果
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.trials_ = study.trials
        
        # 使用最佳参数重新训练模型
        if hasattr(self.model, 'set_params'):
            self.best_estimator_ = self.model.set_params(**self.best_params_)
        else:
            self.best_estimator_ = self.model.__class__(**self.best_params_)
        
        self.best_estimator_.fit(X, y, **kwargs)
        
        # 构建cv_results_
        self.cv_results_ = {
            'params': [trial.params for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
            'mean_test_score': [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
            'rank_test_score': list(range(1, len(study.trials) + 1)),
            'std_test_score': [0.0 for _ in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]  # 简化处理
        }
    
    def _save_results(self) -> Dict[str, Any]:
        """保存调优结果
        
        Returns:
            结果字典
        """
        # 准备结果数据
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'search_time': self.search_time_,
            'tuner_type': self.tuner_type,
            'n_trials': self.n_trials,
            'cv_strategy': self.cv_strategy,
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # 保存结果到文件
        results_dir = os.path.join(self.config.get('results_dir', './results'), 'hyperparameter_tuning')
        results_path = os.path.join(results_dir, f"hyperopt_results_{results['timestamp']}.json")
        
        # 转换numpy类型为Python原生类型
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存最佳模型
        if self.best_estimator_ is not None:
            models_dir = self.config.get('models_dir', './models')
            model_path = os.path.join(models_dir, f"best_model_{results['timestamp']}.joblib")
            
            try:
                joblib.dump(self.best_estimator_, model_path)
                logger.info(f"最佳模型已保存到: {model_path}")
            except Exception as e:
                logger.error(f"保存最佳模型时发生异常: {str(e)}")
        
        logger.info(f"超参数调优结果已保存到: {results_path}")
        
        # 保存详细的交叉验证结果
        if self.cv_results_:
            cv_results_path = os.path.join(results_dir, f"cv_results_{results['timestamp']}.csv")
            
            try:
                if isinstance(self.cv_results_, dict):
                    # 处理GridSearchCV和RandomizedSearchCV的结果
                    if 'params' in self.cv_results_:
                        params_df = pd.DataFrame(self.cv_results_['params'])
                        # 将参数与其他结果合并
                        metrics_df = pd.DataFrame({
                            key: self.cv_results_[key]
                            for key in self.cv_results_
                            if key != 'params'
                        })
                        cv_results_df = pd.concat([params_df, metrics_df], axis=1)
                    else:
                        cv_results_df = pd.DataFrame(self.cv_results_)
                    
                    cv_results_df.to_csv(cv_results_path, index=False)
                
                logger.info(f"交叉验证详细结果已保存到: {cv_results_path}")
            except Exception as e:
                logger.error(f"保存交叉验证结果时发生异常: {str(e)}")
        
        return results
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳超参数
        
        Returns:
            最佳超参数字典
        """
        return self.best_params_
    
    def get_best_estimator(self) -> Optional[Union[BaseEstimator, Model]]:
        """获取最佳模型
        
        Returns:
            最佳模型实例
        """
        return self.best_estimator_
    
    def get_best_score(self) -> float:
        """获取最佳评分
        
        Returns:
            最佳评分值
        """
        return self.best_score_
    
    def plot_optimization_history(self, fig_path: Optional[str] = None) -> Optional[str]:
        """绘制优化历史图表
        
        Args:
            fig_path: 图表保存路径
        
        Returns:
            图表保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            if self.tuner_type.startswith('optuna') and self.trials_:
                # 绘制Optuna优化历史
                values = []
                for trial in self.trials_:
                    if trial.value is not None:
                        values.append(trial.value)
                    else:
                        values.append(np.nan)
                
                plt.plot(values, 'o-', label='Score')
                
                # 绘制最佳分数曲线
                best_values = []
                current_best = -np.inf
                for val in values:
                    if not np.isnan(val) and val > current_best:
                        current_best = val
                    best_values.append(current_best)
                
                plt.plot(best_values, 'r-', label='Best Score')
                
            elif self.cv_results_ and 'mean_test_score' in self.cv_results_:
                # 绘制GridSearchCV或RandomizedSearchCV的结果
                scores = self.cv_results_['mean_test_score']
                plt.plot(scores, 'o-', label='Mean Test Score')
                
            plt.title('Hyperparameter Optimization History')
            plt.xlabel('Trial Index')
            plt.ylabel('Score')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            if fig_path is None:
                figures_dir = os.path.join(self.config.get('figures_dir', './figures'), 'hyperparameter_tuning')
                if not os.path.exists(figures_dir):
                    os.makedirs(figures_dir)
                
                fig_path = os.path.join(figures_dir, f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"优化历史图表已保存到: {fig_path}")
            
            return fig_path
        except Exception as e:
            logger.error(f"绘制优化历史图表时发生异常: {str(e)}")
            return None
    
    def plot_param_importance(self, fig_path: Optional[str] = None, n_params: int = 10) -> Optional[str]:
        """绘制参数重要性图表
        
        Args:
            fig_path: 图表保存路径
            n_params: 显示的参数数量
        
        Returns:
            图表保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.cv_results_ or 'params' not in self.cv_results_:
                logger.warning("没有足够的交叉验证结果来计算参数重要性")
                return None
            
            # 提取参数和分数
            params = self.cv_results_['params']
            scores = self.cv_results_['mean_test_score']
            
            # 创建参数-分数数据框
            param_df = pd.DataFrame(params)
            param_df['score'] = scores
            
            # 计算参数重要性（使用相关性）
            correlations = {}
            
            for param_name in param_df.columns:
                if param_name == 'score':
                    continue
                
                # 尝试计算相关性
                try:
                    if param_df[param_name].dtype == 'object':
                        # 对于分类变量，使用ANOVA
                        from sklearn.feature_selection import f_classif
                        # 编码分类变量
                        encoded_param = pd.get_dummies(param_df[param_name])
                        f_stat, p_value = f_classif(encoded_param, param_df['score'])
                        correlations[param_name] = np.max(f_stat)
                    else:
                        # 对于数值变量，使用Pearson相关性
                        corr = param_df[param_name].corr(param_df['score'])
                        correlations[param_name] = abs(corr)
                except Exception as e:
                    logger.warning(f"计算参数 {param_name} 重要性时发生异常: {str(e)}")
                    correlations[param_name] = 0.0
            
            # 按重要性排序
            sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # 取前n_params个参数
            sorted_correlations = sorted_correlations[:n_params]
            
            if not sorted_correlations:
                logger.warning("无法计算有效的参数重要性")
                return None
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            param_names = [item[0] for item in sorted_correlations]
            importances = [item[1] for item in sorted_correlations]
            
            sns.barplot(x=importances, y=param_names, palette='viridis')
            
            plt.title(f'Top {n_params} Parameter Importances')
            plt.xlabel('Importance (Correlation with Score)')
            plt.ylabel('Parameter')
            plt.grid(True, axis='x')
            
            # 保存图表
            if fig_path is None:
                figures_dir = os.path.join(self.config.get('figures_dir', './figures'), 'hyperparameter_tuning')
                if not os.path.exists(figures_dir):
                    os.makedirs(figures_dir)
                
                fig_path = os.path.join(figures_dir, f"param_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"参数重要性图表已保存到: {fig_path}")
            
            return fig_path
        except Exception as e:
            logger.error(f"绘制参数重要性图表时发生异常: {str(e)}")
            return None
    
    def generate_report(self, report_path: Optional[str] = None) -> Optional[str]:
        """生成超参数调优报告
        
        Args:
            report_path: 报告保存路径
        
        Returns:
            报告保存路径
        """
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 设置保存路径
            if report_path is None:
                reports_dir = os.path.join(self.config.get('results_dir', './results'), 'reports')
                if not os.path.exists(reports_dir):
                    os.makedirs(reports_dir)
                
                report_path = os.path.join(reports_dir, f"hyperparameter_tuning_report_{timestamp}.html")
            
            # 生成图表
            history_plot = self.plot_optimization_history()
            importance_plot = self.plot_param_importance()
            
            # 创建报告内容
            report_content = []
            
            # 添加报告头部
            report_content.append('<!DOCTYPE html>')
            report_content.append('<html>')
            report_content.append('<head>')
            report_content.append('<meta charset="UTF-8">')
            report_content.append('<title>超参数调优报告</title>')
            report_content.append('<style>')
            report_content.append('body { font-family: Arial, sans-serif; margin: 20px; }')
            report_content.append('h1, h2 { color: #333; }')
            report_content.append('.container { max-width: 1200px; margin: 0 auto; }')
            report_content.append('.summary { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }')
            report_content.append('table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
            report_content.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
            report_content.append('th { background-color: #f2f2f2; }')
            report_content.append('tr:nth-child(even) { background-color: #f9f9f9; }')
            report_content.append('img { max-width: 100%; height: auto; margin: 20px 0; }')
            report_content.append('.param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; margin: 20px 0; }')
            report_content.append('.param-card { background: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }')
            report_content.append('.param-name { font-weight: bold; color: #2c3e50; }')
            report_content.append('</style>')
            report_content.append('</head>')
            report_content.append('<body>')
            report_content.append('<div class="container">')
            report_content.append(f'<h1>超参数调优报告 - {timestamp}</h1>')
            
            # 添加摘要信息
            report_content.append('<div class="summary">')
            report_content.append('<h2>调优摘要</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>项目</th><th>值</th></tr>')
            report_content.append(f'<tr><td>模型类型</td><td>{type(self.model).__name__}</td></tr>')
            report_content.append(f'<tr><td>调优方法</td><td>{self.tuner_type}</td></tr>')
            report_content.append(f'<tr><td>试验次数</td><td>{self.n_trials}</td></tr>')
            report_content.append(f'<tr><td>交叉验证策略</td><td>{self.cv_strategy} ({self.n_splits}折)</td></tr>')
            report_content.append(f'<tr><td>最佳分数</td><td>{self.best_score_:.4f}</td></tr>')
            report_content.append(f'<tr><td>调优耗时</td><td>{self.search_time_:.2f}秒</td></tr>')
            report_content.append('</table>')
            report_content.append('</div>')
            
            # 添加最佳参数
            report_content.append('<h2>最佳超参数</h2>')
            report_content.append('<div class="param-grid">')
            for param_name, param_value in self.best_params_.items():
                report_content.append('<div class="param-card">')
                report_content.append(f'<div class="param-name">{param_name}</div>')
                report_content.append(f'<div>{param_value}</div>')
                report_content.append('</div>')
            report_content.append('</div>')
            
            # 添加图表
            report_content.append('<h2>可视化结果</h2>')
            
            # 添加优化历史图表
            if history_plot and os.path.exists(history_plot):
                rel_path = os.path.relpath(history_plot, os.path.dirname(report_path))
                report_content.append(f'<h3>优化历史</h3>')
                report_content.append(f'<img src="{rel_path}" alt="优化历史图">')
            
            # 添加参数重要性图表
            if importance_plot and os.path.exists(importance_plot):
                rel_path = os.path.relpath(importance_plot, os.path.dirname(report_path))
                report_content.append(f'<h3>参数重要性</h3>')
                report_content.append(f'<img src="{rel_path}" alt="参数重要性图">')
            
            # 添加调优配置
            report_content.append('<h2>调优配置</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>配置项</th><th>值</th></tr>')
            report_content.append(f'<tr><td>随机种子</td><td>{self.random_state}</td></tr>')
            report_content.append(f'<tr><td>评分指标</td><td>{self.scoring}</td></tr>')
            report_content.append(f'<tr><td>详细程度</td><td>{self.verbose}</td></tr>')
            report_content.append('</table>')
            
            # 添加超参数空间
            report_content.append('<h2>超参数空间</h2>')
            report_content.append('<table>')
            report_content.append('<tr><th>参数名</th><th>搜索空间</th></tr>')
            for param_name, param_config in self.param_space.items():
                # 格式化参数配置
                if isinstance(param_config, dict):
                    config_str = ', '.join([f'{k}: {v}' for k, v in param_config.items()])
                else:
                    config_str = str(param_config)
                
                report_content.append(f'<tr><td>{param_name}</td><td>{config_str}</td></tr>')
            report_content.append('</table>')
            
            # 添加报告尾部
            report_content.append('</div>')
            report_content.append('</body>')
            report_content.append('</html>')
            
            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            logger.info(f"超参数调优报告已生成: {report_path}")
            
            return report_path
        except Exception as e:
            logger.error(f"生成超参数调优报告时发生异常: {str(e)}")
            return None

# 模块版本
__version__ = '0.1.0'

# 导出模块内容
__all__ = ['HyperparameterTuner']