"""
Enhanced HybridTrainer for backtesting and model training
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Union, cast, Type
import os
import json
import traceback
import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, date
import pytz
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import numbers
import logging
from logger_setup import RateLimitedLogger

from curriculum_learning import CurriculumWrapper
from multi_head_policy import MultiHeadActorCriticPolicy
from indicator_manager import TechnicalIndicatorManager
from risk_manager import RiskManager
from risk_config import RiskConfig
from trading_environment import TradingEnv
from enhanced_monitor import EnhancedProgressMonitor
from logger_setup import setup_logger, RateLimitedLogger
from trading_metrics import TradingMetrics, BaseTradingMetrics, EnhancedTradingMetrics
from preprocess_data import split_dataset, clean_and_validate_dataframe, preprocess_data
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from callbacks import LrSchedulerCallback, CustomEvalCallback 
from logging import Logger

# Import configuration values 
from config_vals import (
    MAX_POSITION_SIZE, STOP_LOSS, TRAILING_STOP_PERCENT,
    MAX_DRAWDOWN, RISK_PER_TRADE, WINDOW_SIZE, MAX_TIMESTEPS,
    TIME_FILTERS, BATCH_SIZE, LEARNING_RATE, GAMMA, TAU,
    ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM, N_STEPS, N_EPOCHS,
    INPUT_DIM, SEQUENCE_LENGTH, REWARD_SCALE, TRANSACTION_COST,
    START_CAP
)

logger = setup_logger()

class HybridTrainer:
    """Enhanced hybrid training system for backtesting and eventual production use"""

    def __init__(self, config: Dict[str, Any], logger: Union[Logger, RateLimitedLogger]):
        """Initialize the hybrid trainer with configuration"""
        self.config = config
        self.logger = logger
        self.results: Dict[str, Any] = {}
        self.model: Optional[PPO] = None
        self.train_env: VecEnv
        self.val_env: VecEnv
        self.test_env: VecEnv
        self.target_symbol: Optional[str] = self.config.get('target_symbol')
        self.eval_noise = self.config.get('eval_noise', 0.1)
        self.total_timesteps = self.config.get('total_timesteps', 200000)
        self.n_eval_episodes = self.config.get('n_eval_episodes', 1)
        self.log_dir = self.config.get('log_dir', 'logs/')
        self.model_save_path = self.config.get('model_save_path', 'rl_models/ppo_hybrid_model')
        self.window_size = self.config.get('window_size', WINDOW_SIZE)
        self.use_progress_bar = self.config.get('progress_bar', True)
        self.policy_kwargs = {}
        self.curriculum_wrapper = None # To hold the curriculum wrapper instance
        self.seed = self.config.get('seed')

        self.setup_directories()
        self._setup_environments()
        self._create_or_load_model()

    def setup_directories(self) -> None:
        """Sets up the directories for the training process."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'tensorboard'), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        """Loads data for the target symbol."""
        symbol_data_path = os.path.join(self.config.get('raw_data_dir_path', 'raw_data_a/'), f"{self.target_symbol}_1min.csv")
        if not os.path.exists(symbol_data_path):
            raise FileNotFoundError(f"Data file not found for symbol {self.target_symbol}")
        df = pd.read_csv(symbol_data_path, index_col='timestamp', parse_dates=True)
        df['symbol'] = self.target_symbol
        return df

    def _setup_environments(self) -> None:
        """Loads data, preprocesses it, and creates train/val/test environments."""
        df = self._load_data()
        validated_df = clean_and_validate_dataframe(df.reset_index())
        if validated_df is None:
            raise ValueError("Dataframe validation failed")
        
        indicator_manager = TechnicalIndicatorManager()
        initial_indicators = self.config.get('initial_indicators', ['rsi', 'macd', 'bollinger', 'atr', 'vwap'])
        for indicator_name in initial_indicators:
            indicator_manager.add_indicator(indicator_name)
        env_df = indicator_manager.calculate_indicators(validated_df)
        
        train_df, val_df, test_df = split_dataset(env_df)
        
        columns_to_drop = ['timestamp', 'symbol']
        train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
        val_df = val_df.drop(columns=columns_to_drop, errors='ignore')
        test_df = test_df.drop(columns=columns_to_drop, errors='ignore')

        # Create the environment instances. _create_env will raise an error if df is empty.
        train_env_instance = self._create_env(train_df, "train", self.config.get('train_max_steps', MAX_TIMESTEPS))
        val_env_instance = self._create_env(val_df, "validation", self.config.get('val_max_steps', 50000))
        test_env_instance = self._create_env(test_df, "test", self.config.get('test_max_steps', MAX_TIMESTEPS // 2))

        # Wrap environments in a DummyVecEnv for compatibility with callbacks
        self.train_env = DummyVecEnv([lambda: train_env_instance])
        self.val_env = DummyVecEnv([lambda: val_env_instance])
        self.test_env = DummyVecEnv([lambda: test_env_instance])

    def _create_env(self, df: pd.DataFrame, purpose: str, max_steps: int) -> gym.Env:
        """Create and configure a trading environment."""
        if df.empty:
            self.logger.error(f"DataFrame for {purpose} is empty. Cannot create environment.")
            raise ValueError(f"Empty DataFrame for {purpose} environment.")

        # Create a new RiskManager instance for each environment to prevent state leakage
        risk_config = RiskConfig(
            base_stop_loss=self.config.get('rm_stop_loss_pct', STOP_LOSS),
            base_take_profit=self.config.get('rm_take_profit_pct', 0.04),
            max_position_size=self.config.get('max_position_size', MAX_POSITION_SIZE),
            risk_per_trade=self.config.get('risk_per_trade', RISK_PER_TRADE),
            transaction_cost=self.config.get('transaction_cost', TRANSACTION_COST),
            max_portfolio_risk=self.config.get('max_drawdown', MAX_DRAWDOWN)
        )
        risk_manager = RiskManager(
            config=risk_config, 
            env_interface=None, # Will be set by the env itself
            initial_capital=self.config.get('initial_capital', START_CAP)
        )

        env_config = {
            'df': df,
            'risk_manager': risk_manager,
            'window_size': self.window_size,
            'max_steps': max_steps,
            'stop_loss_penalty': self.config.get('stop_loss_penalty', 1.0),
            'holding_penalty': self.config.get('holding_penalty', 0.001),
            'profit_incentive': self.config.get('profit_incentive', 0.1),
            'novelty_reward_scale': self.config.get('novelty_reward_scale', 0.0001),
            'seed': self.seed
        }
        env = TradingEnv(**env_config)

        if self.config.get('use_curriculum', False) and purpose == 'train':
            self.logger.info("Applying curriculum learning to the training environment.")
            curriculum_config = {
                'total_timesteps': self.total_timesteps,
                'stall_threshold': self.config.get('curriculum_stall_threshold', 50000),
                'performance_window_size': self.config.get('curriculum_perf_window', 1000),
                'success_threshold': self.config.get('curriculum_success_thresh', 0.6),
                'eval_window': self.config.get('curriculum_eval_window', 100)
            }
            self.curriculum_wrapper = CurriculumWrapper(env=env, config=curriculum_config)
            env = self.curriculum_wrapper

        env = Monitor(env, filename=os.path.join(self.log_dir, f"{purpose}_monitor.csv"))
        return env

    def _create_or_load_model(self):
        """Creates or loads the PPO model."""
        self.policy_kwargs = {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        }
        if self.train_env is None:
            raise ValueError("Training environment is not initialized")
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            policy_kwargs=self.policy_kwargs,
            verbose=self.config.get('verbose', 0),
            tensorboard_log=os.path.join(self.log_dir, 'tensorboard'),
            n_steps=self.config.get('n_steps', N_STEPS),
            batch_size=self.config.get('batch_size', BATCH_SIZE),
            n_epochs=self.config.get('n_epochs', N_EPOCHS),
            gamma=self.config.get('gamma', GAMMA),
            ent_coef=self.config.get('ent_coef', 0.03),
            vf_coef=self.config.get('vf_coef', VF_COEF),
            max_grad_norm=self.config.get('max_grad_norm', MAX_GRAD_NORM),
            learning_rate=self.config.get('learning_rate', LEARNING_RATE),
            device=self.config.get('device', 'auto'),
            seed=self.seed
        )

    def train_model(self) -> None:
        """Train the PPO model."""
        self.logger.info("Starting model training...")
        if self.model is None:
            raise ValueError("Model or validation environment is not initialized")

        # --- Setup the evaluation callback ---
        # This will run periodically, evaluate the model on the validation set,
        # log the performance to TensorBoard, and save the best model.
        eval_callback = CustomEvalCallback(
            eval_env=self.val_env,
            eval_freq=self.config.get('eval_freq', 5000),
            n_eval_episodes=self.n_eval_episodes,
            log_path=self.log_dir,
            deterministic=True,
            early_stopping_patience=self.config.get('early_stopping_patience', 5),
            early_stopping_metric=self.config.get('early_stopping_metric', 'sharpe_ratio')
        )

        callbacks_to_use: List[BaseCallback] = [eval_callback]
        
        if self.config.get('use_lr_scheduler', False):
            self.logger.info("Applying learning rate scheduler.")
            lr_scheduler_callback = LrSchedulerCallback(
                initial_lr=self.config.get('learning_rate', 1e-3),
                final_lr=self.config.get('lr_scheduler_final_lr', 1e-5),
                total_timesteps=self.total_timesteps
            )
            callbacks_to_use.append(lr_scheduler_callback)
            eval_callback.set_lr_scheduler(lr_scheduler_callback)

        # If curriculum is used, link it to the eval callback
        if self.curriculum_wrapper:
            eval_callback.set_curriculum_wrapper(self.curriculum_wrapper)

        # --- END DIAGNOSTIC PRINTS ---

        # Train the model
        if self.model:
            # The 'callback' argument is passed positionally to support different library versions
            self.model.learn(
                total_timesteps=self.config.get("train_max_steps", MAX_TIMESTEPS),
                callback=CallbackList(callbacks_to_use),
                progress_bar=True
            )
            self.logger.info("Model training finished.")
            
            # After training, retrieve the evaluation history directly from the callback's public attribute.
            self.training_evaluation_history = eval_callback.training_evaluation_history
        else:
            raise ValueError("Model is not initialized, cannot run training.")
        
        self.results['training_evaluation_history'] = self.training_evaluation_history
        
        self.logger.info("Model training completed.")

    def evaluate_model(self, use_test_env: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model on either the validation or test environment using evaluate_policy.
        """
        eval_type = "test" if use_test_env else "validation"
        self.logger.info(f"Starting final evaluation on {eval_type} environment...")
        
        eval_env = self.test_env if use_test_env else self.val_env
        if not eval_env or not self.model:
            self.logger.error(f"Evaluation environment or model for {eval_type} is not set.")
            return {}

        # Use the standard evaluation function
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=False # We get detailed metrics from the env itself
        )
        self.logger.info(f"Evaluation on {eval_type} completed. Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Retrieve metrics and trade log from the environment
        summary_metrics = eval_env.env_method('get_metrics')[0]
        trade_log = eval_env.env_method('get_trade_log')[0]

        eval_results = {
            f"{eval_type}_mean_reward": mean_reward,
            f"{eval_type}_std_reward": std_reward,
            f"{eval_type}_summary_metrics": summary_metrics,
            f"{eval_type}_trade_log": trade_log,
        }
        
        self.results.update(eval_results)
        return eval_results

    def save_model_and_results(self, prefix: str = "final") -> None:
        """Save the final model and the collected results."""
        timestamp = datetime.now(pytz.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"iteration_{self.target_symbol}_{timestamp}"
        
        if self.model:
            final_model_path = os.path.join(self.log_dir, "models", f"{prefix}_{run_name}.zip")
            # Purpose: Save the very last state of the model after all timesteps.
            self.model.save(final_model_path)
            self.logger.info(f"Final model saved to: {final_model_path}")

            # The best model path is determined by the callback.
            best_model_path = os.path.join(self.log_dir, 'best_model.zip')
            self.logger.info(f"Best model during training was saved to: {best_model_path}")
        
        output_results = {
            "run_name": run_name,
            "run_timestamp": timestamp,
            "config": self.config,
            "test_results": self.results.get("test_results", {}),
            "validation_results": self.results.get("validation_results", {}),
            # This is now populated by the CustomEvalCallback
            "training_evaluation_history": self.results.get("training_evaluation_history", []),
        }
        
        # Save individual run results first
        results_path = os.path.join(self.log_dir, f"{run_name}.json")
        with open(results_path, 'w') as f:
            json.dump(self._convert_to_serializable(output_results), f, indent=4)
        self.logger.info(f"Run results and configuration saved to: {results_path}")

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.datetime64):
            return np.datetime_as_string(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            if np.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
            if np.isnan(obj):
                return 'NaN'
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Execute the full training and evaluation pipeline."""
        self.logger.info("Starting training pipeline...")
        try:
            self.train_model()
            if self.config.get('evaluate_on_validation_set', True):
                self.logger.info("Starting validation evaluation...")
                self.results['validation_results'] = self.evaluate_model(use_test_env=False)
            if self.config.get('evaluate_on_test_set', True):
                self.logger.info("Starting test evaluation...")
                self.results['test_results'] = self.evaluate_model(use_test_env=True)
            self.save_model_and_results()
        except Exception as e:
            self.logger.error(f"An error occurred during the training pipeline: {e}")
            self.logger.error(traceback.format_exc())
            # Optionally re-raise or handle as needed
            raise
        self.logger.info("Training pipeline finished successfully.")
        return self.results

# Example usage (for testing the trainer class itself)
# if __name__ == '__main__':
#     logger.info("Starting HybridTrainer example execution...")
#     example_config = {
#         'data_path': 'data/sample_ohlcv.csv',  
#         'model_save_path': 'rl_models/test_hybrid_ppo',
#         'log_dir': 'logs/test_hybrid_trainer/',
#         'total_timesteps': 10000, 
#         'eval_freq': 2000,
#         'log_freq': 500,
#         'window_size': 30,      
#         'initial_indicators': ['rsi_14', 'macd_12_26_9'], 
#         'train_max_steps': 200, 
#         'val_max_steps': 100,   
#         'test_max_steps': 100,  
#         'use_curriculum': False,
#         'policy': 'MlpPolicy', 
#         'start_capital': 10000,
#         'transaction_cost_pct': 0.001, # This will be used by TradingEnv
#         'transaction_cost': 0.001, # This will be used for RiskManager
#         'reward_scale': 1.0,
#         'train_size': 0.7,
#         'val_size': 0.15,
#         'test_size': 0.15,
#         'evaluate_on_validation_set': True,
#         'evaluate_on_test_set': True,
#         'progress_bar': True,
#         'ppo_verbose': 0, 
#         'device': 'cpu',
#         'overwrite_metrics_file': True, # Example of enabling overwrite
#
#         # RiskManager specific parameters (can also be taken from config_vals if not here)
#         'max_position_size': MAX_POSITION_SIZE, 
#         'max_drawdown': MAX_DRAWDOWN,
#         'stop_loss_pct': STOP_LOSS,
#         'risk_per_trade': RISK_PER_TRADE,
#         'take_profit_pct': 0.04, # Example explicit value
#         'trailing_stop_pct': TRAILING_STOP_PERCENT,
#
#         # RiskConfig detail parameters (if one wants to fine-tune RiskConfig behavior separately)
#         'base_stop_loss': STOP_LOSS, # For the RiskConfig object
#         'base_take_profit': 0.04,   # For the RiskConfig object
#     }
#
#     sample_data_path = example_config['data_path']
#     # The dummy data creation is now part of load_and_preprocess_data, 
#     # so it will be triggered if the file doesn't exist when trainer.load_and_preprocess_data() is called.
#     # os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
#     # if not os.path.exists(sample_data_path):
#     #     logger.info(f"Creating dummy sample data at {sample_data_path}")
#     #     num_rows = 1000 
#     #     date_rng = pd.date_range(start='2022-01-01', periods=num_rows, freq='H')
#     #     open_prices = np.random.uniform(90, 110, num_rows)
#     #     low_prices = open_prices - np.random.uniform(0, 5, num_rows)
#     #     high_prices = open_prices + np.random.uniform(0, 5, num_rows)
#     #     close_prices = np.random.uniform(low_prices, high_prices)
#     #     volume = np.random.uniform(1000, 10000, num_rows)
#     #     
#     #     dummy_df = pd.DataFrame({
#     #         'timestamp': date_rng,
#     #         'open': open_prices,
#     #         'high': high_prices,
#     #         'low': low_prices,
#     #         'close': close_prices,
#     #         'volume': volume
#     #     })
#     #     dummy_df.set_index('timestamp', inplace=True)
#     #     nan_indices = np.random.choice(dummy_df.index, size=int(num_rows * 0.01), replace=False)
#     #     if 'volume' in dummy_df.columns:
#     #         dummy_df.loc[nan_indices, 'volume'] = np.nan
#
#     #     dummy_df.to_csv(sample_data_path)
#     #     logger.info(f"Dummy data created at {sample_data_path} with {len(dummy_df)} rows.")
#     # else:
#     #     logger.info(f"Using existing sample data from {sample_data_path}")
#
#     try:
#         trainer = HybridTrainer(config=example_config)
#         final_results = trainer.run_training_pipeline()
#         logger.info(f"Example pipeline finished. Check logs in {example_config['log_dir']} and models in {os.path.dirname(example_config['model_save_path'])}")
#
#         saved_model_path = f"{example_config['model_save_path']}_final.zip"
#         if os.path.exists(saved_model_path):
#             logger.info(f"Model successfully saved at {saved_model_path}")
#             if trainer.test_env is not None and trainer.model is not None:
#                 # Ensure test_env is reset and suitable for PPO.load() if it has wrappers
#                 # PPO.load might need the original (or similarly wrapped) env to correctly build the policy network
#                 # For custom policies, providing an env to PPO.load is often necessary.
#                 env_for_loading = trainer.test_env # This should be the Monitor-wrapped env or similar
#                 loaded_model = PPO.load(saved_model_path, env=env_for_loading) 
#                 logger.info("Successfully loaded the saved model.")
#                 obs, _ = trainer.test_env.reset() # Use the instance variable test_env for prediction
#                 action, _ = loaded_model.predict(obs, deterministic=True)
#                 logger.info(f"Prediction with loaded model on test_env: {action}")
#             else:
#                 logger.info("Could not perform load test: test_env or model not available.")
#         else:
#             logger.error(f"Model file not found at {saved_model_path} after training.")
#
#     except Exception as e:
#         logger.error(f"Error in HybridTrainer example execution: {e}", exc_info=True)
