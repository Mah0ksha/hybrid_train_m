from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import time
import numpy as np
import torch
from logger_setup import setup_logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

logger = setup_logger()

# Simple progress callback
class ProgressCallback(BaseCallback):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.last = ""
        self.last_progress_log = 0
        self.progress_log_interval = interval
        
    def _on_step(self) -> bool:
        if time.time() - self.last_progress_log >= self.progress_log_interval:
            steps = self.num_timesteps
            self.last_progress_log = time.time()
            print(f"RL timesteps: {steps}", end="\r")
        return True

# Learning rate scheduler with warmup
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Backtest callback
class BacktestCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=1000, deterministic=False):
        super().__init__(eval_env, eval_freq=eval_freq, deterministic=deterministic)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
        return True

class CustomEvalCallback(BaseCallback):
    """
    Custom callback for evaluating the agent on a separate validation environment
    during training. It logs detailed trading metrics to TensorBoard and supports early stopping.
    """
    def __init__(self, eval_env: VecEnv, eval_freq: int, log_path: str, 
                 deterministic: bool = True, n_eval_episodes: int = 1,
                 early_stopping_patience: int = 5, 
                 early_stopping_metric: str = 'sharpe_ratio'):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.save_path = os.path.join(log_path, 'best_model')
        
        self.best_metric_value = -np.inf
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.patience_counter = 0
        
        self.training_evaluation_history = []
        self.lr_scheduler = None
        self.curriculum_wrapper = None

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_curriculum_wrapper(self, curriculum_wrapper):
        self.curriculum_wrapper = curriculum_wrapper

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            logger.info(f"Starting evaluation at step {self.n_calls}...")
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic
            )
            logger.info(f"Evaluation finished at step {self.n_calls}. Mean reward: {mean_reward:.2f}")
            
            metrics = self.eval_env.env_method('get_metrics')[0]
            
            self.logger.record("eval/mean_reward", mean_reward)
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.record(f"eval/{key}", value)

                metrics['timestep'] = self.n_calls
                metrics['mean_reward'] = mean_reward
                self.training_evaluation_history.append(metrics)
                
                # --- Early Stopping and Best Model Saving ---
                current_metric_value = metrics.get(self.early_stopping_metric)

                if current_metric_value is not None and isinstance(current_metric_value, (int, float)):
                    if current_metric_value > self.best_metric_value:
                        self.best_metric_value = current_metric_value
                        self.patience_counter = 0
                        self.model.save(self.save_path)
                        logger.info(f"New best model saved! {self.early_stopping_metric}: {self.best_metric_value:.3f} at timestep {self.n_calls}")
                        if self.lr_scheduler: self.lr_scheduler.resume()
                        if self.curriculum_wrapper: self.curriculum_wrapper.resume()
                    else:
                        self.patience_counter += 1
                        logger.info(f"No improvement in {self.early_stopping_metric}. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                        if self.lr_scheduler: self.lr_scheduler.pause()
                        if self.curriculum_wrapper: self.curriculum_wrapper.pause()

                    if self.patience_counter >= self.early_stopping_patience:
                        logger.warning(f"Stopping training early. {self.early_stopping_metric} did not improve for {self.early_stopping_patience} evaluations.")
                        return False  # Returning False stops the training
                else:
                    logger.warning(f"Early stopping metric '{self.early_stopping_metric}' not found in metrics. Saving model based on mean reward.")
                    # Fallback to mean_reward if the desired metric is not available
                    if isinstance(mean_reward, (int, float)) and mean_reward > self.best_metric_value:
                         self.best_metric_value = mean_reward
                         self.model.save(self.save_path)

            self.logger.dump(self.num_timesteps)

        return True

class LrSchedulerCallback(BaseCallback):
    def __init__(self, initial_lr, final_lr, total_timesteps, power=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
        self.power = power
        self.paused = False

    def pause(self):
        self.paused = True
        logger.info("Learning rate scheduler paused.")

    def resume(self):
        self.paused = False
        logger.info("Learning rate scheduler resumed.")

    def _on_step(self):
        if self.paused:
            return True

        fraction = 1.0 - (self.num_timesteps / self.total_timesteps)
        lr = self.final_lr + (self.initial_lr - self.final_lr) * (fraction ** self.power)
        
        # Access the optimizer from the model
        if self.model and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            self.optimizer = self.model.policy.optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return True
