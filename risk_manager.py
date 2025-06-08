from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

# Import Constants
from config_vals import (
    CURRENT_UTC_TIME, CURRENT_USER, TIME_FILTERS, UNIVERSE_FILE, RAW_DIR, 
    START_CAP, TRANSACTION_COST, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS, 
    MAX_POSITION_SIZE, TRAILING_STOP_PERCENT, SCALE_IN_LEVELS, SCALE_OUT_LEVELS, 
    MAX_POSITION_DURATION, MIN_LIQUIDITY_THRESHOLD, DYNAMIC_POSITION_SCALING, 
    MAX_LEVERAGE, MIN_POSITION_SIZE, POSITION_STEP_SIZE, REWARD_SCALE, 
    MAX_POSITION_DURATION_HOURS, MIN_TRADE_INTERVAL_MINUTES, 
    OPTIMAL_POSITION_SIZE_MIN, OPTIMAL_POSITION_SIZE_MAX, BASE_TIMESTEPS, 
    MAX_TIMESTEPS, PERFORMANCE_THRESHOLD, CHUNK_SIZE, MARKET_HOURS, 
    INPUT_DIM, SEQUENCE_LENGTH, WINDOW_SIZE, PREDICTION_WINDOW, EVAL_DAYS, 
    VOLATILITY_ADJUSTMENT, BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, 
    ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM, N_STEPS, N_EPOCHS, N_ENVS, 
    RL_TIMESTEPS, EVAL_STEPS, WARMUP_STEPS, PROGRESS_INTERVAL
)

# Import logger first to ensure it's available for other imports
from logger_setup import logger, setup_logger

# Import RiskConfig
from risk_config import RiskConfig

# Re-initialize logger with custom settings
logger = setup_logger()

if TYPE_CHECKING:
    from risk_config import RiskConfig

class RiskManagerEnvInterface(ABC):
    """Defines the interface for the RiskManager to communicate back with the TradingEnv."""
    @abstractmethod
    def update_env_on_close(self, exit_price: float, exit_time: datetime, realized_pnl: float, closed_by: str) -> None:
        """Callback to update the TradingEnv's state when a position is closed by RiskManager."""
        pass

    @abstractmethod
    def set_trade_exit_details(self, exit_price: float, exit_time: datetime, exit_reason: str) -> None:
        """Callback to set specific exit details in TradingEnv's metrics or state for the closed trade."""
        pass

class RiskManager:
    """
    Manages position-level risk, including setting stop-loss/take-profit levels
    and checking if they have been triggered.
    """
    def __init__(self, config: RiskConfig, initial_capital: float, env_interface: RiskManagerEnvInterface | None = None):
        self.config = config
        self.env_interface = env_interface
        self.initial_capital = float(initial_capital)

        # Portfolio-level risk
        self._peak_portfolio_value = float(initial_capital)
        self.current_drawdown = 0.0

        # Position-level state
        self.entry_price: float = 0.0
        self.stop_loss_price: float | None = None
        self.take_profit_price: float | None = None
        self.current_position_type: str | None = None  # 'long' or 'short'
        self.position_size_abs: float = 0.0
        
        # Use config values directly
        self.stop_loss_pct = self.config.base_stop_loss
        self.take_profit_pct = self.config.base_take_profit
        self.transaction_cost_pct = self.config.transaction_cost
        
        self.reset()
        logger.info("RiskManager initialized.")

    def reset(self) -> None:
        """Resets all position-specific states."""
        self.entry_price = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.current_position_type = None
        self.position_size_abs = 0.0
        logger.debug("RiskManager state has been reset.")

    def open_position(self, entry_price: float, position_type: str, position_size_abs: float) -> None:
        """Calculates and sets SL/TP prices for a new position."""
        if position_type not in ['long', 'short']:
            logger.error(f"Invalid position_type: {position_type}")
            return

        self.entry_price = entry_price
        self.current_position_type = position_type
        self.position_size_abs = position_size_abs

        if position_type == 'long':
            if self.stop_loss_pct > 0:
                self.stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            if self.take_profit_pct > 0:
                self.take_profit_price = entry_price * (1 + self.take_profit_pct)
        elif position_type == 'short':
            if self.stop_loss_pct > 0:
                self.stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            if self.take_profit_pct > 0:
                self.take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        logger.info(
            f"RM: Position opened. Type: {position_type}, Entry: {entry_price:.4f}, "
            f"Size: {position_size_abs:.4f}, SL: {self.stop_loss_price or 'N/A'}, TP: {self.take_profit_price or 'N/A'}"
        )

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str) -> float:
        """Internal method to handle position closure logic and return PnL."""
        if self.current_position_type == 'long':
            pnl = (exit_price - self.entry_price) * self.position_size_abs
        elif self.current_position_type == 'short':
            pnl = (self.entry_price - exit_price) * self.position_size_abs
        else:
            return 0.0
        
        # Simplified transaction cost calculation
        total_value = (self.entry_price * self.position_size_abs) + (exit_price * self.position_size_abs)
        transaction_costs = total_value * self.transaction_cost_pct
        realized_pnl = pnl - transaction_costs

        logger.info(
            f"RM: Position closed. Type: {self.current_position_type}, Entry: {self.entry_price:.4f}, "
            f"Exit: {exit_price:.4f}, Size: {self.position_size_abs:.4f}, Reason: {reason}, PnL: {realized_pnl:.4f}"
        )
        
        if self.env_interface:
            self.env_interface.update_env_on_close(exit_price, exit_time, realized_pnl, reason)
        else:
            logger.warning("RiskManager has no environment interface set. Cannot call update_env_on_close.")

        self.reset()
        return realized_pnl

    def check_stop_loss_take_profit(self, current_price: float, current_time: datetime) -> Tuple[bool, str, float]:
        """Checks if the current price has triggered a stop-loss or take-profit."""
        if self.current_position_type is None:
            return False, "NONE", 0.0

        logger.debug(
            f"[RM_SLTP_CHECK] PosType: {self.current_position_type}, CurPrice: {current_price:.4f}, "
            f"Entry: {self.entry_price:.4f}, ConfSL%: {self.stop_loss_pct:.4f}, CalcSL: {self.stop_loss_price or 0:.4f}, "
            f"ConfTP%: {self.take_profit_pct:.4f}, CalcTP: {self.take_profit_price or 0:.4f}"
        )

        pnl = 0.0
        closed = False
        reason = "NONE"

        if self.current_position_type == 'long':
            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                reason = "take_profit_long"
                pnl = self._close_position(current_price, current_time, reason)
                closed = True
            elif self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                reason = "stop_loss_long"
                pnl = self._close_position(current_price, current_time, reason)
                closed = True
        elif self.current_position_type == 'short':
            if self.stop_loss_price is not None and current_price >= self.stop_loss_price:
                reason = "stop_loss_short"
                pnl = self._close_position(current_price, current_time, reason)
                closed = True
            elif self.take_profit_price is not None and current_price <= self.take_profit_price:
                reason = "take_profit_short"
                pnl = self._close_position(current_price, current_time, reason)
                closed = True
        
        return closed, reason, pnl

    def update_peak(self, current_portfolio_value: float) -> None:
        """Update peak portfolio value and calculate current drawdown."""
        if current_portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = current_portfolio_value
        
        if self._peak_portfolio_value > 0:
            self.current_drawdown = (self._peak_portfolio_value - current_portfolio_value) / self._peak_portfolio_value
        else:
            self.current_drawdown = 0.0
            
        logger.debug(f"RM: Peak updated. Peak: {self._peak_portfolio_value}, Current: {current_portfolio_value}, Drawdown: {self.current_drawdown}")

    def has_exceeded_max_drawdown(self) -> bool:
        """Checks if the current drawdown exceeds the maximum allowed."""
        return self.current_drawdown > self.config.max_portfolio_risk

    # --- Methods from old RiskManager to be reviewed/removed/adapted ---
    # The following methods are placeholders or illustrative of old functionality.
    # They need to be critically reviewed. Many might be redundant if TradingEnv
    # and the new RiskManager structure handle their responsibilities.
    def calculate_position_size(self, price: float, portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters.
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        
        # Using risk_per_trade from config
        risk_amount = portfolio_value * self.config.risk_per_trade
        
        # Calculate stop loss to determine risk per share
        stop_loss_pct = self.config.base_stop_loss
        risk_per_share = price * stop_loss_pct
        
        if risk_per_share == 0:
            return 0.0
            
        position_size = risk_amount / risk_per_share
        
        # Max position size constraint
        max_position_value = portfolio_value * self.config.max_position_size
        max_allowed_size = max_position_value / price if price > 0 else 0.0
        
        return min(position_size, max_allowed_size)

    # The following methods are mostly related to internal state or calculations
    # that might be handled differently now or are illustrative.

    def _calculate_time_diff_minutes(self, time1: Union[datetime, int, float, str, None], time2: Union[datetime, int, float, str, None]) -> float:
        """Calculate time difference in minutes, handling different time types."""
        try:
            if time1 is None or time2 is None: return 0.0
            if isinstance(time1, str): time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            elif isinstance(time1, (int, float)): time1 = datetime.fromtimestamp(time1)
            if isinstance(time2, str): time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            elif isinstance(time2, (int, float)): time2 = datetime.fromtimestamp(time2)
            if not isinstance(time1, datetime) or not isinstance(time2, datetime):
                logger.warning(f"Could not convert times to datetime objects: {type(time1)}, {type(time2)}")
                return 0.0
            diff: timedelta = time1 - time2
            return diff.total_seconds() / 60.0
        except Exception as e:
            logger.warning(f"Error calculating time difference: {e}")
            return 0.0

    def update_market_conditions(self, price: Optional[float] = None, volume: Optional[float] = None, volatility: Optional[float] = None) -> None:
        """
        Placeholder for updating RM based on market conditions, if needed for adaptive logic.
        Could use volatility to adjust SL/TP dynamically if RiskConfig supports it.
        """
        if volatility is not None and hasattr(self.config, 'volatility_scaling_factor'):
            # Example: could adjust effective SL/TP percentages based on volatility
            # This logic would need to be defined in RiskConfig or here.
            pass
        logger.debug(f"RM: Market conditions updated. Price: {price}, Vol: {volatility}")

    # Ensure all essential methods for the new API (init, reset, open_pos_calc, check_sltp, _close_pos) are robust.
    # Remaining methods from the old version should be integrated or removed.