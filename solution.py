import gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces

class Benchmark:
    
    def __init__(self, data):
        """
        Initializes the Benchmark class with provided market data.
        """
        self.data = data

    def get_twap_trades(self, data, initial_inventory, preferred_timeframe=390):
        total_steps = len(data)
        twap_shares_per_step = initial_inventory / preferred_timeframe
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            size_of_slice = min(twap_shares_per_step, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['bid_price_1'],  # Use bid price 1 as a proxy for trade price
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)

    def get_vwap_trades(self, data, initial_inventory, preferred_timeframe=390):
        total_volume = data[['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].sum().sum()
        total_steps = len(data)
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            volume_at_step = data.iloc[step][['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].sum()
            size_of_slice = (volume_at_step / total_volume) * initial_inventory
            size_of_slice = min(size_of_slice, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['bid_price_1'],  # Use bid price 1 as a proxy for trade price
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)

    def calculate_vwap(self, idx, shares):
        bid_prices = self.data.iloc[idx][['bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5']]
        bid_sizes = self.data.iloc[idx][['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']]
        cumsum = 0
        for i, size in enumerate(bid_sizes):
            cumsum += size
            if cumsum >= shares:
                break
        return np.sum(bid_prices[:i+1] * bid_sizes[:i+1]) / np.sum(bid_sizes[:i+1])

    def compute_components(self, alpha, shares, idx):
        actual_price = self.calculate_vwap(idx, shares)
        Slippage = (self.data.iloc[idx]['bid_price_1'] - actual_price) * shares
        Market_Impact = alpha * np.sqrt(shares)
        return np.array([Slippage, Market_Impact])

    def simulate_strategy(self, trades, data, preferred_timeframe):
        slippage = []
        market_impact = []
        alpha = 4.439584265535017e-06 
        rewards = []
        shares_traded = []

        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            reward = self.compute_components(alpha, shares, idx)
            slippage.append(reward[0])
            market_impact.append(reward[1])
            shares_traded.append(shares)
            rewards.append(reward)

        return slippage, market_impact

class TradingEnv(gym.Env):
    def __init__(self, data, total_shares=1000, trading_horizon=390, benchmark=None):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.total_shares = total_shares
        self.trading_horizon = trading_horizon
        self.current_step = 0
        self.remaining_shares = total_shares
        self.benchmark = benchmark
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.remaining_shares = self.total_shares
        return self._get_observation()

    def _get_observation(self):
        current_price = self.data.iloc[self.current_step]['bid_price_1']  # Use bid price 1 as the current price proxy
        return np.array([current_price, self.remaining_shares], dtype=np.float32)

    def step(self, action):
        # Debugging actions and remaining shares
        print(f"Action at step {self.current_step}: {action[0]}")
        
        current_price = self.data.iloc[self.current_step]['bid_price_1']  # Use bid price 1 as the current price proxy
        current_volume = self.data.iloc[self.current_step][['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].sum()

        shares_to_sell = max(0.01, action[0]) * self.remaining_shares  # Ensure some shares are sold
        shares_to_sell = min(shares_to_sell, current_volume)  # Sell only what volume allows
        
        # Debug remaining shares and volume
        print(f"Remaining shares: {self.remaining_shares}, Shares to sell: {shares_to_sell}, Volume available: {current_volume}")

        if self.benchmark:
            slippage, market_impact = self.benchmark.compute_components(alpha=0.001, shares=int(shares_to_sell), idx=self.current_step)
            transaction_cost = slippage + market_impact
        else:
            transaction_cost = shares_to_sell * current_price * 0.001  # Basic transaction cost

        self.remaining_shares -= shares_to_sell
        self.current_step += 1
        
        done = self.current_step >= self.trading_horizon or self.remaining_shares <= 0
        reward = -transaction_cost  # Negative of transaction cost as reward
        
        return self._get_observation(), reward, done, {}

# Load your AAPL Quotes market data
data = pd.read_csv('AAPL_Quotes_Data.csv')

# Initialize the benchmark with the market data
benchmark = Benchmark(data)

# Create the environment
env = DummyVecEnv([lambda: TradingEnv(data, benchmark=benchmark)])

# Train the SAC model
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
model.save('sac_trading_model')

# Load the model and simulate trading
model = SAC.load('sac_trading_model')
obs = env.reset()

timestamps = []
share_sizes = []
slippage_costs = []
market_impacts = []

for i in range(env.get_attr('trading_horizon')[0]):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    
    timestamps.append(data.iloc[i]['timestamp'])
    shares_sold = max(0.01, action[0]) * env.get_attr('remaining_shares')[0]  # Ensure non-zero shares sold
    share_sizes.append(shares_sold)
    
    slippage, market_impact = benchmark.compute_components(alpha=0.001, shares=int(shares_sold), idx=i)
    slippage_costs.append(slippage)
    market_impacts.append(market_impact)
    
    if done:
        break

trade_schedule = pd.DataFrame({'timestamp': timestamps, 'shares_to_sell': share_sizes, 'slippage': slippage_costs, 'market_impact': market_impacts})
trade_schedule.to_json('sac_trade_schedule_with_costs.json', orient='records')

# Generate TWAP and VWAP trades using the benchmark
twap_trades = benchmark.get_twap_trades(data, initial_inventory=1000)
vwap_trades = benchmark.get_vwap_trades(data, initial_inventory=1000)

# Simulate transaction costs for TWAP and VWAP
twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, data, preferred_timeframe=390)
vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, data, preferred_timeframe=390)

# Calculate total costs for each strategy
sac_total_cost = sum(slippage_costs) + sum(market_impacts)
twap_total_cost = sum(twap_slippage) + sum(twap_market_impact)
vwap_total_cost = sum(vwap_slippage) + sum(vwap_market_impact)

print(f'SAC Total Cost: {sac_total_cost}')
print(f'TWAP Total Cost: {twap_total_cost}')
print(f'VWAP Total Cost: {vwap_total_cost}')
