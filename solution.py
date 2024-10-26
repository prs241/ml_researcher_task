import gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, PPO
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
                'price': data.iloc[step]['bid_price_1'],
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
                'price': data.iloc[step]['bid_price_1'],
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

        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            reward = self.compute_components(alpha, shares, idx)
            slippage.append(reward[0])
            market_impact.append(reward[1])

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
        current_price = self.data.iloc[self.current_step]['bid_price_1']
        return np.array([current_price, self.remaining_shares], dtype=np.float32)

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['bid_price_1']
        current_volume = self.data.iloc[self.current_step][['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].sum()

        shares_to_sell = max(0.01, action[0]) * self.remaining_shares
        shares_to_sell = min(shares_to_sell, current_volume)
        
        if self.benchmark:
            slippage, market_impact = self.benchmark.compute_components(alpha=0.001, shares=int(shares_to_sell), idx=self.current_step)
            transaction_cost = slippage + market_impact
        else:
            transaction_cost = shares_to_sell * current_price * 0.001

        self.remaining_shares -= shares_to_sell
        self.current_step += 1
        
        done = self.current_step >= self.trading_horizon or self.remaining_shares <= 0
        reward = -transaction_cost
        
        return self._get_observation(), reward, done, {}

data = pd.read_csv('AAPL_Quotes_Data.csv')
benchmark = Benchmark(data)
env = DummyVecEnv([lambda: TradingEnv(data, benchmark=benchmark)])

sac_model = SAC('MlpPolicy', env, verbose=1)
sac_model.learn(total_timesteps=100000)
sac_model.save('sac_trading_model')
obs = env.reset()
timestamps_sac, share_sizes_sac, slippage_costs_sac, market_impacts_sac = [], [], [], []

for i in range(env.get_attr('trading_horizon')[0]):
    action, _ = sac_model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    timestamps_sac.append(data.iloc[i]['timestamp'])
    shares_sold = max(0.01, action[0]) * env.get_attr('remaining_shares')[0]
    share_sizes_sac.append(shares_sold)
    slippage, market_impact = benchmark.compute_components(alpha=0.001, shares=int(shares_sold), idx=i)
    slippage_costs_sac.append(slippage)
    market_impacts_sac.append(market_impact)
    if done:
        break

sac_trade_schedule = pd.DataFrame({'timestamp': timestamps_sac, 'shares_to_sell': share_sizes_sac, 'slippage': slippage_costs_sac, 'market_impact': market_impacts_sac})

ppo_model = PPO('MlpPolicy', env, verbose=1)
ppo_model.learn(total_timesteps=100000)
ppo_model.save('ppo_trading_model')

obs = env.reset()
timestamps_ppo, share_sizes_ppo, slippage_costs_ppo, market_impacts_ppo = [], [], [], []

for i in range(env.get_attr('trading_horizon')[0]):
    action, _ = ppo_model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    timestamps_ppo.append(data.iloc[i]['timestamp'])
    shares_sold = max(0.01, action[0]) * env.get_attr('remaining_shares')[0]
    share_sizes_ppo.append(shares_sold)
    slippage, market_impact = benchmark.compute_components(alpha=0.001, shares=int(shares_sold), idx=i)
    slippage_costs_ppo.append(slippage)
    market_impacts_ppo.append(market_impact)
    if done:
        break

ppo_trade_schedule = pd.DataFrame({'timestamp': timestamps_ppo, 'shares_to_sell': share_sizes_ppo, 'slippage': slippage_costs_ppo, 'market_impact': market_impacts_ppo})

twap_trades = benchmark.get_twap_trades(data, initial_inventory=1000)
vwap_trades = benchmark.get_vwap_trades(data, initial_inventory=1000)

twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, data, preferred_timeframe=390)
vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, data, preferred_timeframe=390)

sac_total_cost = sum(slippage_costs_sac) + sum(market_impacts_sac)
ppo_total_cost = sum(slippage_costs_ppo) + sum(market_impacts_ppo)
twap_total_cost = sum(twap_slippage) + sum(twap_market_impact)
vwap_total_cost = sum(vwap_slippage) + sum(vwap_market_impact)

print(f'SAC Total Cost: {sac_total_cost}')
print(f'PPO Total Cost: {ppo_total_cost}')
print(f'TWAP Total Cost: {twap_total_cost}')
print(f'VWAP Total Cost: {vwap_total_cost}')
