class MarketSimulator:
    def __init__(self, num_agents, agent_params_ranges, initial_df):
        self.num_agents = num_agents
        self.agent_params_ranges = agent_params_ranges
        self.initial_df = self.transform_initialdf(initial_df)  # Transform and set initial_df
        self.current_df = self.initial_df.copy()  # Prepare the current order book for the first iteration
        self.order_book_history = {}
        self.agents_params = self.generate_agents_params()
        
#     def initialize_simulation(self, initial_order_book_df):
#         self.initial_df = initial_order_book_df.copy()
#         self.order_book_history = {}
#         self.agents_params = self.generate_agents_params()

    def transform_initialdf(self,df):    
    
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        first_timestamp = df['timestamp'].min()

        # Filter the DataFrame for only the first timestamp
        df_first_timestamp = df[df['timestamp'] == first_timestamp]

        # Function to prepare data
        def prepare_data(df, depth_suffix):
            df['rank'] = df['depth'].apply(lambda x: int(x.replace(depth_suffix, '')))
            df = df.drop(columns=['depth']).sort_values(by='rank')
            return df

        # Split bids and asks for the first timestamp
        bids = prepare_data(df_first_timestamp[df_first_timestamp['depth'].str.contains('bid')].copy(), '_bid')
        asks = prepare_data(df_first_timestamp[df_first_timestamp['depth'].str.contains('ask')].copy(), '_ask')

        # Reset index to make merging by index easier
        bids.reset_index(drop=True, inplace=True)
        asks.reset_index(drop=True, inplace=True)

        # Merge bids and asks
        result_df = pd.merge(bids, asks, left_index=True, right_index=True, suffixes=('_bid', '_ask'))

        # Rename columns to match the desired output
        result_df.rename(columns={'price_bid': 'Bid Price', 'volume_bid': 'Bid Volume',
                                  'price_ask': 'Ask Price', 'volume_ask': 'Ask Volume'}, inplace=True)

        # Select only the relevant columns
        result_df = result_df[['Bid Price', 'Bid Volume', 'Ask Price', 'Ask Volume']]

        return result_df
        
    def generate_agents_params(self):
        agents_params = []
        for _ in range(self.num_agents):
            agent_params = {param: random.uniform(*range_) if isinstance(range_, tuple) else random.choice(range_)
                            for param, range_ in self.agent_params_ranges.items()}
            agents_params.append(agent_params)
        return agents_params
    
    def compute_coeff(self, xi, gamma, delta, A, k):
        inv_k = np.divide(1, k)
        c1 = np.log(1 + xi * delta * inv_k) / (xi * delta) 
        c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
        return c1, c2

    def calculate_first_timestamp_prices(self, df, params):
        # Extract parameters
        gamma, delta, A, k, vol, order_qty, adj1, adj2, pos = (
            params['gamma'], params['delta'], params['A'], params['k'],
            params['vol'], params['order_qty'], params['adj1'],
            params['adj2'], params['pos']
        )

        # Inline coefficient computation and calculations
        c1, c2 = self.compute_coeff(gamma, gamma, delta, A, k)
        half_spread = (c1 + delta / 2 * c2 * vol) * adj1
        skew = c2 * vol * adj2
        #     half_spread = ((gamma + gamma + A) + delta / 2 * (delta + k) * vol) * adj1
        #     skew = (delta + k) * vol * adj2
        #print(half_spread,skew)
#         first_timestamp = df['timestamp'].iloc[0]
#         df_first = df[df['timestamp'] == first_timestamp]

        mid_price_tick = (df['Bid Price'].iloc[0] + df['Ask Price'].iloc[0]) / 2
        bid_depth = half_spread + skew * pos
        ask_depth = half_spread - skew * pos
        #print(bid_depth,ask_depth)
        # Return new prices as a Series or DataFrame
        new_bid_price = mid_price_tick - bid_depth
        new_ask_price = mid_price_tick + ask_depth
        
        return pd.Series({
            'new_bid_price': new_bid_price,
            'new_bid_vol': order_qty,
            'new_ask_price': new_ask_price,
            'new_ask_vol': order_qty
            
        })

    def simulate_trades(self, prices_for_first_timestamp,df):
        #print(prices_for_first_timestamp)
        bids=[]
        asks=[]
        df_bids = df[['Bid Price', 'Bid Volume']].values
        df_asks = df[['Ask Price', 'Ask Volume']].copy()

        df_asks.columns = ['price', 'volume']

        bids = np.vstack([np.array([[q[0], q[1]] for q in prices_for_first_timestamp]), df_bids])
        asks = np.vstack([np.array([[q[2], q[3]] for q in prices_for_first_timestamp]), df_asks.values])
        
        bids = bids[bids[:, 0].argsort()[::-1]]
        asks = asks[asks[:, 0].argsort()]
#         bids = np.array([q[0] for q in prices_for_first_timestamp])
#         asks = np.array([q[1] for q in prices_for_first_timestamp])
        print(bids)
    
        # Sort bids descending and asks ascending
        sorted_bids = np.sort(bids)[::-1]
        sorted_asks = np.sort(asks)
#         #print(sorted_bids)
#         num_trades = np.searchsorted(sorted_asks, sorted_bids, side='right')
#         print(num_trades)
#         n_trades = np.count_nonzero(num_trades)
#         # Remove matched trades
#         remaining_bids = sorted_bids[n_trades:]
#         remaining_asks = sorted_asks[n_trades:]

#         return remaining_bids, remaining_asks

        i, j = 0, 0
        while i < len(bids) and j < len(asks):
            bid_price, bid_vol = bids[i]
            ask_price, ask_vol = asks[j]

            if bid_price >= ask_price:
                # Determine the volume that can be traded
                trade_vol = min(bid_vol, ask_vol)

                # Update volumes after trade
                bids[i, 1] -= trade_vol
                asks[j, 1] -= trade_vol

                # Move to next bid or ask if volume is exhausted
                if bids[i, 1] == 0:
                    i += 1
                if asks[j, 1] == 0:
                    j += 1
            else:
                # No more matches possible if bid < ask
                break
            
        # Filter out bids and asks that have been completely traded
        remaining_bids = bids[i:]
        remaining_asks = asks[j:]

        #print(remaining_bids)
        return remaining_bids, remaining_asks

    def create_order_book(self, remaining_bids, remaining_asks):

        def aggregate_volumes(pairs):
            # Round the prices to two decimals and separate price and volume
            rounded_prices = np.round(pairs[:, 0], 2)
            volumes = pairs[:, 1]

            # Aggregate volumes by rounded price
            aggregated_data = {}
            for price, volume in zip(rounded_prices, volumes):
                if price in aggregated_data:
                    aggregated_data[price] += volume
                else:
                    aggregated_data[price] = volume

            unique_prices = np.array(list(aggregated_data.keys()))
            total_volumes = np.array(list(aggregated_data.values()))

            sorted_indices = np.argsort(unique_prices)
            sorted_prices = unique_prices[sorted_indices]
            sorted_volumes = total_volumes[sorted_indices]

            return sorted_prices, sorted_volumes

        bid_prices, bid_volumes = aggregate_volumes(np.array(remaining_bids))
        ask_prices, ask_volumes = aggregate_volumes(np.array(remaining_asks))
        order_book_bids = pd.DataFrame({'Bid Price': bid_prices[::-1], 'Bid Volume': bid_volumes[::-1]})
        order_book_asks = pd.DataFrame({'Ask Price': ask_prices, 'Ask Volume': ask_volumes})

        min_length = min(len(order_book_bids), len(order_book_asks))
        order_book_bids = order_book_bids.head(min_length)
        order_book_asks = order_book_asks.head(min_length)

        order_book = pd.concat([order_book_bids.reset_index(drop=True), 
                                order_book_asks.reset_index(drop=True)], axis=1)

        return order_book
    
    def run_simulation(self, n_iterations):
        self.order_book_history[0] = self.initial_df.copy()
        for iteration in range(1, n_iterations + 1):
            # Placeholder logic for updating the order book; integrate with real market dynamics for practical use
            prices_for_current_timestamp = [self.calculate_first_timestamp_prices(self.current_df, params) for params in self.agents_params]
            remaining_bids, remaining_asks = self.simulate_trades(prices_for_current_timestamp,self.current_df)
            new_order_book = self.create_order_book(remaining_bids, remaining_asks)
            self.order_book_history[iteration] = new_order_book
            self.current_df = new_order_book
        self.finalize_simulation()
        return self.order_book_history

    def finalize_simulation(self):
        # Placeholder for any finalization logic
        print("Simulation completed. Number of iterations:", len(self.order_book_history))

    

# Example usage
num_agents = 100
agent_params_ranges = {
    'gamma': (0.01, 0.1), 'delta': (0.5, 1.5), 'A': (1, 3), 'k': (1, 50),
    'vol': (0.05, 0.25), 'order_qty': [1], 'adj1': (0.5, 1), 'adj2': (0.001, 0.01),
    'pos': (0, 1)  # Using a list for discrete choices
}

#market_simulator = MarketSimulator(num_agents, agent_params_ranges)
market_simulator = MarketSimulator(num_agents, agent_params_ranges, initdf)
histo = market_simulator.run_simulation(5)
# agents_params = market_simulator.generate_agents_params()
# #print(final_df)
# # Assuming final_df is provided and calculate_first_timestamp_prices() is implemented
# prices_for_first_timestamp = [market_simulator.calculate_first_timestamp_prices(orderbook, params) for params in agents_params]
# #print(prices_for_first_timestamp)
# remaining_bids, remaining_asks = market_simulator.simulate_trades(prices_for_first_timestamp,orderbook)
# #print(remaining_asks)
# order_book = market_simulator.create_order_book(remaining_bids, remaining_asks)

