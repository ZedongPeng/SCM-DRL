# -*- coding: utf-8 -*-
# @Author: Zedong Peng
# @Date:   2019-07-25 13:38:18
# @Last Modified by:   Zedong Peng
# @Last Modified time: 2019-07-25 13:42:14

'''
continuous action space
use np.clip to map infeasible action to feasible action
use proportion to scale the actions to satisfy sum constraint
random demand
'''
import numpy as np
import itertools
from gym import spaces


class SupplyDistribution:
    """
    The supply distribution environment
    """

    def __init__(self, n_stores=3, cap_truck=2, prod_cost=1, max_prod=8,
                 store_cost=np.array([0.01, 0.1, 0.1, 0.1]), truck_cost=np.array([1, 2, 3]),
                 cap_store=np.array([20, 5, 5, 5]), penalty_cost=2, price=30, gamma=0.90,
                 max_demand=8, episode_length=48):
        """
        :param n_stores: the number of stores
        :param cap_truck: capacity of truck, to determine how many trucks we need
        :param prod_cost: production cost
        :param store_cost: storage cost(inventory cost)
        :param truck_cost: cost of truck
        :param cap_store: capacity of storage
        :param penalty_cost: penalty cost
        :param price: price of products
        """
        self.n_stores = n_stores
        self.s = np.zeros(self.n_stores + 1, dtype=np.float32)  # state
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.demand_old = np.zeros(self.n_stores, dtype=int)
        self.price = price
        self.max_prod = max_prod
        # capacity
        # self.cap_store = np.ones(n_stores + 1, dtype=int)
        self.cap_store = cap_store
        self.cap_truck = cap_truck
        # costs
        self.prod_cost = prod_cost
        self.store_cost = np.array(store_cost)
        self.truck_cost = np.array(truck_cost)
        self.penalty_cost = penalty_cost
        # demand
        self.max_demand = max_demand
        self.episode_length = episode_length
        # other variables
        self.gamma = gamma
        self.t = 0

        self.action_space = spaces.Box(low=np.zeros(self.n_stores + 1), high=cap_store, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.full(3 * self.n_stores + 1, -np.inf),
                                            high=np.append(cap_store, self.n_stores * [np.inf, np.inf]), dtype=np.float32)
        # observation_space might be negative
        self.reset()

    def reset(self):
        """
        Resets the environment to the starting conditions
        """
        self.s = self.cap_store / 2  # np.zeros(self.n_stores + 1, dtype=int)  # +1 Because the central warehouse is not counted as a store
        #self.s[0] = self.cap_store[0]/2
        self.t = 0
        # Initialize demand and update it directly to avoid jumps in demand of first step
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.update_demand()
        self.demand_old = self.demand.copy()  # np.zeros(self.n_stores, dtype=int)
        return np.hstack((self.s.copy(), self.demand.copy(), self.demand_old.copy()))  # return current state

    def step(self, action):
        # Update state
        # print('Action: ', action)
        upper_bound = self.cap_store - self.s
        upper_bound[0] = upper_bound[0] + sum(action[1:])
        action = np.clip(action, np.zeros(self.n_stores + 1), upper_bound)
        # print('action[1:]: ', action[1:])
        # print('sum(action[1:]): ', sum(action[1:]))
        # print('self.s[0]: ', self.s[0])
        # print('sum(action[1:]) > self.s[0]: ', sum(action[1:]) > self.s[0])
        if sum(action[1:]) > self.s[0]:
            action[1:] = action[1:] * self.s[0] / sum(action[1:])
        action = np.around(action, 4)
        # print("actual action: ", action)

        self.s[0] = min(self.s[0] + action[0] - sum(action[1:]), self.cap_store[0])
        if self.s[0] < 0:
            with open('test.txt', 'a') as f:
                f.write(str(self.s[0]) + '\n')
        self.s[0] = max(self.s[0], 0)
        # print('self.s[0]: ', self.s[0])
        self.s[1:] = np.minimum(self.s[1:] - self.demand + action[1:], self.cap_store[1:])
        # print('self.s[1:]: ', self.s[1:], '\n')
        self.s = np.around(self.s, 4)
        # print('self.s[1:] ', self.s[1:], 'self.demand ', self.demand, 'action[1:] ', action[1:])
        # print(self.s)

        # Update reward
        reward = (sum(self.demand) * self.price  # revenue
                  - action[0] * self.prod_cost   # production cost
                  - np.sum(np.maximum(np.zeros(self.n_stores + 1), self.s[:self.n_stores + 1]) * self.store_cost)
                  # Changed to + so that penalty cost actually decrease reward -- Luke 26/02
                  + np.sum(np.minimum(np.zeros(self.n_stores + 1), self.s[:self.n_stores + 1])) * self.penalty_cost
                  - np.sum(np.ceil(action[1:] / self.cap_truck) * self.truck_cost))

        info = "Demand was: ", self.demand

        # Define state
        state = np.hstack((self.s.copy(), self.demand.copy(), self.demand_old.copy()))  # hstack: Stack arrays in sequence horizontally

        # Update demand old
        self.demand_old = self.demand.copy()

        # Update t
        self.t += 1

        # Update demand
        self.update_demand()

        # Set if done 0 since unused
        done = 0
        # print("State: ", np.around(state, 2), '\n', 'Reward: ', np.around(reward, 2), '\n')
        return state, reward, done, info

    def update_demand(self):
        """
        Updates the demand using the update demand function
        :return:
        """
        demand = np.zeros(self.n_stores, dtype=int)
        for i in range(self.n_stores):
            # We need an integer so we use the ceiling because if there is demand then we asume the users will buy
            # what they need and keep the rests. We use around to get an integer out of it.

            # try not random:
            # demand[i] = int(np.floor(.5 * self.max_demand * np.sin(np.pi * (self.t + 2 * i) / (.5 * self.episode_length) -
            #                                                        np.pi) + .5 * self.max_demand + np.random.randint(0, 2)))  # 2 month cycles

            demand[i] = np.random.randint(low=0, high=self.max_demand)

            # demand[i] = int(np.ceil(1.5 * np.sin(2 * np.pi * (self.t + i) / 26) + 1.5 + np.random.randint(0, 2)))
        self.demand = demand

    def possible_action(self, action):
        if sum(action[1:]) > self.s[0]:
            return False
        if self.s[0] + action[0] - sum(action[1:]) > self.cap_store[0]:
            return False
        for i in range(1, len(action)):
            if self.cap_store[i] - self.s[i] < action[i]:
                return False
        return True

    def allowed_action(self):
        a_allowed = np.zeros(self.action_dim())
        for i in range(self.action_dim()):
            if self.possible_action(self.discrete2continuous[i]):
                a_allowed[i] = 1

        # warning message
        if(sum(a_allowed)) < 1:
            print("Warning: we have an action space of zero!!!!!!!!!")
        # print(a_allowed)
        return a_allowed
