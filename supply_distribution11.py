# -*- coding: utf-8 -*-
# @Author: Zedong Peng
# @Date:   2019-07-25 13:38:41
# @Last Modified by:   Zedong Peng
# @Last Modified time: 2019-07-25 13:42:36

'''
discrete action space
allowed action vector (algorithm perspective)
action are float, step length 0.5*capacity
random demand
'''

import numpy as np
import itertools
from gym import spaces
import itertools
import math


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
        self.cap_store = np.ones(n_stores + 1, dtype=int)
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

        self.reset()
        self.actions_per_store = 3

        # available_actions = np.zeros((self.actions_per_store, self.n_stores + 1))
        # available_actions[:, 0] = [0, int(self.max_prod / 2), self.max_prod]
        # for i in range(self.n_stores):
        #     available_actions[:, i + 1] = [0, self.cap_truck, self.cap_truck * 2]

        available_actions = np.zeros(
            (self.n_stores + 1, self.actions_per_store))
        available_actions[0, :] = [0, int(self.max_prod / 2), self.max_prod]
        for i in range(self.n_stores):
            available_actions[i + 1, :] = [0,
                                           self.cap_truck, self.cap_truck * 2]
        self.available_actions = available_actions
        # print(available_actions)

        self.discrete2continuous = np.array(
            [*itertools.product(*(i for i in available_actions))])
        self.action_space = spaces.Discrete(len(self.discrete2continuous))
        self.observation_space = spaces.Box(low=np.full(3 * self.n_stores + 1, -np.inf),
                                            high=np.append(cap_store, self.n_stores * [np.inf, np.inf]), dtype=np.float32)

    def reset(self):
        """
        Resets the environment to the starting conditions
        """
        self.s = (self.cap_store / 2).astype(
            np.int)  # np.zeros(self.n_stores + 1, dtype=int)  # +1 Because the central warehouse is not counted as a store
        #self.s[0] = self.cap_store[0]/2
        self.t = 0
        # Initialize demand and update it directly to avoid jumps in demand of first step
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.update_demand()
        self.demand_old = self.demand.copy()  # np.zeros(self.n_stores, dtype=int)
        # return current state
        return np.hstack((self.s.copy(), self.demand.copy(), self.demand_old.copy()))

    def step(self, action):
        # Update state
        # print('action: ', action)
        action = self.discrete2continuous[action]
        # print('actual action: ', action)
        # Change the space.discrete() into actual action
        upper_bound = self.cap_store - self.s
        # print('upper_bound: ', upper_bound)

        action = np.clip(action, np.zeros(self.n_stores + 1), upper_bound)
        # print('action after clipping: ', action)
        if sum(action[1:]) > self.s[0]:
            action[1:] = action[1:] * self.s[0] / sum(action[1:])
        action = np.around(action, 4)
        action = np.floor(action)
        # print('action after scaling: ', action)

        self.s[0] = min(self.s[0] + action[0] -
                        sum(action[1:]), self.cap_store[0])
        self.s[1:] = np.minimum(
            self.s[1:] - self.demand + action[1:], self.cap_store[1:])

        # Update reward
        reward = (sum(self.demand) * self.price  # revenue
                  - action[0] * self.prod_cost   # production cost
                  - np.sum(np.maximum(np.zeros(self.n_stores + 1),
                                      self.s[:self.n_stores + 1]) * self.store_cost)
                  # Changed to + so that penalty cost actually decrease reward -- Luke 26/02
                  + np.sum(np.minimum(np.zeros(self.n_stores + 1),
                                      self.s[:self.n_stores + 1])) * self.penalty_cost
                  - np.sum(np.ceil(action[1:] / self.cap_truck) * self.truck_cost))
        info = "Demand was: ", self.demand

        # Define state
        # hstack: Stack arrays in sequence horizontally
        state = np.hstack(
            (self.s.copy(), self.demand.copy(), self.demand_old.copy()))

        # Update demand old
        self.demand_old = self.demand.copy()

        # Update t
        self.t += 1

        # Update demand
        self.update_demand()

        # Set if done 0 since unused
        done = 0
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

    def action_space_recur(self):
        '''
        Returns [[a0, a1, ..., ak]]
        '''
        feasible_actions_aux = self.action_space_recur_aux(0, [[]], self.s[0])
        feasible_actions = []
        for action in feasible_actions_aux:
            prod_being_send = sum(action)
            s_0 = self.s[0] - prod_being_send
            for production in np.arange(0, min(self.max_prod, self.cap_store[0] - s_0) + 1):
                feasible_actions.append([production] + action)
        return np.array(feasible_actions)

    def action_space_recur_all(self):
        '''
        Returns [[a0, a1, ..., ak]]
        '''
        feasible_actions_aux = self.action_space_recur_aux_all(0, [[]])
        # print(feasible_actions_aux)
        feasible_actions = []
        for action in feasible_actions_aux:
            for production in np.arange(0, min(self.max_prod, self.cap_store[0]) + 1):
                feasible_actions.append([production] + action)
        return np.array(feasible_actions)

    # prod_left = self.s[0]
    def action_space_recur_aux(self, store_num, current_actions, prod_left):
        '''
        Returns [[a1, ..., ak]]
        '''
        feasible_actions = []
        if store_num == self.n_stores:
            return current_actions
        for prod_being_send in range(0, min(prod_left, self.cap_store[store_num + 1] - self.s[store_num + 1]) + 1):
            new_actions = []
            for action in current_actions:
                new_action = action + [prod_being_send]
                new_actions.append(new_action)
            feasible_actions.extend(self.action_space_recur_aux(
                store_num + 1, new_actions, prod_left - prod_being_send))
        return feasible_actions

    def action_space_recur_aux_all(self, store_num, current_actions):
        feasible_actions = []
        if store_num == self.n_stores:
            return current_actions
        for prod_being_send in range(0, self.cap_store[store_num + 1] + 1):
            new_actions = []
            for action in current_actions:
                new_action = action + [prod_being_send]
                new_actions.append(new_action)
            feasible_actions.extend(
                self.action_space_recur_aux_all(store_num + 1, new_actions))
        return feasible_actions

    def action_space_recur2(self):
        actions = np.array([])
        for i in range(self.n_stores + 1):
            if i == 1:
                action = np.array
        pass

    # def action_space(self):
    #     # return self.action_space_recur()  # this function returns the feasible action space at current state
    #     return self.action_space_recur_all()  # this function returns the feasible action space at current state

    def action_dim(self):
        '''
        return the dimention of actions
        '''
        # return self.action_space_recur_all().shape[0]
        return len(self.discrete2continuous)

    def action_size(self):
        return self.discrete2continuous().shape[0]

    def observation_dim(self):
        '''
        return the dimention of observation(states)
        '''
        return 1 + 3 * self.n_stores

    def possible_action(self):
        allowed_action = []
        i = 0
        for action in self.discrete2continuous:
            i = i + 1
            # print(i)
            if sum(action[1:]) > self.s[0]:
                allowed_action.append(False)
                print(False)
                continue
            if self.s[0] + action[0] - sum(action[1:]) > self.cap_store[0]:
                allowed_action.append(False)
                print(False)
                continue
            for i in range(1, len(action)):
                if self.cap_store[i] - self.s[i] < action[i]:
                    allowed_action.append(False)
                    print(False)
                    continue
            allowed_action.append(True)
            print(True)
        return allowed_action

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
