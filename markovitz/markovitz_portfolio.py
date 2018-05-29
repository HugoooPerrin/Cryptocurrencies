

import pandas as pd 

import numpy as np

import scipy.optimize as sco

import matplotlib.pyplot as plt

from datetime import datetime

from os import listdir
from os.path import join
from os.path import splitext


class MarkovitzPortfolio(object):


    def __init__(self, dataset=None):
        if dataset is not None:
            self.prices = dataset
            self.quotes = list(self.prices.columns)
            self.quote_number = len(self.quotes)
            

    def save(self, path, name):
        self.prices.to_csv(path + name, index=True)


    def reload(self, path, name):
        self.prices = pd.read_csv(path + name)
        self.prices = self.prices.sort_values('Date', ascending=False).set_index('Date')

        self.quotes = list(self.prices.columns)
        self.quote_number = len(self.quotes)


    def compute_return(self):
        self.returns = np.log(self.prices / self.prices.shift(-1))


    def compute_covariance_matrix(self):
        self.covariance = self.returns.cov() * 252


    def show_statistics(self, weights=None):
        if weights is None:
            weights = np.random.random(self.quote_number)
            weights /= np.sum(weights)
            self.weights = weights.round(3)

        weights = np.array(weights)
        expected_rentability = np.sum(self.returns.mean() * weights) * 252 * 100
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights))) * 100
        sharp_ratio = expected_rentability/expected_volatility

        print("Expected returns: %.2f" % expected_rentability, end=' %\n')
        print("Expected volatility: %.2f" % expected_volatility, end=' %\n')
        print("Sharpe ratio: %.2f" % sharp_ratio)


    def show_distribution(self, weights=None, save=False, path=None, name=None):
        if weights is None:
            self.distribution = pd.DataFrame()
            self.distribution['Quote'] = self.quotes
            self.distribution['Weights'] = self.weights*100
        else:
            self.distribution = pd.DataFrame()
            self.distribution['Quote'] = self.quotes
            self.distribution['Weights'] = weights*100

        if save:
            self.distribution.to_csv(path + name, index=False,  sep=',', float_format='%.1f')

        return self.distribution


    def optimize(self, criterion):

        if criterion == 'sharpe':
            # Function to minimize
            def neg_sharpe_ratio(weights):
                expected_rentability = np.sum(self.returns.mean() * weights) * 252
                expected_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))
                sharp_ratio = expected_rentability/expected_volatility

                return -sharp_ratio

            # Constraint: all weights should sum to 1
            constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1}) 

            # Bounds: all weights belong to [0,1]
            bounds = tuple((0, 1) for x in range(self.quote_number))

            # Initial parameters
            initials = self.quote_number * [1. / self.quote_number,]

            # Running optimization
            self.sharpe_weights = sco.minimize(neg_sharpe_ratio, initials, method='SLSQP', bounds=bounds, constraints=constraint)['x'].round(3)

        elif criterion == 'volatility':
            # Function to minimize
            def variance(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights))) ** 2

            # Constraint: all weights should sum to 1
            constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1}) 

            # Bounds: all weights belong to [0,1]
            bounds = tuple((0, 1) for x in range(self.quote_number))

            # Initial parameters
            initials = self.quote_number * [1. / self.quote_number,]

            # Running optimization
            self.minvar_weights = sco.minimize(variance, initials, method='SLSQP', bounds=bounds, constraints=constraint)['x'].round(3)


    def compute_efficient_frontier(self, rentability):
        """
            Minimization of volatility given a target return
        """
        # Function to minimize
        def volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))

        # Anticipating constraints: 
        # All weights should sum to 1 & return == target return
        def expected_return(weights):
            return np.sum(self.returns.mean() * weights) * 252

        # Bounds: all weights belong to [0,1]
        bounds = tuple((0, 1) for x in range(self.quote_number))

        # Initial parameters
        initials = self.quote_number * [1. / self.quote_number,]

        # Running optimization
        rentabilities = np.linspace(rentability[0], rentability[1], 50)
        min_volatility = []
        for target_return in rentabilities:
            # Updating constraints (since target return has changed)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1},
                           {'type': 'eq', 'fun': lambda x: expected_return(x) - target_return})
            # Getting minimal volatility:
            res = sco.minimize(volatility, initials, method='SLSQP', bounds=bounds, constraints=constraints)
            # Saving result
            min_volatility.append(res['fun'])

        self.rentabilities = rentabilities
        self.min_volatility = np.array(min_volatility)


    def show_efficient_frontier(self):

        def volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))

        def expected_return(weights):
            return np.sum(self.returns.mean() * weights) * 252

        plt.figure(figsize=(16, 8))
        plt.scatter(self.min_volatility, self.rentabilities,
                    c=self.rentabilities/self.min_volatility,
                    marker='o')
        plt.plot(volatility(self.minvar_weights), expected_return(self.minvar_weights),
                 'r*', markersize=15)
        plt.plot(volatility(self.sharpe_weights), expected_return(self.sharpe_weights),
                 'r*', markersize=15)
        plt.grid(True)
        plt.xlabel('Expected volatility')
        plt.ylabel('Expected return')
        plt.title('EFFICIENT FRONTIER')
        plt.colorbar(label='Sharpe ratio')
        plt.show()




