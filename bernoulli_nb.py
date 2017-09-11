'''
Copyright 2017 Greg Mogavero

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
'''

import pandas as pd


class BernoulliNB:

    def __init__(self):
        self.prob_table = []
        self.prob_y = None

    def fit(self, X, y):
        self.prob_table = []
        X = pd.DataFrame(X)
        y = pd.Series(y)
        n = len(y)
        y_true = y == 1
        y_false = y == 0
        n_y_true = len(y_true[y_true == True])
        n_y_false = n - n_y_true
        self.prob_y = n_y_true / n
        for f in X.columns.values:
            # Calculate probability of feature given y is true
            f_count = len(X.loc[y_true][X[f] == 1])
            prob_f_given_y = f_count / n_y_true

            # Calculate probability of feature given y is false
            f_count = len(X.loc[y_false][X[f] == 1])
            prob_f_given_not_y = f_count / n_y_false

            '''
            prob_table[i] gives the following probability table for a given feature x with index i:
            
                               y=0          y=1
            P(x|y) = x=0 [[P(x=0|y=0)   P(x=0|y=1)]
                     x=1  [P(x=1|y=0)   P(x=1|y=1)]]
            '''
            self.prob_table.append([[1-prob_f_given_not_y, 1-prob_f_given_y],[prob_f_given_not_y, prob_f_given_y]])

    def predict(self, X):
        X = pd.DataFrame(X)
        pred = []
        for x in X.values:
            prob_y_given_x = self.prob_y
            prob_not_y_given_x = 1 - self.prob_y
            for i in range(len(x)):
                prob_y_given_x *= self.prob_table[i][x[i]][1]
                prob_not_y_given_x *= self.prob_table[i][x[i]][0]
            pred.append(1 if prob_y_given_x > prob_not_y_given_x else 0)
        return pred
