#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import pandas as pd

from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


class FinancialInteractionData:
    """
    Class for loading and storing the interactions between customers and financial assets.
    We assume that the format has the following format:
    customer_id, asset_id, rating (binary), timestamp (YYYY-mm-dd format)
    """
    def __init__(self, file_name, repeated=False):
        """
        Initialize the financial interaction dataset.
        :param file_name: location of the file containing the interactions.
        """
        # Initialize the financial interaction dataset.
        self.file_name = file_name
        self.data = None
        self.repeated = repeated

    def load(self):
        """
        Loads the interactions from the raw file into memory. It stores the data into
        a dataframe containing the user-item interactions.
        """
        data = pd.read_csv(self.file_name)
        data[DEFAULT_RATING_COL] = 0.0
        data.loc[data["transactionType"] == "Buy", DEFAULT_RATING_COL] = 1.0
        data.rename(columns={"customerID": DEFAULT_USER_COL, "ISIN": DEFAULT_ITEM_COL,
                             "timestamp": DEFAULT_TIMESTAMP_COL}, inplace=True)
        data = data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]].copy()
        data[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(data[DEFAULT_TIMESTAMP_COL], format='%Y-%m-%d')
        self.data = data

    def valid_divide(self, max_train_date, max_valid_date):
        """
        Divides data into training and validation
        :param max_train_date: the maximum training date.
        :param max_valid_date: the maximum validation date.
        :return: a tuple with training and validation data.
        """
        if self.data is None:
            self.load()

        training = self.data[self.data[DEFAULT_TIMESTAMP_COL] < max_train_date]
        valid = self.data[(self.data[DEFAULT_TIMESTAMP_COL] >= max_train_date) &
                          (self.data[DEFAULT_TIMESTAMP_COL] < max_valid_date)]

        # If we do not allow them, we remove the items already consumed during training.
        if self.repeated is False:
            aux_valids = []
            for user in valid[DEFAULT_USER_COL].unique():
                items_user = set(training[training[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                items_valid_user = set(valid[valid[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                diff = items_valid_user - items_user
                if len(diff) > 0:
                    user_valid_df = valid[valid[DEFAULT_USER_COL] == user]
                    user_valid_df = user_valid_df[user_valid_df[DEFAULT_ITEM_COL].isin(diff)]
                    aux_valids.append(user_valid_df)
            valid = pd.concat(aux_valids)

        training['weight'] = training.groupby(by=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]).transform('sum')
        training = training.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
        training[DEFAULT_RATING_COL] = training["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
        training = training.drop(columns=["weight"])

        valid['weight'] = valid.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).transform('sum')
        valid = valid.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
        valid[DEFAULT_RATING_COL] = valid["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
        valid = valid.drop(columns=["weight"])

        return training, valid

    def split(self, min_date, rec_date, max_date):
        """
        Splits the dataset into training and test.
        :param min_date: the minimum date to consider.
        :param rec_date: the recommendation date.
        :param max_date: the maximum date to consider.
        :return: a training, test pair.
        """
        if self.data is None:
            self.load()

        # As a first step, we divide the data
        training = self.data[self.data[DEFAULT_TIMESTAMP_COL].between(min_date, rec_date, inclusive="left")]
        test = self.data[self.data[DEFAULT_TIMESTAMP_COL].between(rec_date, max_date, inclusive="both")]

        # Filter: we only leave those users and items in the training set.
        test = test[test[DEFAULT_USER_COL].isin(training[DEFAULT_USER_COL].unique().flatten())]
        test = test[test[DEFAULT_ITEM_COL].isin(training[DEFAULT_ITEM_COL].unique().flatten())]

        # Then, for each of the remaining users, we remove the items already consumed during the training phase:
        # If we do not allow them, we remove the items already consumed during training.
        if self.repeated is False:
            aux_test = []
            for user in test[DEFAULT_USER_COL].unique():
                items_user = set(training[training[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                items_test_user = set(test[test[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                diff = items_test_user - items_user
                if len(diff) > 0:
                    user_test_df = test[test[DEFAULT_USER_COL] == user]
                    user_test_df = user_test_df[user_test_df[DEFAULT_ITEM_COL].isin(diff)]
                    aux_test.append(user_test_df)
            if len(aux_test) == 0:
                test = pd.DataFrame(columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL])
            else:
                test = pd.concat(aux_test)

        # Remove duplicates
        if training.shape[0] > 0:
            training['weight'] = training.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])[DEFAULT_RATING_COL].transform('sum')
            training = training.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
            training[DEFAULT_RATING_COL] = training["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
            training = training.drop(columns=["weight"])

        # Finally, we remove duplicates:
        if test.shape[0] > 0:
            test['weight'] = test.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL])[DEFAULT_RATING_COL].transform('sum')
            test = test.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
            test[DEFAULT_RATING_COL] = test["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
            test = test.drop(columns=["weight"])

        return training, test

    def divide(self, max_train_date, max_valid_date, max_test_date):
        """
        Splits the dataset in the training, validation and tests in a temporal manner.
        :param max_train_date: maximum date for the training data.
        :param max_valid_date: maximum date for the validation data.
        :param max_test_date: maximum date for the test data.
        :return: a training, validation, test triplet.
        """
        if self.data is None:
            self.load()

        training = self.data[self.data[DEFAULT_TIMESTAMP_COL] < max_train_date]
        valid = self.data[(self.data[DEFAULT_TIMESTAMP_COL] >= max_train_date) &
                          (self.data[DEFAULT_TIMESTAMP_COL] < max_valid_date)]
        test = self.data[(self.data[DEFAULT_TIMESTAMP_COL] >= max_valid_date) &
                         (self.data[DEFAULT_TIMESTAMP_COL] < max_test_date)]

        if self.repeated == False:
            # We remove the items already consumed during training
            aux_valids = []
            for user in valid[DEFAULT_USER_COL].unique():
                items_user = set(training[training[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                items_valid_user = set(valid[valid[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                diff = items_valid_user - items_user
                if len(diff) > 0:
                    user_valid_df = valid[valid[DEFAULT_USER_COL] == user]
                    user_valid_df = user_valid_df[user_valid_df[DEFAULT_ITEM_COL].isin(diff)]
                    aux_valids.append(user_valid_df)
            valid = pd.concat(aux_valids)

            # We remove the items already consumed during training
            aux_test = []
            for user in test[DEFAULT_USER_COL].unique():
                items_user = set(training[training[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                items_valid_user = set(valid[valid[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                items_test_user = set(test[test[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL].unique().flatten())
                diff = items_test_user - (items_user | items_valid_user)
                if len(diff) > 0:
                    user_test_df = test[test[DEFAULT_USER_COL] == user]
                    user_test_df = user_test_df[user_test_df[DEFAULT_ITEM_COL].isin(diff)]
                    aux_test.append(user_test_df)
            test = pd.concat(aux_test)

        training['weight'] = training.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).transform('sum')
        training = training.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
        training[DEFAULT_RATING_COL] = training["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
        training = training.drop(columns=["weight"])

        # Finally, we remove duplicates:
        valid['weight'] = valid.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).transform('sum')
        valid = valid.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
        valid[DEFAULT_RATING_COL] = valid["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
        valid = valid.drop(columns=["weight"])

        # Finally, we remove duplicates:
        test['weight'] = test.groupby([DEFAULT_USER_COL, DEFAULT_ITEM_COL]).transform('sum')
        test = test.drop_duplicates(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep='first')
        test[DEFAULT_RATING_COL] = test["weight"].apply(lambda x: 1.0 if x > 0.0 else 0.0)
        test = test.drop(columns=["weight"])

        return training, valid, test
