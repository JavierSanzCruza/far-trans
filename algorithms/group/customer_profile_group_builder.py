#  Copyright (c) 2023. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"


import math
from enum import Enum

from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

from algorithms.group.group_builder import GroupBuilder


class FeatType(Enum):
    CONTINUOUS = 0
    CATEGORY = 1
    DATEYEAR = 2
    DATEDAY = 3


class CustomerProfileGroupBuilder(GroupBuilder):

    def __init__(self, data, customer_data, user_feats, types, num_divs):
        """
        Constructor.
        :param data: the full financial data.
        :param customer_data: the information about customers.
        :param user_feats: the features to use.
        :param types: the categorization of the features to use. They can be continuous, a category or a date.
        In case it is a category, we create a group per category. If it is a continuous feature, we divide it
        in groups using equal size ranges. If it is a date, we get the difference with the recommendation date, and
        create bins according to given date ranges (starting at 0)
        :param num_divs: a number must be provided for each feature. If the feature is continuous, this represents
        the number of bins. In case the feature is a date, it represents the length of the date range. In case it is a
        category, the value is ignored.
        """
        super().__init__(data, customer_data)

        self.user_feats = user_feats
        self.types = types
        self.num_divs = num_divs

    def group(self, date):
        # First, we obtain the customer data:
        def_customers = self.customer_data[self.customer_data[DEFAULT_TIMESTAMP_COL] <= date]
        def_customers = def_customers.sort_values(by=DEFAULT_TIMESTAMP_COL, ascending=True)
        def_customers = def_customers.drop_duplicates(subset=[DEFAULT_USER_COL], keep="last")

        # Then, we obtain the customer groups:
        def_groups = []
        for i in range(0, len(self.user_feats)):

            user_feat = self.user_feats[i]
            type = self.types[i]
            num_divs = self.num_divs[i]

            cat_groups = []
            if type == FeatType.CATEGORY:
                for x in def_customers[user_feat].unique():
                    customers_x = {row[DEFAULT_USER_COL] for index, row in def_customers[def_customers[user_feat]==x].iterrows()}
                    cat_groups.append(customers_x)
            elif type == FeatType.CONTINUOUS:
                min_val = def_customers[user_feat].min()
                max_val = def_customers[user_feat].max()
                div = (max_val - min_val)/(num_divs + 0.0)
                for j in range(0, num_divs):
                    min_range = min_val + j*div
                    max_range = min_val + (j+1)*div
                    if j == num_divs - 1:
                        customers_x = {row[DEFAULT_USER_COL] for index, row in def_customers[def_customers[user_feat] >= min_range].iterrows()}
                    else:
                        customers_x = {row[DEFAULT_USER_COL] for index, row in def_customers[def_customers[user_feat].between(min_range, max_range, inclusive="left")].iterrows()}
                    if len(customers_x) > 0:
                        cat_groups.append(customers_x)
            elif type == FeatType.DATEYEAR:
                def_customers["DIFF_DATE"] = def_customers[user_feat].apply(lambda y: math.floor((date - y).days / 365))
                max_date = def_customers["DIFF_DATE"].max()
                max_date = math.ceil(max_date / num_divs)*num_divs

                for j in range(0, max_date):
                    min_range = j*num_divs
                    max_range = (j+1)*num_divs
                    customers_x = {row[DEFAULT_USER_COL] for index, row in def_customers[
                        def_customers["DIFF_DATE"].between(min_range, max_range, inclusive="left")].iterrows()}
                    if len(customers_x) > 0:
                        cat_groups.append(customers_x)

                def_customers = def_customers.drop(columns=["DIFF_DATE"])
            else: ## type == FeatType.DATEDAY:
                def_customers["DIFF_DATE"] = def_customers[user_feat].apply(lambda y: (date - y).days)
                max_date = def_customers["DIFF_DATE"].max()
                max_date = math.ceil(max_date / num_divs) * num_divs

                for j in range(0, max_date):
                    min_range = j * num_divs
                    max_range = (j + 1) * num_divs
                    customers_x = {row[DEFAULT_USER_COL] for index, row in def_customers[
                        def_customers["DIFF_DATE"].between(min_range, max_range, inclusive="left")].iterrows()}
                    if len(customers_x) > 0:
                        cat_groups.append(customers_x)

                def_customers = def_customers.drop(columns=["DIFF_DATE"])
            if len(def_groups) == 0:
                def_groups = cat_groups
            else:
                aux_groups = []
                for group in def_groups:
                    for cat_group in cat_groups:
                        inter = group & cat_group
                        if len(inter) > 0:
                            aux_groups.append(inter)
                def_groups = aux_groups

        cust_groups = dict()
        for i in range(0, len(def_groups)):
            group = def_groups[i]
            for cust in group:
                cust_groups[cust] = i

        return cust_groups, def_groups






