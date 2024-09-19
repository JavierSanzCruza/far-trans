#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod

import pandas as pd
from beta_rec.utils.constants import (
    DEFAULT_USER_COL, )


class Metric(ABC):
    """
    Evaluation metric.
    """

    def __init__(self, data):
        """
        Initializes the metric.
        :param data: the data.
        """
        self.data = data

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        """
        Evaluates the recommendations at a list of cutoffs.
        :param recs: a dataframe containing the recommendations.
        :param cutoffs: the numbers of items to consider.
        :param target_custs: the customers to consider.
        :param only_test_customers: true if we only consider customers with test, false otherwise
        :return: a dictionary, indexed by cutoff, containing a) a dataframe with the customer evaluations and the average
        evaluation.
        """
        aux_cutoffs = [x for x in cutoffs]
        aux_cutoffs.sort(reverse=True)

        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        customers = customers & target_custs

        num_customers = len(customers)

        aux_recs = recs[recs[DEFAULT_USER_COL].isin(customers)]
        aux_recs = aux_recs.groupby(DEFAULT_USER_COL).head(aux_cutoffs[0])


        cust_evals = dict()
        gen_evals = dict()

        for cutoff in cutoffs:
            cust_evals[cutoff] = []
            gen_evals[cutoff] = 0.0

        for customer in customers:
            customer_df = aux_recs[aux_recs[DEFAULT_USER_COL] == customer]
            if customer_df.shape[0] == 0:
                for cutoff in cutoffs:
                    cust_evals[cutoff].append(0.0)
            else:
                aux_dict = self.evaluate_indiv_cutoffs(customer_df, aux_cutoffs)
                for cutoff in cutoffs:
                    cust_evals[cutoff].append((customer, aux_dict[cutoff]))
                    gen_evals[cutoff] += (aux_dict[cutoff])/(num_customers+0.0)

        def_dict = dict()
        for cutoff in cutoffs:
            def_dict[cutoff] = (pd.DataFrame(cust_evals[cutoff], columns=[DEFAULT_USER_COL, "metric"]), gen_evals[cutoff])
        return def_dict

    def evaluate(self, recs, cutoff, target_custs, only_test_customers):
        """
        Evaluates the recommendations at a given cutoff.
        :param recs: a dataframe containing the recommendations.
        :param cutoff: the number of items recommended to each user.
        :param target_custs: the customers to consider.
        :param only_test_customers: true if we only consider customers with test, false otherwise
        :return: A dataframe containing the value of the metric, and the aggregated value over all users.
        """

        customer_eval = []
        aggregated = 0.0

        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) if only_test_customers else self.data.users
        num_customers = len(customers & target_custs)

        for customer in customers & target_custs:
            # Step 1:
            customer_df = recs[recs[DEFAULT_USER_COL] == customer]
            if customer_df.shape[0] == 0:
                customer_eval.append((customer, 0.0))
            else:
                customer_df = customer_df.head(cutoff)
                val = self.evaluate_indiv(customer_df, cutoff)
                customer_eval.append((customer, val))
                aggregated += val/(num_customers + 0.0)
        return pd.DataFrame(customer_eval, columns=[DEFAULT_USER_COL, "metric"]), aggregated

    @abstractmethod
    def evaluate_indiv(self, customer_df, cutoff):
        """
        Evaluates an individual customer.
        :param customer_df: the recommendation for the customer.
        :param cutoff: the cutoff to apply.
        :return: the value of the metric.
        """
        pass

    @abstractmethod
    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        """
        Evaluates an individual customer at different cutoffs.
        :param customer_df: the recommendation for the customer.
        :param cutoffs: the list of cutoffs.
        :return: the value of the metrics.
        """
        pass