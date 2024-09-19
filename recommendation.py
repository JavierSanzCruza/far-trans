#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import datetime
import datetime as dt
import os
import sys
from multiprocessing import Process
from multiprocessing import Semaphore

import argparse


import numpy as np
import pandas as pd
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR

from algorithms.arm import ARMAlgorithm
from algorithms.beta_recsys_algorithm import BetaRecSysAlgorithm
from algorithms.beta_recsys_algs.LightGCN_Train import LightGCN_train
from algorithms.beta_recsys_algs.MF_Train import MF_train
from algorithms.ubknn import UBkNNAlgorithm
from algorithms.cps import CPSAlgorithm
from algorithms.group.customer_profile_group_builder import FeatType, CustomerProfileGroupBuilder
from algorithms.group.group_popularity import GroupPopularityAlgorithm
from algorithms.group.individual_group_algorithm import IndividualGroupAlgorithm
from algorithms.hybrid_lambdamart import HybridLambdaMART
from algorithms.hybrid_rankagg import HybridRankAggregation
from algorithms.kpi_gen.load_kpi_generator import LoadKPIGenerator
from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator
from algorithms.kpi_popularity import KPIPopularityAlgorithm
from algorithms.popularity import PopularityAlgorithm
from algorithms.profitability_classification import ProfitabilityClassification
from algorithms.profitability_prediction import ProfitabilityPrediction
from algorithms.random import RandomAlgorithm
from algorithms.rankingbased import RankingBasedAlgorithm
from algorithms.read_from_ranking import ReadFromRankingAlgorithm
from data.filter.asset.asset_with_test_price import AssetWithTestPrice
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData
from metrics.kpi_ann_evaluation_metric import AnnualizedKPIEvaluationMetric
from metrics.kpi_evaluation_metric import KPIEvaluationMetric
from metrics.kpi_monthly_evaluation_metric import MonthlyKPIEvaluationMetric
from metrics.pure_ndcg import PureNDCG

pd.options.mode.chained_assignment = None  # default='warn'


class Object(object):
    pass


basic_kpis = ["past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
              "volatility_63d", "volatility_126d", "volatility_189d",
              "avg_price_63d", "avg_price_126d", "avg_price_189d"]
full_kpis = ["past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
             "volatility_63d", "volatility_126d", "volatility_189d",
             "avg_price_63d", "avg_price_126d", "avg_price_189d",
             "sharpe_63d", "sharpe_126d", "sharpe_189d",
             "m_63d", "m_126d", "m_189d",
             "roc_63d", "roc_126d", "roc_189d",
             "MACD", "rsi_14", "dco_22",
             "min_63d", "min_126d", "min_189d",
             "max_63d", "max_126d", "max_189d",
             "exp_mean_63d", "exp_mean_126d", "exp_mean_189d"]
basic_short_kpis = ["past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
                    "volatility_21d", "volatility_63d", "volatility_126d",
                    "avg_price_21d", "avg_price_63d", "avg_price_126d"]
full_short_kpis = ["past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
                   "volatility_21d", "volatility_63d", "volatility_126d",
                   "avg_price_21d", "avg_price_63d", "avg_price_126d",
                   "sharpe_21d", "sharpe_63d", "sharpe_126d",
                   "m_21d", "m_63d", "m_126d",
                   "roc_21d", "roc_63d", "roc_126d",
                   "MACD", "rsi_14", "dco_22",
                   "min_21d", "min_63d", "min_126d",
                   "max_21d", "max_63d", "max_126d",
                   "exp_mean_21d", "exp_mean_63d", "exp_mean_126d"]

# Basic algorithms
RANDOM = "random"
POPULARITY = "pop"
PROFITPOP = "profitpop"
ARM = "arm"
RANKING = "ranking"

# Regression
SVR_NAME = "svr"
RFR = "rfr"
LR = "lr"
LIGHTGBM = "lgbm"

# Classification
SVM = "svm"
RFC = "rfc"

# Collaborative filtering
LIGHTGCN = "lightgcn"
MF = "mf"

# Similarity models
CAS = "ubknn"
CPS = "cps"

# Hybrid
AGGMODEL = "rankagg"
LMARTHYBRID = "lambdamart"

FROMRANKING = "fromranking"

GROUPPOP = "grouppop"


def test(algorithm, eval_metrics, file, recomm_date, customers):
    """
    Function that (a) trains an algorithm, (b) generates recommendations and (c) evaluates an algorithm.
    Recommendations and evaluations are written into text files.
    :param algorithm: the recommendation algorithm to apply.
    :param eval_metrics: the evaluation metrics to apply.
    :param file: the name of the file in which to store the recommendation.
    :param recomm_date: the date of the recommendation.
    :param customers: the set of customers to use.
    """
    if os.path.exists(file):
        return

    timeaa = dt.datetime.now()
    print("Started " + file)

    # 1. Train the algorithm:
    algorithm.train(recomm_date)
    time_elapsed = dt.datetime.now() - timeaa
    print("Algorithm " + file + " trained (" + '{}'.format(time_elapsed) + ")")

    # 2. Generate the recommendations:
    recs = algorithm.recommend(recomm_date, False, True)
    recs = recs.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[False, False])
    recs.to_csv(file + "_recs.txt", index=False)
    time_elapsed = dt.datetime.now() - timea
    print("Generated recommendations for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")

    # 3. Compute the metrics:
    cutoffs = [1, 5, 10, 20, 50, 100, 1000]
    metric_res = dict()
    for metric in eval_metrics:
        print("Started metric " + metric[0] + " for " + file)

        metric_dict = metric[1].evaluate_cutoffs(recs, cutoffs, customers, True)
        for cutoff in cutoffs:
            metric_name = metric[0] + "@" + str(cutoff)
            metric_res[metric_name] = metric_dict[cutoff]
        time_elapsed = dt.datetime.now() - timeaa
        print("Computed metric " + metric[0] + " for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")

    time_elapsed = dt.datetime.now() - timeaa
    print("Metrics computed for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")

    # Output the metrics:
    f = open(file + "_metrics.csv", "w")
    for key, val in metric_res.items():
        f.write(key + "\t" + str(val[1]) + "\n")
    f.close()

    cust_metric_df = None
    # Output the metrics by customer
    for key, val in metric_res.items():
        if cust_metric_df is None:
            cust_metric_df = val[0].rename(columns={"metric" : key})
        else:
            aux_df = val[0].rename(columns={"metric": key})
            cust_metric_df = cust_metric_df.merge(aux_df, on=DEFAULT_USER_COL)
    cust_metric_df.to_csv(file + "_customers.csv", index=False)

    time_elapsed = dt.datetime.now() - timea
    print("Algorithm " + file + " finished (" + '{}'.format(time_elapsed) + ")")


def regressor(regression_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file, num_months):
    """
    Configures and runs regression models (predict future profitability of stocks, and rank them according to that
    prediction).
    :param regression_model: the name of the regression model to use.
    :param param: the parameters of the regression model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    :param num_months: the number of months to look into the future.
    """
    alg_model = None
    full = False
    if regression_model == RFR:
        n = int(param[0])
        full = param[1]
        alg_model = RandomForestRegressor(n_estimators=n)
    elif regression_model == SVR_NAME:
        kernel = param[0]
        epsilon = float(param[1])
        full = param[2]
        alg_model = SVR(kernel=kernel, epsilon=epsilon)
    elif regression_model == LR:
        full = param[0]
        alg_model = LinearRegression()
    elif regression_model == LIGHTGBM:
        full = param[0]
        alg_model = LGBMRegressor()

    if full == "full":
        feats = full_kpis
    elif full == "basic":
        feats = basic_kpis
    elif full == "basic_short":
        feats = basic_short_kpis
    else:
        # if full == "full_short":
        feats = full_short_kpis
    algorithm = ProfitabilityPrediction(alg_model, financial_data, num_months, feats, -1)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def classifier(classification_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file,
               num_months):
    """
    Configures and runs classification models (predict whether assets are going to be profitable, and rank them
    according to their probability of being profitable).
    :param classification_model: the name of the regression model to use.
    :param param: the parameters of the regression model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    :param num_months: the number of months to look into the future.
    """
    alg_model = None
    full = False
    if classification_model == RFC:
        n = int(param[0])
        full = param[1]
        alg_model = RandomForestClassifier(n_estimators=n)
    elif classification_model == SVM:
        kernel = param[0]
        full = param[2]
        alg_model = SVC(kernel=kernel, probability=True)

    if full == "full":
        feats = full_kpis
    elif full == "basic":
        feats = basic_kpis
    elif full == "basic_short":
        feats = basic_short_kpis
    else:
        # if full == "full_short":
        feats = full_short_kpis
    algorithm = ProfitabilityClassification(alg_model, financial_data, num_months, feats)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def collaborative_filtering(cf_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file):
    """
    Configures and runs collaborative filtering Beta-RecSys models.
    :param cf_model: the collaborative filtering model to use.
    :param param: the parameters of the model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    """
    alg_model = None
    if cf_model == LIGHTGCN:
        emb_dim = int(param[0])
        lr = float(param[1])
        backup = param[2]

        config = Object()
        config.emb_dim = emb_dim
        config.lr = lr
        config.max_epoch = 50
        config.max_n_update = 50
        config.save_name = file + ".model"
        config.config_file = "/tmp/beta-recsys/configs/lightgcn_default.json"

        if os.path.exists(os.path.join(backup, config.save_name)):
            config.device = "cpu"
        else:
            config.device = "gpu"

        dataset_name = "nbg_" + recommendation_date.strftime("%Y-%m-%d")

        alg_model = LightGCN_train(config, financial_data, dataset_name, backup)
    elif cf_model == MF:
        emb_dim = int(param[0])
        loss_function = param[1]
        backup = param[2]
        alg_name = "mf_" + str(emb_dim) + "_" + str(loss_function) + "_" + recommendation_date.strftime("%Y-%m-%d")

        config = Object()
        config.emb_dim = emb_dim
        config.loss = loss_function
        config.max_epoch = 50
        config.max_n_update = 50
        config.batch_size = 256
        config.save_name = alg_name + ".model"
        config.config_file = "/tmp/beta-recsys/configs/mf_default.json"
        if os.path.exists(os.path.join(backup, config.save_name)):
            config.device = "cpu"
        else:
            config.device = "gpu"

        dataset_name = "nbg_" + recommendation_date.strftime("%Y-%m-%d")

        financial_data.split.train.to_csv(os.path.join(output_dir, dataset_name + "_train.csv"))
        financial_data.split.test[0].to_csv(os.path.join(dataset_name + "_test.csv"))

        alg_model = MF_train(config, financial_data, dataset_name, backup)

    file_name = os.path.join(output_dir, file)
    algorithm = BetaRecSysAlgorithm(financial_data, alg_model)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def basic(basic_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file, period):
    """
    Configures and runs basic, parameter-less models.
    :param basic_model: the name of the basic model.
    :param param: parameters of the model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    :param period: whether we look short time into the past (1 month), mid time (3 months), or long time (6 months).
    """
    file_name = os.path.join(output_dir, file)

    algorithm = None
    if basic_model == RANDOM:
        algorithm = RandomAlgorithm(financial_data)
    elif basic_model == POPULARITY:
        algorithm = PopularityAlgorithm(financial_data)
    elif basic_model == PROFITPOP:
        if period == "long":
            algorithm = KPIPopularityAlgorithm(financial_data, "past_profitability_126d", 0.0, 1.0)
        elif period == "mid":
            algorithm = KPIPopularityAlgorithm(financial_data, "past_profitability_63d", 0.0, 1.0)
        else:
            algorithm = KPIPopularityAlgorithm(financial_data, "past_profitability_21d", 0.0, 1.0)
    elif basic_model == ARM:
        algorithm = ARMAlgorithm(financial_data)
    elif basic_model == RANKING:
        if period == "long":
            algorithm = RankingBasedAlgorithm(financial_data, "past_profitability_126d")
        elif period == "mid":
            algorithm = RankingBasedAlgorithm(financial_data, "past_profitability_63d")
        else:
            algorithm = RankingBasedAlgorithm(financial_data, "past_profitability_21d")

    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def similarity(similarity_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file):
    """
    Configures and runs similarity-based model.
    :param similarity_model: the name of the similarity-based model.
    :param param: the parameters of the model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    """
    algorithm = None
    if similarity_model == CAS:
        sim = param[0]
        k = int(param[1])
        norm = param[2] == "True"
        algorithm = UBkNNAlgorithm(financial_data, sim, k, norm)
    elif similarity_model == CPS:
        k = int(param[0])
        norm = param[1] == "True"
        profiles = param[2]
        algorithm = CPSAlgorithm(financial_data, profiles, k, norm)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def grouprec(group_model, param, financial_data, recommendation_date, eval_metrics, output_dir, file):
    """
    Configures and runs group-based recommendation models.
    :param group_model: the name of the group-based model.
    :param param: the parameters of the model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    """
    algorithm = None
    if group_model == GROUPPOP:
        user_file = params[0]
        user_feats = pd.read_csv(user_file)
        regular_feats = [x for x in params[1].split(",")]
        categories = []
        for cat in params[2].split(","):
            if cat == "dateyear":
                categories.append(FeatType.DATEYEAR)
            elif cat == "dateday":
                categories.append(FeatType.DATEDAY)
            elif cat == "continuous":
                categories.append(FeatType.CONTINUOUS)
            else:
                categories.append(FeatType.CATEGORY)
        sizes = [int(x) for x in params[3].split(",")]

        train_grp_builder = CustomerProfileGroupBuilder(financial_data, user_feats, regular_feats, categories,
                                                        sizes)
        algorithm = IndividualGroupAlgorithm(financial_data,
                                             GroupPopularityAlgorithm(financial_data, train_grp_builder))
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def hybrid(param, financial_data, recommendation_date, eval_metrics, output_dir, file):
    """
    Configures and runs a hybrid ranking aggregation model.
    :param param: the parameters of the model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file
    """
    alg_files = []
    for par in param:
        name = par + recommendation_date.strftime("%Y-%m-%d") + "_recs.txt"
        alg_files.append(name)

    algorithm = HybridRankAggregation(financial_data, alg_files)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def lmarthybrid(param, financial_data, prev_data, recommendation_date, previous_date, eval_metrics, output_dir, file,
                num_months):
    """
    Configures and runs a hybrid ranking aggregation model.
    :param param: the parameters of the model.
    :param financial_data: the split financial data to use.
    :param prev_data: the split financial data at a previous date
    :param recommendation_date: the recommendation date.
    :param previous_date: the previous date to use.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file
    :param num_months: the number of months to look into the future.
    """
    algorithms_with_params = dict()
    alg_directories = dict()

    remaining = -1
    name_alg = True
    current_alg = None
    alg_directory = None
    alg_params = []

    print(param)
    # As a first step, we get the algorithms:
    mode = param[0]
    for par in param[1:]:
        if name_alg:
            name_alg = False
            current_alg = par
        elif not name_alg and alg_directory is None:
            alg_directory = par
            alg_directories[current_alg] = alg_directory
        elif not name_alg and alg_directory is not None and remaining == -1:
            remaining = int(par)
            if remaining == 0:
                algorithms_with_params[current_alg] = alg_params
                alg_params = []
                name_alg = True
                alg_directory = None
                remaining = -1
        else:  # if not alg_name and remaining > 0, we are retrieving parameters:
            alg_params.append(par)
            remaining -= 1
            if remaining == 0:
                algorithms_with_params[current_alg] = alg_params
                alg_params = []
                name_alg = True
                alg_directory = None
                remaining = -1
    alg_list = dict()
    for algorithm in algorithms_with_params:
        name_alg = get_name(algorithm, algorithms_with_params[algorithm])
        print(name_alg)
        train_alg, test_alg = get_algorithm(algorithm, algorithms_with_params[algorithm], financial_data, prev_data,
                                            num_months, previous_date, recommendation_date)
        alg_list[alg_directories[algorithm] + name_alg] = (train_alg, test_alg)

    algorithm = HybridLambdaMART(financial_data, prev_data, alg_list, previous_date, mode, False, True)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def fromranking(param, financial_data, recommendation_date, eval_metrics, output_dir, file):
    """
    Configure, read and evaluate a recommendation from a previous file.
    :param param: the parameters.
    :param financial_data: the split financial data to use.
    :param prev_data: the split financial data at a previous date
    :param recommendation_date: the recommendation date.
    :param previous_date: the previous date to use.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file
    """
    old_file_name = param[0] + recommendation_date.strftime("%Y-%m-%d") + "_recs.txt"
    print(old_file_name)

    file_name = os.path.join(output_dir, file)

    algorithm = ReadFromRankingAlgorithm(financial_data, old_file_name)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)


def get_algorithm(rec_model, param, train_data, rec_data, num_months, train_date, recommendation_date):
    """
        Given a model name, and its parameters, obtains a version of the model. This method is intended for its
        use in hybrid models -- where a pair of models is necessary: a training model (for training feature generation)
        and a test model (for test feature generation).
        :param recommendation_date: the recommendation date.
        :param train_date: the (previous) training date
        :param num_months: number of months to look into the future.
        :param rec_data: recommendation data split at recommendation_date.
        :param train_data: training split data split at train_date.
        :param rec_model: the name of the model.
        :param param: the parameters of the model.
        :return: a pair of algorithms (the training, and test algorithms).
        """
    train_alg = None
    test_alg = None
    if rec_model == RFR:
        if len(param) >= 2:
            n = int(param[0])
            full = param[1]
            alg_model = RandomForestRegressor(n_estimators=n)
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityPrediction(alg_model, train_data, num_months, feats, -1)
            test_alg = ProfitabilityPrediction(alg_model, rec_data, num_months, feats, -1)
    elif rec_model == SVR_NAME:
        if len(param) >= 3:
            kernel = param[0]
            epsilon = float(param[1])
            full = param[2]
            alg_model = SVR(kernel=kernel, epsilon=epsilon)
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityPrediction(alg_model, train_data, num_months, feats, -1)
            test_alg = ProfitabilityPrediction(alg_model, rec_data, num_months, feats, -1)
    elif rec_model == LR:
        if len(param) >= 1:
            full = param[0]
            alg_model = LinearRegression()
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityPrediction(alg_model, train_data, num_months, feats, -1)
            test_alg = ProfitabilityPrediction(alg_model, rec_data, num_months, feats, -1)
    elif rec_model == LIGHTGBM:
        if len(param) >= 1:
            full = param[0]
            alg_model = LGBMRegressor()
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityPrediction(alg_model, train_data, num_months, feats, -1)
            test_alg = ProfitabilityPrediction(alg_model, rec_data, num_months, feats, -1)
    elif rec_model == RFC:
        if len(param) >= 2:
            n = int(param[0])
            full = param[1]
            alg_model = RandomForestClassifier(n_estimators=n)
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityClassification(alg_model, train_data, num_months, feats)
            test_alg = ProfitabilityClassification(alg_model, train_data, num_months, feats)
    elif rec_model == SVM:
        if len(param) >= 2:
            kernel = param[0]
            full = param[1]
            alg_model = SVC(kernel=kernel, probability=True)
            if full == "full":
                feats = full_kpis
            elif full == "basic":
                feats = basic_kpis
            elif full == "basic_short":
                feats = basic_short_kpis
            else:
                # if full == "full_short":
                feats = full_short_kpis
            train_alg = ProfitabilityClassification(alg_model, train_data, num_months, feats)
            test_alg = ProfitabilityClassification(alg_model, train_data, num_months, feats)
    elif rec_model == LIGHTGCN:
        if len(param) >= 3:
            emb_dim = int(param[0])
            lr = float(param[1])
            backup = param[2]

            config = Object()
            config.emb_dim = emb_dim
            config.lr = lr
            config.max_epoch = 50
            config.max_n_update = 50
            config.config_file = "/tmp/beta-recsys/configs/lightgcn_default.json"

            dataset_name = "nbg_" + train_date.strftime("%Y-%m-%d")
            config.save_name = "lightgcn_train_" + train_date.strftime("%Y-%m-%d") + ".model"
            train_alg = BetaRecSysAlgorithm(train_data, LightGCN_train(config, train_data, dataset_name, backup))

            if os.path.exists(os.path.join(backup, config.save_name)):
                config.device = "cpu"
            else:
                config.device = "gpu"

            dataset_name = "nbg_" + recommendation_date.strftime("%Y-%m-%d")
            config.save_name = "lightgcn_rec_" + recommendation_date.strftime("%Y-%m-%d") + ".model"

            if os.path.exists(os.path.join(backup, config.save_name)):
                config.device = "cpu"
            else:
                config.device = "gpu"

            test_alg = BetaRecSysAlgorithm(rec_data, LightGCN_train(config, rec_data, dataset_name, backup))

    elif rec_model == MF:
        if len(param) >= 3:
            emb_dim = int(param[0])
            loss_function = param[1]
            backup = param[2]
            alg_name = MF + "_" + str(emb_dim) + "_" + str(loss_function)
            config = Object()
            config.emb_dim = emb_dim
            config.loss = loss_function
            config.max_epoch = 50
            config.max_n_update = 50
            config.batch_size = 256
            config.save_name = alg_name + "_train_" + train_date.strftime("%Y-%m-%d") + ".model"
            config.config_file = "/tmp/beta-recsys/configs/mf_default.json"
            if os.path.exists(os.path.join(backup, config.save_name)):
                config.device = "cpu"
            else:
                config.device = "gpu"
            dataset_name = "nbg_" + train_date.strftime("%Y-%m-%d")

            train_alg = BetaRecSysAlgorithm(train_data, MF_train(config, train_data, dataset_name, backup))

            config.save_name = alg_name + "_test_" + train_date.strftime("%Y-%m-%d") + ".model"
            dataset_name = "nbg_" + recommendation_date.strftime("%Y-%m-%d")

            test_alg = BetaRecSysAlgorithm(train_data, MF_train(config, rec_data, dataset_name, backup))
    elif rec_model == RANDOM:
        train_alg = RandomAlgorithm(train_data)
        test_alg = RandomAlgorithm(rec_data)
    elif rec_model == POPULARITY:
        train_alg = PopularityAlgorithm(train_data)
        test_alg = PopularityAlgorithm(rec_data)
    elif rec_model == PROFITPOP:
        train_alg = KPIPopularityAlgorithm(train_data, "past_profitability_126d", 0.0, 1.0)
        test_alg = KPIPopularityAlgorithm(rec_data, "past_profitability_126d", 0.0, 1.0)
    elif rec_model == ARM:
        train_alg = ARMAlgorithm(train_data)
        test_alg = ARMAlgorithm(rec_data)
    elif rec_model == CAS:
        if len(param) >= 3:
            sim = param[0]
            k = int(param[1])
            norm = param[2] == "True"
            train_alg = UBkNNAlgorithm(train_data, sim, k, norm)
            test_alg = UBkNNAlgorithm(rec_data, sim, k, norm)
    elif rec_model == CPS:
        if len(param) >= 3:
            k = int(param[0])
            norm = param[1] == "True"
            profiles = param[2]
            train_alg = CPSAlgorithm(train_data, profiles, k, norm)
            test_alg = CPSAlgorithm(rec_data, profiles, k, norm)
    elif rec_model == AGGMODEL:
        # At least we need to combine two algorithms
        if len(param) >= 2:
            train_alg_files = []
            rec_alg_files = []
            for param in params:
                train_name = param + train_date.strftime("%Y-%m-%d") + "_recs.txt"
                train_alg_files.append(train_name)
                rec_name = param + recommendation_date.strftime("%Y-%m-%d") + "_recs.txt"
                rec_alg_files.append(rec_name)

            train_alg = HybridRankAggregation(train_data, train_alg_files)
            test_alg = HybridRankAggregation(rec_data, rec_alg_files)
    elif rec_model == GROUPPOP:
        # The group recommendation algorithm:
        if len(param) >= 4:
            user_file = param[0]
            user_feats = pd.read_csv(user_file)
            regular_feats = [x for x in param[1].split(",")]
            categories = []
            for cat in param[2].split(","):
                if cat == "dateyear":
                    categories.append(FeatType.DATEYEAR)
                elif cat == "dateday":
                    categories.append(FeatType.DATEDAY)
                elif cat == "continuous":
                    categories.append(FeatType.CONTINUOUS)
                else:
                    categories.append(FeatType.CATEGORY)
            sizes = [int(x) for x in param[3].split(",")]

            train_grp_builder = CustomerProfileGroupBuilder(train_data, user_feats, regular_feats, categories, sizes)
            train_alg = IndividualGroupAlgorithm(train_data, GroupPopularityAlgorithm(train_data, train_grp_builder))

            test_grp_builder = CustomerProfileGroupBuilder(rec_data, user_feats, regular_feats, categories, sizes)
            test_alg = IndividualGroupAlgorithm(train_data, GroupPopularityAlgorithm(train_data, test_grp_builder))

    return train_alg, test_alg


def get_name(rec_model, param):
    """
    Given a model, its parameters and a date, obtains the name of the file
    where the results shall be stored.
    :param rec_model: the name of the model.
    :param param: the parameters of the model.
    :return: the name of the model if everything goes right, None otherwise.
    """
    print("model:" + rec_model)

    algorithm_name = None
    if rec_model == FROMRANKING:
        if len(param) >= 1:
            name = param[0]
            algorithm_name = os.path.basename(name)
    if rec_model == RFR:
        if len(param) >= 2:
            n = int(param[0])
            full = param[1]
            algorithm_name = RFR + "_" + str(n) + "_" + full
    elif rec_model == SVR_NAME:
        if len(param) >= 3:
            kernel = param[0]
            epsilon = float(param[1])
            full = param[2]
            algorithm_name = SVR_NAME + "_" + str(kernel) + "_" + str(epsilon) + "_" + full
    elif rec_model == LR:
        if len(param) >= 1:
            full = param[0]
            algorithm_name = LR + "_" + full
    elif rec_model == LIGHTGBM:
        if len(param) >= 1:
            full = param[0]
            algorithm_name = LIGHTGBM + "_" + full
    elif rec_model == RFC:
        if len(param) >= 2:
            n = int(param[0])
            full = param[1]
            algorithm_name = RFC + "_" + str(n) + "_" + full
            return algorithm_name
    elif rec_model == SVM:
        if len(param) >= 2:
            kernel = param[0]
            full = param[1]
            algorithm_name = SVM + "_" + str(kernel) + "_" + full
    elif rec_model == LIGHTGCN:
        if len(param) >= 3:
            emb_dim = int(param[0])
            lr = float(param[1])
            backup = param[2]
            algorithm_name = LIGHTGCN + "_" + str(emb_dim) + "_" + str(lr)
    elif rec_model == MF:
        if len(param) >= 3:
            emb_dim = int(param[0])
            loss_function = param[1]
            backup = param[2]
            algorithm_name = MF + "_" + str(emb_dim) + "_" + str(loss_function)
    elif rec_model == ARM:
        algorithm_name = ARM
    elif rec_model == RANDOM:
        algorithm_name = RANDOM
    elif rec_model == POPULARITY:
        algorithm_name = POPULARITY
    elif rec_model == PROFITPOP:
        algorithm_name = PROFITPOP
    elif rec_model == RANKING:
        print("model:" + RANKING)
        algorithm_name = RANKING
    elif rec_model == CAS:
        if len(param) >= 3:
            sim = param[0]
            k = int(param[1])
            norm = param[2] == "True"
            algorithm_name = CAS + "_" + str(k) + "_" + sim + "_" + ("norm" if norm else "notnorm")
    elif rec_model == CPS:
        if len(param) >= 3:
            k = int(param[0])
            norm = param[1] == "True"
            profiles = param[2]
            algorithm_name = CPS + "_" + str(k) + "_" + ("norm" if norm else "notnorm")
    elif rec_model == AGGMODEL:
        # At least we need to combine two algorithms
        if len(param) >= 2:
            algorithm_name = AGGMODEL
            for param in params:
                algo = param.split(os.sep)[-1]
                aux_name = algo.split("_")[0]
                algorithm_name += "_" + aux_name
    elif rec_model == LMARTHYBRID:
        # At least we need to combine two algorithms
        algorithm_name = LMARTHYBRID
        name = True
        aux_dir = True
        remaining = -1
        mode = param[0]
        algorithm_name += "_" + mode
        for p in param[1:]:
            if name:
                name = False
                algorithm_name += "_" + p
            elif not name and aux_dir:
                aux_dir = False
            elif not name and not aux_dir and remaining == -1:
                remaining = int(p)
                if remaining == 0:
                    name = True
                    aux_dir = True
                    remaining = -1
            else:  # if not alg_name and remaining > 0, we are retrieving parameters:
                remaining -= 1
                if remaining == 0:
                    name = True
                    aux_dir = True
                    remaining = -1
    elif rec_model == GROUPPOP:
        if len(param) >= 4:
            algorithm_name = GROUPPOP
    return algorithm_name


def compute_profitability(time_series, recommendation_date, evaluation_date, min_values):
    """
    Computes the profitability of assets.
    :param time_series: the time series containing the asset prices.
    :param recommendation_date: the recommendation date (starting date)
    :param evaluation_date: the future date (end date)
    :param min_values: if available, a file containing min values of prices.
    :return: a dataframe containing the (raw) profitability of assets between rec_date and future_date.
    """
    # In this case, it is impossible (as of now) that there is an asset without future date pricing:
    rec_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == recommendation_date]
    future_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == evaluation_date]
    # ndays = (future_date - rec_date).days

    aux_series = rec_series.merge(future_series, on=DEFAULT_ITEM_COL, suffixes=("_present", "_future"))
    aux_series["profitability"] = (aux_series[DEFAULT_RATING_COL + "_future"] - aux_series[
        DEFAULT_RATING_COL + "_present"]) / aux_series[DEFAULT_RATING_COL + "_present"]
    prof_dict = dict()
    for index, row in aux_series.iterrows():
        prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]

    if min_values is not None:
        max_series = rec_series.merge(min_values, on=DEFAULT_ITEM_COL)
        max_series["profitability"] = (max_series["max_price"] - max_series[DEFAULT_RATING_COL]) / max_series[
            DEFAULT_RATING_COL]
        for index, row in max_series.iterrows():
            if row[DEFAULT_ITEM_COL] not in prof_dict:
                prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]
    return prof_dict


def compute_volatility(time_series, recommendation_date, evaluation_date):
    """
   Computes the volatility of assets.
   :param time_series: the time series containing the asset prices.
   :param recommendation_date: the recommendation date (starting date)
   :param evaluation_date: the future date (end date)
   :return: a dataframe containing the (raw) profitability of assets between rec_date and future_date.
   """
    series = time_series[time_series[DEFAULT_TIMESTAMP_COL].between(recommendation_date, evaluation_date)]

    series_asset = dict()
    for asset in series[DEFAULT_ITEM_COL].unique().flatten():
        aux_series = series[series[DEFAULT_ITEM_COL] == asset]
        aux_series["profit"] = (aux_series[DEFAULT_RATING_COL] - aux_series[DEFAULT_RATING_COL].shift(1)) / aux_series[
            DEFAULT_RATING_COL].shift(1)
        aux_series = aux_series.dropna()

        series_asset[asset] = aux_series["profit"].std() * np.sqrt(252)

    return series_asset


def print_error_message():
    """
    Prints an error message in case there is an error with the program execution
    :return: the error message.
    """
    text = "ERROR: Invalid arguments:"
    text += "\n\tInteraction data file: file containing the interaction data."
    text += "\n\tTime series data file: file containing the time series."
    text += "\n\tDate format: the format to read the dates. Two valid options:"
    text += "\n\t\trange: to specify a range of dates. In this case, the following arguments are:"
    text += "\n\t\t\tMin. date: The minimum recommendation date to consider."
    text += "\n\t\t\tMax. date: The maximum recommendation date to consider."
    text += "\n\t\t\tNum. splits: the number of recommendation dates to consider (equally separated)."
    text += "\n\t\t\tNum. future: Number of steps in the future to consider."
    text += "\n\t\tfixed_dates: to specify a list of dates, the following arguments are:"
    text += "\n\t\t\trec_dates: a comma separated list of dates in %Y-%m-%d format."
    text += "\n\t\t\tfuture_dates: a comma separated list of evaluation dates in %Y-%m-%d format."
    text += "\n\tDirectory: the directory in which to store all the data"
    text += "\n\tDelta: how many days to consider before the recommendation date as training data."
    text += "\n\tModel: the recommendation model to consider"
    text += "\n\t"
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="financial_asset_recommendation",
        description="Runs financial asset recommendations and evaluates them.",
        epilog="Developed by University of Glasgow"
    )

    parser.add_argument("interactions", help="Customer-asset transaction data file.")
    parser.add_argument("time_series", help="Asset pricing data file.")
    subparsers = parser.add_subparsers(title='date_format', help='Data choice format.', dest='date_format')

    parser_range = subparsers.add_parser('range', help='Range of dates to use. This mode divides the dataset as '
                                                       'follows:\n'
                                                       '- First, divide the period between min_date and max_date into'
                                                       'num_splits + num_future dates\n'
                                                       '- Second, the first num_split dates are considered the split '
                                                       'dates (everything before them is the training set).\n'
                                                       '- The test set contains the data between the split date and '
                                                       '  num_future dates in the list afterwards.')
    parser_range.add_argument("min_date", help='Date of the first split. Format: %Y-%m-%d')
    parser_range.add_argument("max_date", help='End date of the last test set. Format: %Y-%m-%d')
    parser_range.add_argument("num_splits", help='Number of splits to consider.', type=int)
    parser_range.add_argument("num_future", help='Number of dates to look formward', type=int)

    parser_fixed = subparsers.add_parser('fixed_dates', help='List of fixed dates to use. This mode provides fixed '
                                                             'lists of dates for split and test.')
    parser_fixed.add_argument('split_dates', help='Comma separated list of split dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument('future_dates', help='Comma separated list of test end dates. Date format: %Y-%m-%d')

    parser.add_argument("output_dir", help="directory on which to store the outputs.")
    parser.add_argument("months", help="number of months to look into the future.")
    parser.add_argument("model", help="model identifier", choices=[RANDOM, POPULARITY, RFR, LR, LIGHTGBM,
                                                                   LIGHTGCN, MF, ARM, CAS, LMARTHYBRID])
    parser.add_argument("params", help="model parameters", action="store", nargs="*")

    timea = dt.datetime.now()

    args = parser.parse_args()

    # First, we read the parameters:
    interaction_data_file = args.interactions
    time_series_data_file = args.time_series
    date_format = args.date_format

    p = 0
    dates_args = []
    future_dates_args = []
    num_splits = 0
    num_future = 0
    min_date = ""
    max_date = ""

    if date_format == "range":
        min_date = dt.datetime.strptime(args.min_date, "%Y-%m-%d")
        max_date = dt.datetime.strptime(args.max_date, "%Y-%m-%d")
        num_splits = args.num_splits
        num_future = args.num_future

    elif date_format == "fixed_dates":
        dates_args = args.split_dates.split(",")
        future_dates_args = args.future_dates.split(",")
        num_splits = len(dates_args)
        num_future = len(future_dates_args)
        min_date = min(dates_args)
        max_date = max(future_dates_args)
    else:
        sys.stderr.write(print_error_message())
        exit(-1)

    directory = args.output_dir
    months_term = args.months
    model = args.model

    # If the number of days is 0 for the delta, we choose as minimum date one in the distant past
    # (36525 days is exactly 100 years before the established date)
    delta = dt.timedelta(days=36525)
    # Now, we load the data:
    interaction_data = FinancialInteractionData(interaction_data_file)
    time_series_data = FinancialAssetTimeSeries(time_series_data_file)

    # First, load the data.
    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    timeb = dt.datetime.now() - timea
    print("Dataset loaded (" + '{}'.format(timeb) + ")")

    # Compute the technical indicators
    kpi_file = os.path.join(directory, "kpis.csv")
    kpi_type = "full_short"

    if os.path.exists(kpi_file):
        kpi_gen = LoadKPIGenerator(kpi_file)
    else:
        kpi_gen = MAKPIGenerator(data.time_series.data, 5, kpi_type)
    kpi_gen.compute()
    kpis = kpi_gen.get_kpis()

    if not os.path.exists(kpi_file):
        kpi_gen.print_kpis(kpi_file)

    data.add_kpis(kpis)

    timeb = dt.datetime.now() - timea
    print("Technical indicators computed (" + '{}'.format(timeb) + ")")

    dates = []
    future_dates = []
    # Now, we select the possible dates:
    if date_format == "range":
        print("Num splits:" + str(num_splits) + " Num future: " + str(num_future))
        dates, future_dates = data.get_dates(min_date, max_date, num_splits, num_future)
    else:
        print("Num splits:" + str(num_splits))
        for date in dates_args:
            dates.append(pd.to_datetime(date))
        for date in future_dates_args:
            future_dates.append(pd.to_datetime(date))

    print("Selected dates:")
    for i in range(0, len(dates)):
        print("\t" + str(i) + "Training date: " + str(dates[i]) + "\tFuture date: " + str(future_dates[i]))

    procs = []
    values = []

    def_dates = []
    def_future_dates = []
    def_name = []

    params = args.params

    semaphore = Semaphore(4)

    # We first check the selected model is good.
    f_name = get_name(model, params)
    if f_name is None:
        print("ERROR: Invalid parameters")
        exit(-1)

    # Then, we generate the dates for this.
    for i in range(0, len(dates)):
        if not os.path.exists(os.path.join(directory, f_name)):
            def_dates.append(dates[i])
            def_future_dates.append(future_dates[i])
            def_name.append(f_name)

    for i in range(0, len(def_dates)):
        rec_date = def_dates[i]
        future_date = def_future_dates[i]
        min_split_date = rec_date - delta

        alg_name = def_name[i] + "_" + rec_date.strftime("%Y-%m-%d")
        # We only generate recommendations for those dates on which we have not previously generated
        # the recommendations.
        if os.path.exists(os.path.join(directory, alg_name)):
            print("Skipped " + alg_name + " as it already exists")
            continue

        # Get the corresponding file names:
        splitted_data = data.split(min_split_date, rec_date, future_date,
                                   DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(),
                                              NoFilter(), False, True, False))

        timeb = dt.datetime.now() - timea
        print("Dataset splitted (" + '{}'.format(timeb) + ")")

        # We compute the profitability and volatility.
        profitability_df = compute_profitability(splitted_data.time_series, rec_date, future_date, None)
        volatility_df = compute_volatility(splitted_data.time_series, rec_date, future_date)

        # Define the metrics
        metrics = [
            ("profitability", KPIEvaluationMetric(splitted_data, profitability_df)),
            ("annualized_prof", AnnualizedKPIEvaluationMetric(splitted_data, profitability_df,
                                                              (future_date - rec_date).days)),
            ("monthly_prof", MonthlyKPIEvaluationMetric(splitted_data, profitability_df,
                                                        (future_date - rec_date).days)),
            ("volatility", KPIEvaluationMetric(splitted_data, volatility_df)),
            ("ndcg", PureNDCG(splitted_data))]

        # Now, we choose metrics:
        print("Executing algorithm: " + model + " Start date: " + str(rec_date) + " End date: " + str(future_date))
        # Next: we get the algorithm and the parameters:
        if model == RFR:
            if len(params) < 2:
                sys.stderr.write("ERROR: Invalid arguments for random forest")
                sys.stderr.write("\tn: Number of regression trees.")
                sys.stderr.write("\tfull: whether to use the full set of technical indicators or just three of them.")
                exit(-1)
            proc = Process(target=regressor, args=(model, params, splitted_data, rec_date, metrics, directory, alg_name,
                                                   months_term))
            procs.append(proc)
            proc.start()
        elif model == SVR_NAME:
            if len(params) < 3:
                sys.stderr.write("ERROR: Invalid arguments for SVR")
                sys.stderr.write("\tkernel: The kernel we want to use.")
                sys.stderr.write("\tepsilon: Specifies the tube in which no penalty is associated in the training "
                                 "loss function.")
                sys.stderr.write("\tfull: whether to use the full set of technical indicators or just three of them.")
                exit(-1)
            proc = Process(target=regressor, args=(model, params, splitted_data, rec_date, metrics, directory, alg_name,
                                                   months_term))
            procs.append(proc)
            proc.start()
        elif model == LR or model == LIGHTGBM:
            if len(params) < 1:
                sys.stderr.write("ERROR: Invalid arguments for linear regression")
                sys.stderr.write("\tfull: whether to use the full set of technical indicators or just three of them.")
                exit(-1)
            regressor(model, params, splitted_data, rec_date, metrics, directory, alg_name, months_term)
        elif model == RFC:
            if len(params) < 2:
                sys.stderr.write("ERROR: Invalid arguments for random forest")
                sys.stderr.write("\tn: Number of regression trees.")
                sys.stderr.write("\tfull: whether to use the full set of technical indicators or just three of them.")
                exit(-1)
            proc = Process(target=classifier,
                           args=(model, params, splitted_data, rec_date, metrics, directory, alg_name,
                                 months_term))
            procs.append(proc)
            proc.start()
        elif model == SVM:
            if len(params) < 2:
                sys.stderr.write("ERROR: Invalid arguments for SVR")
                sys.stderr.write("\tkernel: The kernel we want to use.")
                sys.stderr.write("\tfull: whether to use the full set of technical indicators or just three of them.")
                exit(-1)
            proc = Process(target=classifier,
                           args=(model, params, splitted_data, rec_date, metrics, directory, alg_name,
                                 months_term))
            procs.append(proc)
            proc.start()
        elif model == LIGHTGCN:
            if len(params) < 3:
                sys.stderr.write("ERROR: Invalid arguments for LightGCN")
                sys.stderr.write("\temb_dim: dimension of the embeddings.")
                sys.stderr.write("\tlr: learning rate.")
                sys.stderr.write("\tbackup: backup directory.")
                exit(-1)
            collaborative_filtering(model, params, splitted_data, rec_date, metrics, directory, alg_name)
        elif model == MF:
            if len(params) < 3:
                sys.stderr.write("ERROR: Invalid arguments for matrix factorization")
                sys.stderr.write("\temb_dim: dimension of the embeddings.")
                sys.stderr.write("\tloss: loss function.")
                sys.stderr.write("\tbackup: backup directory.")
                exit(-1)
            collaborative_filtering(model, params, splitted_data, rec_date, metrics, directory, alg_name)
        elif model == ARM or model == RANDOM or model == POPULARITY or model == PROFITPOP or model == RANKING:
            proc = Process(target=basic, args=(
                model, params, splitted_data, rec_date, metrics, directory, alg_name, months_term))
            procs.append(proc)
            proc.start()
        elif model == CAS:
            if len(params) < 3:
                sys.stderr.write("ERROR: Invalid arguments for CAS")
                sys.stderr.write("\tsim: similarity function to use.")
                sys.stderr.write("\tk: number of neighbors to consider.")
                sys.stderr.write("\tnorm: the norm to consider")
                exit(-1)
            proc = Process(target=similarity, args=(model, params, splitted_data, rec_date, metrics, directory,
                                                    alg_name))
            procs.append(proc)
            proc.start()
        elif model == CPS:
            if len(params) < 3:
                sys.stderr.write("ERROR: Invalid arguments for CPS")
                sys.stderr.write("\tk: number of neighbors to consider.")
                sys.stderr.write("\tnorm: the norm to consider.")
                sys.stderr.write("\tprofiles: file containing the information about profiles.")
                exit(-1)
            proc = Process(target=similarity, args=(
                model, params, splitted_data, rec_date, metrics, directory, alg_name))
            procs.append(proc)
            proc.start()
        elif model == AGGMODEL:
            if len(params) < 2:
                sys.stderr.write("ERROR: Invalid algorithms for the aggregated hybrid model")
                sys.stderr.write("\talgs: a list of space separated files containing the recommendations to use.")
            proc = Process(target=hybrid, args=(
                params, splitted_data, rec_date, metrics, directory, alg_name))
            procs.append(proc)
            proc.start()
        elif model == LMARTHYBRID:
            prev_date = None
            if rec_date in def_future_dates:
                index = def_future_dates.index(rec_date)
                prev_date = def_dates[index]
            else:
                prev_date = rec_date - datetime.timedelta(days=14 * num_future)

            second_splitted_data = data.split(min_split_date, prev_date, rec_date)
            lmarthybrid(params, splitted_data, second_splitted_data, rec_date, prev_date, metrics, directory, alg_name,
                        months_term)
        elif model == FROMRANKING:
            if len(params) < 1:
                sys.stderr.write("ERROR: Invalid parameters for the model")
                sys.stderr.write("\talg: the ranking to generate the recommendations from")
            fromranking(params, splitted_data, rec_date, metrics, directory, alg_name)
            # proc = Process(target=fromranking, args=(params, splitted_data, rec_date, metrics, directory, alg_name,
            #                                         compute_metrics, repetitions, only_test_customers))
            # procs.append(proc)
            # proc.start()
        elif model == GROUPPOP:
            if len(params) < 4:
                sys.stderr.write("ERROR: Invalid parameters for the model")
                sys.stderr.write("\tuser_file: file containing user features")
                sys.stderr.write("\tfeatures: the features to use, comma separated")
                sys.stderr.write("\ttypes: the feature types, comma separated")
                sys.stderr.write("\trange: the range of the features, comma separated")

            proc = Process(target=grouprec, args=(model, params, splitted_data, rec_date, metrics, directory, alg_name))
            procs.append(proc)
            proc.start()
    if len(procs) > 0:
        for proc in procs:
            proc.join()
