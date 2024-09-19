# FAR-Trans: An Investment Dataset for Financial Asset Recommendation

[![GitHub license](https://img.shields.io/badge/license-MPL--2.0-orange)](https://www.mozilla.org/en-US/MPL/)

This repository contains the code needed to reproduce the experiments of the paper:

> J. Sanz-Cruzado, N. Droukas, R. McCreadie. 2024. [FAR-Trans: An Investment Dataset for Financial Asset Recommendation](https://doi.org/10.48550/arXiv.2407.08692). IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys). Jeju, South Korea, August 2024.

If you use this repository in your research / development, cite the paper.

## Authors

- Javier Sanz-Cruzado, University of Glasgow ([javier.sanz-cruzadopuig@glasgow.ac.uk]())
- Nikolaos Droukas, National Bank of Greece
- Richard McCreadie, University of Glasgow ([richard.mccreadie@glasgow.ac.uk]())

## Installation

This software has been tested in Python 3.9.
We provide two different ways to access the library. First, through manual installation.
Second, through a Docker container image.

### Manual installation

1. Install the library requirements:
```
pandas==1.3.1
numpy
scikit-learn==0.24.2
joblib==1.1.0
ray==1.12.0
inspect-it==0.3.2
gdown==3.13.0
colorama==0.4.4
msgpack==1.0.3
redis==4.1.4
pyyaml==6.0
py-cpuinfo==8.0.0
aiohttp==3.8.1
mlxtend
xgboost==1.5.2
tabulate
lightgbm==3.3.2
cvxopt
psycopg2-binary
```

2. Clone the FAR-Trans library
```
    git clone https://github.com/JavierSanzCruza/far-trans.git 
```

3. In the same directory where FAR-Trans is stored, create a directory named `beta-recsys`
4. Clone the Beta-RecSys library into that directory:
```commandline
    git clone https://github.com/JavierSanzCruza/beta-recsys.git
```
5. From the `beta-recsys` directory, install Beta-RecSys
```commandline
    cd ./beta-recsys
    pip setup.py install 
    pip setup.py 
```

### Docker

Alternatively, we provide a Docker image to use this library. The Docker image initiates a Jupyter notebook server.
```commandline
docker compose up
```

The Jupyter notebook server can be accessed from 'https://localhost:8888' and it will mount on a directory
named 'iPythonNotebooks'.

## Running the code

### Option 1: Scripts
We provide three scripts in the code for running the experiments:

- **Dataset analysis:** Analyzes the asset/customer profitability over time. Command:
  ```commandline
  python run_dataset_analysis.py dataset_directory output_directory
  ```
  where
  - dataset_directory: route of the dataset
  - output_directory: directory in which to store the results.
- **Basic recommendations:** Runs and executes basic, collaborative filtering and profitability prediction models. Command:
  ```commandline
  python run_experiments.py dataset_directory output_directory
  ```
  where
  - dataset_directory: route of the dataset
  - output_directory: directory in which to store the results.
- **Hybrid recommendations:** Runs and executes hybrid models. Command:
  ```commandline
  python run_hybrid_experiments.py dataset_directory output_directory
  ```
  where
  - dataset_directory: route of the dataset
  - output_directory: directory in which to store the results.

### Option 2: Command line programs

Less straightforward, we provide 3 different programs that can be run from command line.

#### Dataset analysis
This program analyzes the dataset (with focus on the asset profitability and volatility)

Command:
```
python dataset_analysis.py interactions prices date_format <date_params> directory summary_file```
```

Input:
- **interactions:** the interactions file.
- **prices:** time series prices.
- **date_format**: two options: `range` (range of dates), `fixed` (fixed dates)
- **date_params**: the parameters for the dates.
  - Parameters for `range`:
    - `min_date`: the initial split date.
    - `max_date`: the final end of the dataset.
    - `num_splits`: the number of splits to make.
    - `num_future`: the number of splits to look ahead in the future
    
    For the experiments in the paper, we performed two experiments on this configuration.
    They ensure dividing the dataset every two weeks, with 6 months of test.
    1. `min_date = 2019-08-01`, `max_date = 2021-02-26`, `num_splits = 28`, `num_future = 13`
    2. `min_date = 2020-09-14`, `max_date = 2022-05-23`, `num_splits = 31`, `num_future = 13`
  - Parameters for `fixed`:
    - `dates_args`: a comma separated list of the split dates
    - `fixed_dates`: a comma separated list of the test dates
    
    Both lists need to be of the same size
- **directory:** the directory in which to store the recommendations, metrics, etc.
- **summary_file:** name of the summary file.

Output:
- A file for every date, named assets_stats_<date>.csv, containing asset information:
  
  Format: `item \t current_price \t future_price \t ROI \t Annualizer ROI \t Monthly ROI \t Volatility`
- A summary file:

  Format: `timestamp \t stat_1 \t stat_2 \t ... stat_N`

#### Customer analysis
This program analyzes the dataset (with focus on the customer profitability)

Command:
```
python customer_analysis.py interactions prices limit_prices date_format <date_params> directory summary_file```
```

Input:
- **interactions:** the interactions file.
- **prices:** time series prices.
- **limit_prices:** file containing the minimum and maximum prices of assets.
- **date_format**: two options: `range` (range of dates), `fixed` (fixed dates)
- **date_params**: the parameters for the dates.
  - Parameters for `range`:
    - `min_date`: the initial split date.
    - `max_date`: the final end of the dataset.
    - `num_splits`: the number of splits to make.
    - `num_future`: the number of splits to look ahead in the future
    
    For the experiments in the paper, we performed two experiments on this configuration.
    They ensure dividing the dataset every two weeks, with 6 months of test.
    1. `min_date = 2019-08-01`, `max_date = 2021-02-26`, `num_splits = 28`, `num_future = 13`
    2. `min_date = 2020-09-14`, `max_date = 2022-05-23`, `num_splits = 31`, `num_future = 13`
  - Parameters for `fixed`:
    - `dates_args`: a comma separated list of the split dates
    - `fixed_dates`: a comma separated list of the test dates
    
    Both lists need to be of the same size
- **directory:** the directory in which to store the recommendations, metrics, etc.
- **summary_file:** name of the summary file.

Output:
- A file for every date, named customer_stats_<date>.csv, containing customer information:
  
  Format: `customer \t buy_price \t sell_price \t ROI \t Annualizer ROI \t Monthly ROI`
- A summary file:

  Format: `timestamp \t stat_1 \t stat_2 \t ... stat_N`


#### Recommendation
Program that executes recommendations and evaluates them.

Command:
```
python recommendation.py interactions prices date_format <date_params> directory delta smoothed kpi_type assets_time compute_metrics repetitions only_test_customers min_prices customer_file
```

Input:
- **interactions:** the interactions file.
- **prices:** time series prices.
- **date_format**: two options: `range` (range of dates), `fixed` (fixed dates)
- **date_params**: the parameters for the dates.
  - Parameters for `range`:
    - `min_date`: the initial split date.
    - `max_date`: the final end of the dataset.
    - `num_splits`: the number of splits to make.
    - `num_future`: the number of splits to look ahead in the future
    
    For the experiments in the paper, we performed two experiments on this configuration.
    They ensure dividing the dataset every two weeks, with 6 months of test.
    1. `min_date = 2019-08-01`, `max_date = 2021-02-26`, `num_splits = 28`, `num_future = 13`
    2. `min_date = 2020-09-14`, `max_date = 2022-05-23`, `num_splits = 31`, `num_future = 13`
  - Parameters for `fixed`:
    - `dates_args`: a comma separated list of the split dates
    - `fixed_dates`: a comma separated list of the test dates
    
    Both lists need to be of the same size
- **directory:** the directory in which to store the recommendations, metrics, etc.
- **months_term:** number of months to look into the future (in our experiments, 6)
- **algorithm_name:** The name of the algorithm. Values:
  - `random`: Random recommendation
  - `pop`: Popularity-based recommendation
  - `arm`: Association-rule mining recommendation
  - `svr`: Support vector regression
  - `rfr`: Random forest regression
  - `lr`: Linear regression
  - `lgbm`: LightGBM regression
  - `lightgcn`: LightGCN collaborative filtering
  - `mf`: Matrix factorization (MF-BPR)
  - `ubknn`: User-based kNN
  - `lambdamart`: Hybrid recommendation
- **algorithm parameters:** The hyperparameters of the algorithm.

Output: for every, date, three files


- File 1: A recommendation file, titled <algorithm>_<date>_recs.csv.

  Format: `user \t item \t rating` (sorted by user in ascending order, rating in descending)
- File 2: An evaluation file, titled "algorithm_date_metrics.csv"

  Format: `metric \t value`
- File 3: A customer evaluation file, where the metrics for every customer are detailed:
  
  Format: `user \t metric1 \t metric2 ... \t metricN`

## Experiment hyperparameters

We list below the hyper-parameters selected for our experiments.

| Algorithm                 | Acronym    | Hyperparameters                                                                |
|---------------------------|------------|--------------------------------------------------------------------------------|
| Random                    | random     | None                                                                           |
| Random forest regression  | rfr        | n = 100                                                                        |
| Linear regression         | lr         | None                                                                           |
| LightGBM                  | lgbm       | Default parameters (see library)                                               |
| Popularity                | pop        | None                                                                           |
| LightGCN                  | lightgcn   | n = 64, lr = 0.01                                                              |
| Association-rule mining   | arm        | None                                                                           |
| Matrix factorization      | mf         | n = 128, loss=bpr                                                              |
| User-based kNN            | ubknn      | similarity = cosine, k = 16, normalized = False                                |
| Hybrid-nDCG               | lambdamart | mode=nDCG, LightGBM default parameters, all other algorithms as features       |
| Hybrid-regression         | lambdamart | mode=regression, LightGBM default parameters, all other algorithms as features |


