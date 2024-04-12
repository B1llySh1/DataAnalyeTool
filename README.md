# Data Analyzer

A Python library for analyzing the influence of individual data points and features on machine learning models with different methods and metrics. Parallel speed up is avaiable for Leave One Out data influence computing. Currently supporting Sklearn models, with plans to extend support to PyTorch models.

## Installation
Install Data Analyzer using pip:

```bash
pip install -i https://test.pypi.org/simple/ DataAnalyzer
```
In your python files

```python
from data_analyze_tool import DataAnalyzer
```
## Usage

Give the model, X data, y data (target data). And choose the task you are performing (Regression, Classification)

### Options: 
Provide your own test sets (X_test, y-test), and set test_set=Ture to enable analysis base on test set, or if test_set not provided the function will automatically split the data into train test base on split (default=0.1)

if n_cores set to > 0, then we can utilze muti threading for computing LOO model retraining. This will make computing LOO data influence on the whole data feasible


### Metrics
Different Metrics are also availbale for different tasks:

Task: [Regression]

Supported Metrics: ["MSE", "MAE", "r2"]

Task: [classification]

Supported Metrics: ["accuracy", "precision", "recall", "f1"]

Task: [probabilities]

Supported Metrics: ["log_loss"]

## Sample

```python
from data_analyze_tool import DataAnalyzer

# Initialize the model analysis with your model and dataset
dataAnalyzer = DataAnalyzer(random_forest_pipeline, X, y, task="classification", test_set=True, metric="f1", n_cores=8)

```

## Supported Functions

```python
# This will analyze the influence of each feature by excluding each of them.
dataAnalyzer.Feature_analyze()

# This automatically analyze each feature and determine the preprocess that should be done to each feature and return them as a pipeline
dataAnalyzer.Auto_preprocess()

# This automatically analyze each feature and determine the preprocess that should be done to each feature and return them as a pipeline
dataAnalyzer.Auto_preprocess()

# This compute the data influence for the data with the given method
dataAnalyzer.CalculateInfluence(method='shapley', num_shuffles=5, threshold=0.98, stat=True)

# This compute the data influence using LOO
# By setting n_random_row = -1, the function will compute all data influences, if n_cores are also set above 1, then it will use multi-thread function
dataAnalyzer.CalculateInfluence(method='LOO', n_random_row=-1)

# This prints the stat of the influences that are previously computed
dataAnalyzer.PrintInfluence()

# After the data influence have been computed, use this function to analyze the negative impact data points
dataAnalyzer.Analyze_data_influence(negative_threshold_percent=0.1)

```

Provide the model with your model and data, the model can do analysis in both direction X (individual data influence) by Leave One Out or Shapley Value and direction Y (features by statistic analysis).

The model can auto analyze the features of the model and can provide a pipeline about how to preprocess the model.


## Demo
```python
# Automatically preprocess data
dataAnalyzer.auto_preprocess()
```
```
Analyzing each features
100%|██████████| 16/16 [00:03<00:00,  4.45it/s]
None of the preprocess works for this column: Gender. Consier removing it or examine it
Trying scaler on column: NCP
This preprocess has influence: -1.8044515586956855e-05
Preprocess dones't work
None of the preprocess works for this column: NCP. Consier removing it or examine it
Trying scaler on column: CAEC
This preprocess has influence: 0.000987806708002581
Performance Improved, saved this preprocess
None of the preprocess works for this column: CAEC. Consier removing it or examine it
Trying scaler on column: SCC
This preprocess has influence: -0.0004951958705641246
Preprocess dones't work
None of the preprocess works for this column: SCC. Consier removing it or examine it
Trying scaler on column: MTRANS
This preprocess has influence: -0.0
Preprocess dones't work
None of the preprocess works for this column: MTRANS. Consier removing it or examine it
Preprocess pipeline: [('scaler_CAEC', PowerTransformer(), ['CAEC'])]
New score 1.0, with improvement 0.000987806708002581
```

```python
# Print the stats of computed data influence
dataAnalyzer.PrintInfluence()
```
```
The data last used: LOO
[ 0.0126183   0.0126183   0.01577287 ...  0.00315457 -0.00315457
 -0.00315457]
Percentage of negative influence data points in data: 2.23%
Average influence: 0.012321126502994571
Most negative influence: -0.009463722397476282 , index: 1241
The data with min influence:
Gender                             1.000000
Age                               19.637947
Height                             1.809101
Weight                            85.000000
family_history_with_overweight     1.000000
FAVC                               1.000000
FCVC                               3.000000
NCP                                3.000000
CAEC                               2.000000
SMOKE                              0.000000
CH2O                               2.229171
SCC                                0.000000
FAF                                1.607953
TUE                                0.628059
CALC                               2.000000
MTRANS                             3.000000
Name: 768, dtype: float64
Most positive influence: 0.04100946372239744 , index: 99
...
TUE                                0.0
CALC                               2.0
MTRANS                             3.0
Name: 411, dtype: float64
```