# Data Analyzer

A Python library for analyzing the influence of individual data points and features on machine learning models. Currently supporting Sklearn models, with plans to extend support to PyTorch models.

## Installation
Install Data Analyzer using pip:

```bash
pip install data_analyze_tool
```

```python
from DataAnalyzer import AnalyzeModel

# Initialize the model analysis with your model and dataset
myInfluenceModel = AnalyzeModel(my_model, X, y, task="classification", metric="f1")

# Analyze the influence of data
myInfluenceModel.analyze_data_influence()

# Automatically preprocess data
myInfluenceModel.auto_preprocess()
```

Provide the model with your model and data, the model can do analysis in both direction X (individual data influence) by Leave One Out or Shapley Value and direction Y (features by statistic analysis).

The model can auto analyze the features of the model and can provide a pipeline about how to preprocess the model.


## Demo
```python
# Automatically preprocess data
myInfluenceModel.auto_preprocess()
```
```bash
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