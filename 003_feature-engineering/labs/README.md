# Notes from Labs

### 1_bqml_basic_feat_eng_bqml-lab.ipynb
[1_bqml_basic_feat_eng_bqml-lab.ipynb](./1_bqml_basic_feat_eng_bqml-lab.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/1_bqml_basic_feat_eng_bqml-lab.ipynb)
### Learnings
1. Create BQ dataset in Bash from Notebook
```
%%bash
# Create a BigQuery dataset for feat_eng if it doesn't exist
datasetexists=$(bq ls -d | grep -w feat_eng)

if [ -n "$datasetexists" ]; then
    echo -e "BigQuery dataset already exists, let's not recreate it."

else
    echo "Creating BigQuery dataset titled: feat_eng"
    
    bq --location=US mk --dataset \
        --description 'Taxi Fare' \
        $PROJECT:feat_eng
   echo "\nHere are your current datasets:"
   bq ls
fi   
```

2. Taking `RMSE` from BQ `Create Model` statement
```
%%bigquery
SELECT
# Here, ML.EVALUATE function is used to evaluate model metrics
*,  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL feat_eng.baseline_model)
```
**NOTE:** Because you performed a linear regression, the results include the following columns:

*   mean_absolute_error
*   mean_squared_error
*   mean_squared_log_error
*   median_absolute_error
*   r2_score
*   explained_variance

**Resource** for an explanation of the [Regression Metrics](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234).

**Mean squared error** (MSE) - Measures the difference between the values our model predicted using the test set and the actual values. You can also think of it as the distance between your regression (best fit) line and the predicted values. 

**Root mean squared error** (RMSE) - The primary evaluation metric for this ML problem is the root mean-squared error. RMSE measures the difference between the predictions of a model, and the observed values. A large RMSE is equivalent to a large average error, so smaller values of RMSE are better. One nice property of RMSE is that the error is given in the units being measured, so you can tell very directly how incorrect the model might be on unseen data.

**R2**:  An important metric in the evaluation results is the R2 score. The R2 score is a statistical measure that determines if the linear regression predictions approximate the actual data. Zero (0) indicates that the model explains none of the variability of the response data around the mean.  One (1) indicates that the model explains all the variability of the response data around the mean.

3. Using `concat` function in BigQuery
```
%%bigquery

CREATE OR REPLACE MODEL
  feat_eng.model_3 OPTIONS (model_type='linear_reg',
    input_label_cols=['fare_amount']) AS
SELECT
  fare_amount,
  passengers,
  concat(cast(extract(hour from pickup_datetime) as string), 
  cast(extract(dayofweek from pickup_datetime) as string)) as combined,
  pickuplon,
  pickuplat,
  dropofflon,
  dropofflat
FROM
  `feat_eng.feateng_training_data`
```
