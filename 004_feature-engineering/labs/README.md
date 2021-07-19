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
### 3_keras_basic_feat_eng.ipynb
[3_keras_basic_feat_eng.ipynb](./3_keras_basic_feat_eng.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/3_keras_basic_feat_eng-lab.ipynb)
1. Let's split the dataset into train, validation, and test sets
```
train, test = train_test_split(housing_df, test_size=0.2)
print(len(train), 'intial train examples with test')
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'final train examples with val')
print(len(val), 'validation examples')
print(len(test), 'test examples')
```
```
train.to_csv('../data/housing-train.csv', encoding='utf-8', index=False)

val.to_csv('../data/housing-val.csv', encoding='utf-8', index=False)

test.to_csv('../data/housing-test.csv', encoding='utf-8', index=False)
```
2. A utility method to create a tf.data dataset from a Pandas Dataframe
```
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('median_house_value')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
```
3. Print features
```
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of households:', feature_batch['households'])
    print('A batch of ocean_proximity:', feature_batch['ocean_proximity'])
    print('A batch of targets:', label_batch)
```
4. create a variable called `numeric_cols` to hold only the numerical feature columns.
```
numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income']
```
5. #### Scaler function
It is very important for numerical variables to get scaled before they are "fed" into the neural network. Here we use min-max scaling. Here we are creating a function named 'get_scal' which takes a list of numerical features and returns a 'minmax' function, which will be used in tf.feature_column.numeric_column() as normalizer_fn in parameters. 'Minmax' function itself takes a 'numerical' number from a particular feature and return scaled value of that number. 

Next, we scale the numerical feature columns that we assigned to the variable "numeric cols".

```
# 'get_scal' function takes a list of numerical features and returns a 'minmax' function
# 'Minmax' function itself takes a 'numerical' number from a particular feature and return scaled value of that number.
# Scalar def get_scal(feature):
def get_scal(feature):
    def minmax(x):
        mini = train[feature].min()
        maxi = train[feature].max()
        return (x - mini)/(maxi-mini)
        return(minmax)
```
```
feature_columns = []
for header in numeric_cols:
    scal_input_fn = get_scal(header)
    feature_columns.append(fc.numeric_column(header,
                                             normalizer_fn=scal_input_fn))
```
Next, we should validate the total number of feature columns. Compare this number to the number of numeric features you input earlier.
```
print('Total number of feature coLumns: ', len(feature_columns))
```
6. Create kera sequential model
```
# Model create
# `tf.keras.layers.DenseFeatures()` is a layer that produces a dense Tensor based on given feature_columns.
feature_layer = tf.keras.layers.DenseFeatures(feature_columns, dtype='float64')

# `tf.keras.Sequential()` groups a linear stack of layers into a tf.keras.Model.
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(12, input_dim=8, activation='relu'),
  layers.Dense(8, activation='relu'),
  layers.Dense(1, activation='linear',  name='median_house_value')
])

# Model compile
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

# Model Fit
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=32)
```
7. We show loss as Mean Square Error (MSE). Remember that MSE is the most commonly used regression loss function. MSE is the sum of squared distances between our target variable (e.g. housing median age) and predicted values.
```
# Let's show loss as Mean Square Error (MSE)
loss, mse = model.evaluate(train_ds)
print("Mean Squared Error", mse)
```
8. #### Visualize the model loss curve

Next, we will use matplotlib to draw the model's loss curves for training and validation.  A line plot is also created showing the mean squared error loss over the training epochs for both the train (blue) and test (orange) sets.
```
# Use matplotlib to draw the model's loss curves for training and validation
def plot_curves(history, metrics):
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(10, 5))

    for idx, key in enumerate(metrics):  
        ax = fig.add_subplot(nrows, ncols, idx+1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left');  
```
```
plot_curves(history, ['loss', 'mse'])
```
9. Test model prediction
```
# Ocean_proximity is NEAR OCEAN
model.predict({
    'longitude': tf.convert_to_tensor([-122.43]),
    'latitude': tf.convert_to_tensor([37.63]),
    'housing_median_age': tf.convert_to_tensor([34.0]),
    'total_rooms': tf.convert_to_tensor([4135.0]),
    'total_bedrooms': tf.convert_to_tensor([687.0]),
    'population': tf.convert_to_tensor([2154.0]),
    'households': tf.convert_to_tensor([742.0]),
    'median_income': tf.convert_to_tensor([4.9732]),
    'ocean_proximity': tf.convert_to_tensor(['NEAR OCEAN'])
}, steps=1)
```
10. Next, we create a categorical feature using `ocean_proximity`
```
for feature_name in categorical_cols:
    vocabulary = housing_df[feature_name].unique()
    categorical_c = fc.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    one_hot = fc.indicator_column(categorical_c)
    feature_columns.append(one_hot)
```
11. Next we create a bucketized column using `housing_median_age`
```
age = fc.numeric_column("housing_median_age")

# Bucketized cols
age_buckets = fc.bucketized_column(age, boundaries=[10, 20, 30, 40, 50, 60, 80, 100])
feature_columns.append(age_buckets)
```
12. ### Feature Cross

Combining features into a single feature, better known as [feature crosses](https://developers.google.com/machine-learning/glossary/#feature_cross), enables a model to learn separate weights for each combination of features.

Next, we create a feature cross of `housing_median_age` and `ocean_proximity`
```
vocabulary = housing_df['ocean_proximity'].unique()
ocean_proximity = fc.categorical_column_with_vocabulary_list('ocean_proximity',
                                                             vocabulary)

crossed_feature = fc.crossed_column([age_buckets, ocean_proximity],
                                    hash_bucket_size=1000)
crossed_feature = fc.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
```
### grepc.py
[grepc.py](./grepc.py)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/data_analysis/lab2/python/grepc.py)
### Learnings
1. Setting bucket name in cloudshell
```
BUCKET="<your unique bucket name (Project ID)>"
echo $BUCKET
```
2. Using `nano` to check out file
```
cd ~/training-data-analyst/courses/data_analysis/lab2/python
nano grep.py
```
### is_popular.py
[is_popular.py](./is_popular.py)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/data_analysis/lab2/python/is_popular.py)
### Learnings
1. Examine the output file with `cat`
```
cat /tmp/output-*
```
### Dataprep Lab
Scheduling and sampling arrive for Google Cloud Dataprep blog/instructions found [here](https://cloud.google.com/blog/products/gcp/scheduling-and-sampling-arrive-for-google-cloud-dataprep)

Lab quiz results [here](../quizzes/06_preprocessing_with_cloud_dataprep)

### Feature Hashing Scheme
Code gathered from this [article](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63). Illustrates how to encode categorical variables into dummy variables with as less columns as possible
```
unique_genres = np.unique(vg_df[['Genre']])
print("Total game genres:", len(unique_genres))
print(unique_genres)
```
```
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=6, input_type='string')
hashed_features = fh.fit_transform(vg_df['Genre'])
hashed_features = hashed_features.toarray()
pd.concat([vg_df[['Name', 'Genre']], pd.DataFrame(hashed_features)], 
          axis=1).iloc[1:7]
```
### Binarisation on Numberic Columns
Code gathered from this [article](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b). Illustrates simple code from sklearn to do binary classifications on a numberic column as opposed to doing a `ifelse` statement
```
watched = np.array(popsong_df['listen_count']) 
watched[watched >= 1] = 1
popsong_df['watched'] = watched
```
```
from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold=0.9)
pd_watched = bn.transform([popsong_df['listen_count']])[0]
popsong_df['pd_watched'] = pd_watched
popsong_df.head(11)
```
#### Taking Quantiles
```
quantile_list = [0, .25, .5, .75, 1.]
quantiles = fcc_survey_df['Income'].quantile(quantile_list)
quantiles
```
#### Quantile based binning
```
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
fcc_survey_df['Income_quantile_range'] = pd.qcut(
                                            fcc_survey_df['Income'], 
                                            q=quantile_list)
fcc_survey_df['Income_quantile_label'] = pd.qcut(
                                            fcc_survey_df['Income'], 
                                            q=quantile_list,       
                                            labels=quantile_labels)

fcc_survey_df[['ID.x', 'Age', 'Income', 'Income_quantile_range', 
               'Income_quantile_label']].iloc[4:9]
```
#### BOC-COx
Another technique to make things normally distributed

### feateng.ipynb and model.py

[feateng.ipynb](./feateng.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/feateng/feateng.ipynb)

[model.py](./model.py)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/feateng/taxifare/trainer/model.py)

### Learnings
1. Add subsampling criteria by modding with hashkey in BigQuery
```
def create_query(phase, EVERY_N):
  if EVERY_N == None:
    EVERY_N = 4 #use full dataset
    
  #select and pre-process fields
  base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  DAYOFWEEK(pickup_datetime) AS dayofweek,
  HOUR(pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
  CONCAT(STRING(pickup_datetime), STRING(pickup_longitude), STRING(pickup_latitude), STRING(dropoff_latitude), STRING(dropoff_longitude)) AS key
FROM
  [nyc-tlc:yellow.trips]
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """
  
  #add subsampling criteria by modding with hashkey
  if phase == 'train': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} < 2".format(base_query,EVERY_N)
  elif phase == 'valid': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 2".format(base_query,EVERY_N)
  elif phase == 'test':
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 3".format(base_query,EVERY_N)
  return query
    
print(create_query('valid', 100)) #example query using 1% of data
```
2. How to head files in bash
```
%%bash
#print first 10 lines of first shard of train.csv
gsutil cat "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-*" | head
```

### 2_bqml_adv_feat_eng-lab.ipynb

[2_bqml_adv_feat_eng-lab.ipynb](./2_bqml_adv_feat_eng-lab.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/feature_engineering/labs/2_bqml_adv_feat_eng-lab.ipynb)

1. Create a BigQuery dataset in `bash`
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
   echo "\n Here are your current datasets:"
   bq ls
fi
```
2. **NOTE:** Because you performed a linear regression, the results include the following columns:

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

3. ### Model 4:  Apply the ML.FEATURE_CROSS clause to categorical features

BigQuery ML now has ML.FEATURE_CROSS, a pre-processing clause that performs a feature cross.  

* ML.FEATURE_CROSS generates a [STRUCT](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#struct-type) feature with all combinations of crossed categorical features, except for 1-degree items (the original features) and self-crossing items.  

* Syntax:  ML.FEATURE_CROSS(STRUCT(features), degree)

* The feature parameter is a categorical features separated by comma to be crossed. The maximum number of input features is 10. An unnamed feature is not allowed in features. Duplicates are not allowed in features.

* Degree(optional): The highest degree of all combinations. Degree should be in the range of [1, 4]. Default to 2.

Output: The function outputs a STRUCT of all combinations except for 1-degree items (the original features) and self-crossing items, with field names as concatenation of original feature names and values as the concatenation of the column string values.

```
%%bigquery

CREATE OR REPLACE MODEL feat_eng.model_4
OPTIONS
  (model_type='linear_reg',
    input_label_cols=['fare_amount'])  
AS
SELECT
  fare_amount,
  passengers,
 ML.FEATURE_CROSS(STRUCT(CAST(EXTRACT(DAYOFWEEK FROM pickup_datetime) AS STRING) AS dayofweek,
  CAST(EXTRACT(HOUR FROM pickup_datetime) AS STRING) AS hourofday)) AS day_hr,
  pickuplon,
  pickuplat,
  dropofflon,
  dropofflat
FROM `feat_eng.feateng_training_data`
```
```
%%bigquery
SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL feat_eng.model_4)
```

4. Because the lat and lon by themselves don't have meaning, but only in conjunction, it may be useful to treat the fields as a pair instead of just using them as numeric values. However, lat and lon are continuous numbers, so we have to discretize them first. That's what SnapToGrid does. 


* ST_SNAPTOGRID:  ST_SNAPTOGRID(geography_expression, grid_size).  Returns the input GEOGRAPHY, where each vertex has been snapped to a longitude/latitude grid. The grid size is determined by the grid_size parameter which is given in degrees.

**REMINDER**: The ST_GEOGPOINT creates a GEOGRAPHY with a single point. ST_GEOGPOINT creates a point from the specified FLOAT64 longitude and latitude parameters and returns that point in a GEOGRAPHY value.  The ST_Distance function returns the minimum distance between two spatial objects.  It also returns meters for geographies and SRID units for geometrics.  

5. ### BQML's Pre-processing functions:

Here are some of the preprocessing functions in BigQuery ML:
* ML.FEATURE_CROSS(STRUCT(features))    does a feature cross of all the combinations
* ML.POLYNOMIAL_EXPAND(STRUCT(features), degree)    creates x, x<sup>2</sup>, x<sup>3</sup>, etc.
* ML.BUCKETIZE(f, split_points)   where split_points is an array 

### Model 7:  Apply the BUCKETIZE Function 


#### BUCKETIZE 
Bucketize is a pre-processing function that creates "buckets" (e.g bins) - e.g. it bucketizes a continuous numerical feature into a string feature with bucket names as the value.

* ML.BUCKETIZE(feature, split_points)

* feature: A numerical column.

* split_points: Array of numerical points to split the continuous values in feature into buckets. With n split points (s1, s2 … sn), there will be n+1 buckets generated. 

* Output: The function outputs a STRING for each row, which is the bucket name. bucket_name is in the format of bin_<bucket_number>, where bucket_number starts from 1.

* Currently, our model uses the ST_GeogPoint function to derive the pickup and dropoff feature.  In this lab, we use the BUCKETIZE function to create the pickup and dropoff feature.

```
%%bigquery

CREATE OR REPLACE MODEL
  feat_eng.model_7 OPTIONS (model_type='linear_reg',
    input_label_cols=['fare_amount']) AS
SELECT
  fare_amount,
  passengers,
  ST_Distance(ST_GeogPoint(pickuplon,
      pickuplat),
    ST_GeogPoint(dropofflon,
      dropofflat)) AS euclidean,
  ML.FEATURE_CROSS(STRUCT(CAST(EXTRACT(DAYOFWEEK
        FROM
          pickup_datetime) AS STRING) AS dayofweek,
      CAST(EXTRACT(HOUR
        FROM
          pickup_datetime) AS STRING) AS hourofday)) AS day_hr,
  CONCAT( 
      ML.BUCKETIZE(pickuplon, GENERATE_ARRAY(-78, -70, 0.01)), 
      ML.BUCKETIZE(pickuplat, GENERATE_ARRAY(37, 45, 0.01)), 
      ML.BUCKETIZE(dropofflon, GENERATE_ARRAY(-78, -70, 0.01)), 
      ML.BUCKETIZE(dropofflat, GENERATE_ARRAY(37, 45, 0.01)) 
  ) AS pickup_and_dropoff
FROM
  `feat_eng.feateng_training_data`
```
```
%%bigquery
SELECT
  *
FROM
  ML.EVALUATE(MODEL feat_eng.model_7)
```
6. Apply the TRANSFORM clause and L2 Regularization

##### [L2 Regularization](https://developers.google.com/machine-learning/glossary/#L2_regularization) 
Sometimes, the training RMSE is quite reasonable, but the evaluation RMSE illustrate more error. Given the severity of the delta between the EVALUATION RMSE and the TRAINING RMSE, it may be an indication of overfitting. When we do feature crosses, we run into the risk of overfitting (for example, when a particular day-hour combo doesn't have enough taxi rides).

Overfitting is a phenomenon that occurs when a machine learning or statistics model is tailored to a particular dataset and is unable to generalize to other datasets. This usually happens in complex models, like deep neural networks.  Regularization is a process of introducing additional information in order to prevent overfitting.

Therefore, we will apply L2 Regularization to the final model.  As a reminder, a regression model that uses the L1 regularization technique is called Lasso Regression while a regression model that uses the L2 Regularization technique is called Ridge Regression.  **The key difference between these two is the penalty term.  Lasso shrinks the less important feature’s coefficient to zero, thus removing some features altogether.  Ridge regression adds “squared magnitude” of coefficient as a penalty term to the loss function**

In other words, L1 limits the size of the coefficients. L1 can yield sparse models (i.e. models with few coefficients); Some coefficients can become zero and eliminated. 

L2 regularization adds an L2 penalty equal to the square of the magnitude of coefficients. L2 will not yield sparse models and all coefficients are shrunk by the same factor (none are eliminated). 

The regularization terms are ‘constraints’ by which an optimization algorithm must ‘adhere to’ when minimizing the loss function, apart from having to minimize the error between the true y and the predicted ŷ.  This in turn reduces model complexity, making our model simpler. A simpler model can reduce the chances of overfitting.
```
(input_label_cols=['fare_amount'],
    model_type='linear_reg',
    l2_reg=0.1) AS
SELECT
  *
FROM
  feat_eng.feateng_training_data
```

7.  Apply the ML.PREDICT function
```
%%bigquery

SELECT
  *
FROM
  ML.PREDICT(MODEL feat_eng.final_model,
    (
    SELECT
      -73.982683 AS pickuplon,
      40.742104 AS pickuplat,
      -73.983766 AS dropofflon,
      40.755174 AS dropofflat,
      3.0 AS passengers,
      TIMESTAMP('2019-06-03 04:21:29.769443 UTC') AS pickup_datetime ))
```
8. Create a RMSE summary table:
| Model       | Taxi Fare | Description                           |
|-------------|-----------|---------------------------------------|
| model_4     | 9.65      | --Feature cross categorical features  |
| model_5     | 5.58      | --Create a Euclidian feature column   |
| model_6     | 5.90      | --Feature cross Geo-location features |
| model_7     | 6.23      | --Apply the TRANSFORM Clause          |
| final_model | 5.39      | --Apply L2 Regularization             |

9. Visualize a RMSE bar chart
```
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

x = ['m4', 'm5', 'm6','m7', 'final']
RMSE = [9.65,5.58,5.90,6.23,5.39]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, RMSE, color='green')
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE Model Summary")

plt.xticks(x_pos, x)

plt.show()
```