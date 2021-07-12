# Notes on Workboks
### improve_data_quality
[improve_data_quality](./improve_data_quality.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/improve_data_quality.ipynb)
### Learnings
1. Changing ownership of cloned repo to user
```
!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst`  
```
2. Creating folder directoy in notebook for data in cell
```
if not os.path.isdir("../data/transport"):
    os.makedirs("../data/transport")
```
3. Using gsutil to copy data from storage into Step 2 (in cell)
```
!gsutil cp gs://cloud-training-demos/feat_eng/transport/untidy_vehicle_data.csv ../data/transport``
```
4. Missing values print statement
```
print ("Rows     : " ,df_transport.shape[0])
print ("Columns  : " ,df_transport.shape[1])
print ("\nFeatures : \n" ,df_transport.columns.tolist())
print ("\nUnique values :  \n",df_transport.nunique())
print ("\nMissing values :  ", df_transport.isnull().sum().values.sum())
```
5. Filling NA values with most common values. By using `value_counts.index[0]` it has most common ordered at top
```
df_transport = df_transport.apply(lambda x:x.fillna(x.value_counts().index[0]))
```
6. Converting yes/no to 1/0. Any nulls were filled with most common, either yes or no. In this case yes has a higher frequency count
```
df.loc[:,'lightduty'] = df['lightduty'].apply(lambda x: 0 if x=='No' else 1)
df['lightduty'].value_counts(0)
```
7. **Making Dummy Variables Function**. Unsure why drop_first is used in this case (Whether to get k-1 dummies out of k categorical levels by removing the first level)
```
data_dummy = pd.get_dummies(df[['zipcode','modelyear', 'fuel', 'make']], drop_first=True)
```
8. Checking **dates are unique**
```
print ('Unique values of month:',df.month.unique())
print ('Unique values of day:',df.day.unique())
print ('Unique values of year:',df.year.unique())
```
9. Converting **date to sin cos (for fancy temporal graph, yet to understand)**. Here we map each temporal variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using the sin and cos trigonometric functions.
```
df['day_sin'] = np.sin(df.day*(2.*np.pi/31))
df['day_cos'] = np.cos(df.day*(2.*np.pi/31))
df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))
```
### python.BQ_explore_data.ipynb
[python.BQ_explore_data.ipynb](./python.BQ_explore_data.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/python.BQ_explore_data.ipynb)

### Learnings
1. Create a jointplot showing "median_income"  versus "median_house_value".
```
sns.jointplot(data = df_USAhousing, x = 'median_income', y = "median_house_value")
```
2. Multiple small graphs for 1 variable. **Multi-plot grid for plotting conditional relationships**
```
#plt.figure(figsize=(20,20))
g = sns.FacetGrid(df_USAhousing, col="ocean_proximity")
g.map(plt.hist, "households");
```
3. To properly sample the dataset, let's use the HASH of the pickup time and return 1 in 100,000 records -- because there are 1 billion records in the data, we should get back approximately 10,000 records if we do this.
```
%%bigquery trips
SELECT
    FORMAT_TIMESTAMP(
        "%Y-%m-%d %H:%M:%S %Z", pickup_datetime) AS pickup_datetime,
    passenger_count,
    trip_distance
FROM
    `nyc-tlc.yellow.trips`
WHERE
    ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 100000)) = 1
```
3. **Heatmap Correlation Matrix**
```
sns.heatmap(df_USAhousing.corr())
```
4. Graphs for all variables, **Pair Plot**, learnt from this [reading](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
```
sns.pairplot(df)
```
5. **Relationship Plot**, learnt from this [reading](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
```
sns.relplot(x='x_var', y='y_var', hue = 'some_var', data = df)
```
6. **Distribution Plot**, learnt from this [reading](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
```
sns.distplot(df['var']) # bins = 5
```
7. **Box and Whisker**, learnt from this [reading](https://www.youtube.com/watch?v=-o3AxdVcUtQ)
```
sns.catplot(x= 'x_var', kind= 'box', data= df)
```
8. Deleting multiple columns, gathered from [here](https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)
```
del_col_list = ['keywords', 'homepage', 'status', 'tagline', 'original_language']
movies_df = movies_df.drop(del_col_list, axis=1)
```
9. Generic function to parse JSON columns, gathered from [here](https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)
```
# we see that there are columns which are in json format,
# let's flatten these json data into easyily interpretable lists

def parse_col_json(column, key):
    """
    Args:
        column: string
            name of the column to be processed.
        key: string
            name of the dictionary key which needs to be extracted
    """
    for index,i in zip(movies_df.index,movies_df[column].apply(json.loads)):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j][key]))# the key 'name' contains the name of the genre
        movies_df.loc[index,column]=str(list1) 
parse_col_json('genres', 'name')
parse_col_json('spoken_languages', 'name')
parse_col_json('cast', 'name')
parse_col_json('production_countries', 'name')
movies_df.head()
```
10. Compare the minimums and maximums of a particlar column and look at rest of data frame, gathered from [here](https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)
def find_min_max_in(col):
    """
    The function takes in a column and returns the top 5
    and bottom 5 movies dataframe in that column.
    
    args:
        col: string - column name
    return:
        info_df: dataframe - final 5 movies dataframe
    """
    
    top = movies_df[col].idxmax()
    top_df = pd.DataFrame(movies_df.loc[top])
    
    bottom = movies_df[col].idxmin()
    bottom_df = pd.DataFrame(movies_df.loc[bottom])
    
    info_df = pd.concat([top_df, bottom_df], axis=1)
    return info_df
find_min_max_in('budget')

### intro_linear_regression.ipynb
[intro_linear_regression.ipynb](./intro_linear_regression.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/intro_linear_regression.ipynb)

### Learnings
1. Training linearn regression model
```
X = df_USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df_USAhousing['Price']
```
Training and testing split (40% for testing)
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
Creating and training model
```
from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(X, y)
```
Model evaluation
```
# print the intercept
print(lm.intercept_)
# Coefficient print out
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```
Interpreting the coefficients:

    Holding all other features fixed, a 1 unit increase in Avg. Area Income is associated with an *increase of $21.52 *.
    Holding all other features fixed, a 1 unit increase in Avg. Area House Age is associated with an *increase of $164883.28 *.
    Holding all other features fixed, a 1 unit increase in Avg. Area Number of Rooms is associated with an *increase of $122368.67 *.
    Holding all other features fixed, a 1 unit increase in Avg. Area Number of Bedrooms is associated with an *increase of $2233.80 *.
    Holding all other features fixed, a 1 unit increase in Area Population is associated with an *increase of $15.15 *.
Predictions from model
```
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
```
Residual Histogram
```
sns.distplot((y_test-predictions),bins=50);
```
Regression Evaluation Metrics

Here are three common evaluation metrics for regression problems. Comparing these metrics:
- **MAE** is the easiest to understand, because it's the average error.
- **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
- **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
All of these are **loss functions**, because we want to minimize them.
```
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
### intro_logistic_regression.ipynb
[intro_logistic_regression.ipynb](./intro_logistic_regression.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/intro_logistic_regression.ipynb)

### Learnings
1. **Jointplot showing the kde distributions**
```
sns.jointplot(data = ad_data, y = 'Daily Time Spent on Site' , x = 'Age', color= 'red', kind= 'kde')
```
2. Logisitic Regression 
Train and fit a logistic regression model on the training set
```
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```
predict values for the testing data
```
predictions = logmodel.predict(X_test)
```
Create a classification report for the model.
```
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```
### decision_trees_and_random_Forests_in_Python.ipynb
[decision_trees_and_random_Forests_in_Python.ipynb](./decision_trees_and_random_Forests_in_Python.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/decision_trees_and_random_Forests_in_Python.ipynb)

### Learnings
Classification done for decision tree then ensembled into a random foreste
1. Decision Tree
Train/test split
```
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```
Train a single tree
```
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
```
Predict and evaluation
```
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
# Here we will build a text report showing the main classification metrics
print(classification_report(y_test,predictions))
# Now we can compute confusion matrix to evaluate the accuracy of a classification
print(confusion_matrix(y_test,predictions))
```
Tree visualisation
Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:
```
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
features = list(df.columns[1:])
features
```
```
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 
```
2. Compare the decision tree to the random forest
Train and predict
```
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
```
Evaluate
```
# Now we can compute confusion matrix to evaluate the accuracy
print(confusion_matrix(y_test,rfc_pred))
# Finally we will build a text report showing the main metrics
print(classification_report(y_test,rfc_pred))
```

### repeatable_splitting.ipynb
[repeatable_splitting.ipynb](./repeatable_splitting.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/repeatable_splitting.ipynb)

### Learnings
How to do repeatable training, validation and testing split using BigQuery and evaluating loss metrics in python
1. 
```
SELECT
    IF(ABS(MOD(FARM_FINGERPRINT(date), 10)) < 8, 'train', 'eval') AS dataset
```

### explore_data.ipynb
[explore_data.ipynb](./explore_data.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/explore_data.ipynb)

### Learnings
1. To properly sample the dataset, let's use the HASH of the pickup time and return 1 in 100,000 records -- because there are 1 billion records in the data, we should get back approximately 10,000 records if we do this.
```
FROM
    `nyc-tlc.yellow.trips`
WHERE
    ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 100000)) = 1
```
2. Python function preprocess / feature enginnering / cleaning data / quality control. The quality control has removed about 300 rows (11400 - 11101) or about 3% of the data. This seems reasonable.
```
def preprocess(trips_in):
    trips = trips_in.copy(deep=True)
    trips.fare_amount = trips.fare_amount + trips.tolls_amount
    del trips["tolls_amount"]
    del trips["total_amount"]
    del trips["trip_distance"]  # we won't know this in advance!

    qc = np.all([
        trips["pickup_longitude"] > -78,
        trips["pickup_longitude"] < -70,
        trips["dropoff_longitude"] > -78,
        trips["dropoff_longitude"] < -70,
        trips["pickup_latitude"] > 37,
        trips["pickup_latitude"] < 45,
        trips["dropoff_latitude"] > 37,
        trips["dropoff_latitude"] < 45,
        trips["passenger_count"] > 0
    ], axis=0)

    return trips[qc]

tripsqc = preprocess(trips)
tripsqc.describe()
```
3. Shuffle the sampled data then split into training, validation, test set in Python
```
shuffled = tripsqc.sample(frac=1)
trainsize = int(len(shuffled["fare_amount"]) * 0.70)
validsize = int(len(shuffled["fare_amount"]) * 0.15)

df_train = shuffled.iloc[:trainsize, :]
df_valid = shuffled.iloc[trainsize:(trainsize + validsize), :]
df_test = shuffled.iloc[(trainsize + validsize):, :]
```
4. Creating into csvs from training, validation, test dataframes
```
def to_csv(df, filename):
    outdf = df.copy(deep=False)
    outdf.loc[:, "key"] = np.arange(0, len(outdf))  # rownumber as key
    # Reorder columns so that target is first column
    cols = outdf.columns.tolist()
    cols.remove("fare_amount")
    cols.insert(0, "fare_amount")
    print (cols)  # new order of columns
    outdf = outdf[cols]
    outdf.to_csv(filename, header=False, index_label=False, index=False)

to_csv(df_train, "taxi-train.csv")
to_csv(df_valid, "taxi-valid.csv")
to_csv(df_test, "taxi-test.csv")
```
5. My model is going to be to simply divide the mean fare_amount by the mean trip_distance to come up with a rate and use that to predict. Let's compute the RMSE of such a model. **Modelling is only 1 line of code!** see `# only model part!` below
```
def distance_between(lat1, lon1, lat2, lon2):
    # Haversine formula to compute distance "as the crow flies".
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    lon_diff_r = np.radians(lon2 - lon1)
    sin_prod = np.sin(lat1_r) * np.sin(lat2_r)
    cos_prod = np.cos(lat1_r) * np.cos(lat2_r) * np.cos(lon_diff_r)
    minimum = np.minimum(1, sin_prod + cos_prod)
    dist = np.degrees(np.arccos(minimum)) * 60 * 1.515 * 1.609344

    return dist

def estimate_distance(df):
    return distance_between(
        df["pickuplat"], df["pickuplon"], df["dropofflat"], df["dropofflon"])

def compute_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def print_rmse(df, rate, name):
    print ("{1} RMSE = {0}".format(
        compute_rmse(df["fare_amount"], rate * estimate_distance(df)), name))

# TODO 4
FEATURES = ["pickuplon", "pickuplat", "dropofflon", "dropofflat", "passengers"]
TARGET = "fare_amount"
columns = list([TARGET])
columns.append("pickup_datetime")
columns.extend(FEATURES)  # in CSV, target is first column, after the features
columns.append("key")
df_train = pd.read_csv("taxi-train.csv", header=None, names=columns)
df_valid = pd.read_csv("taxi-valid.csv", header=None, names=columns)
df_test = pd.read_csv("taxi-test.csv", header=None, names=columns)
rate = df_train["fare_amount"].mean() / estimate_distance(df_train).mean() # only model part!
print ("Rate = ${0}/km".format(rate))
print_rmse(df_train, rate, "Train")
print_rmse(df_valid, rate, "Valid") 
print_rmse(df_test, rate, "Test") 
```
6. The RMSE depends on the dataset, and for comparison, we have to evaluate on the same dataset each time
```
validation_query = """
SELECT
    (tolls_amount + fare_amount) AS fare_amount,
    pickup_datetime,
    pickup_longitude AS pickuplon,
    pickup_latitude AS pickuplat,
    dropoff_longitude AS dropofflon,
    dropoff_latitude AS dropofflat,
    passenger_count*1.0 AS passengers,
    "unused" AS key
FROM
    `nyc-tlc.yellow.trips`
WHERE
    ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 10000)) = 2
    AND trip_distance > 0
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

client = bigquery.Client()
df_valid = client.query(validation_query).to_dataframe()
print_rmse(df_valid, 2.59988, "Final Validation Set")

Final Validation Set RMSE = 8.135336354024895

The simple distance-based rule gives us a RMSE of $8.14. We have to beat this, of course, but you will find that simple rules of thumb like this can be surprisingly difficult to beat.

Let's be ambitious, though, and make our goal to build ML models that have a RMSE of less than $6 on the test set.

```
