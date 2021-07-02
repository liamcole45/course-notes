# Notebooks
### tensors_variables
[tensors_variables](./tensors_variables.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/tensors-variables.ipynb)
### Learnings
1. Setting constant
```
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```
2. Specifying data types in tensors
```
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype= tf.float16)
print(rank_2_tensor)
```
3. Convert a tensor to a NumPy array using `np.array` method
```
np.array(rank_2_tensor)
```
4. Convert a tensor to a NumPy array using `tensor.numpy`
```
rank_2_tensor.numpy()
```
5. `tf.add`, `tf.multiply` and `tf.matmul`
```
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
```
6. Finding largest value in tensor, index of larges value and computing the softmax
```
# Find the largest value
print(tf.reduce_max(c))
# TODO 1d
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))
```
7. Tensor shapes
```
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```
8. Single axis indexing
```
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
```
9. Multi axis indexing
```
# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())

# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

print(rank_3_tensor[:, :, 4])
```
10. `tf.variable` (and naming it) and `tf.reshape`
```
# Shape returns a `TensorShape` object that shows the size on each dimension
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)

# We can reshape a tensor to a new shape.
reshaped = tf.reshape(var_x, [1, 3])

print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))

# Create a and b; they have the same value but are backed by different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)
```
11. `tf.cast`, used for casting different data types
```
# Use the `Tensor.dtype` property
# You can cast from type to type
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, let's cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```
12. `tf.broadcast_to`, broadcasting tensors. Means stretching smaller ones
```
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)

print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```
13. Ragged tensor, `tf.ragged`
```
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```
14. `tf.strings.split`
```
print(tf.strings.split(tensor_of_strings))
```
15. `tf.sparse.SparseTensor`
```
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],values=[1, 2],
 dense_shape=[3, 4])
print(sparse_tensor, "\n")
```

### write_low_level_code
[write_low_level_code](./write_low_level_code.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/write_low_level_code.ipynb)
1. Modeling linear regression `y = 2x + 10`
```
X = tf.constant(range(10), dtype=tf.float32)
Y = 2 * X + 10

print("X:{}".format(X))
print("Y:{}".format(Y))

X_test = tf.constant(range(10, 20), dtype=tf.float32)
Y_test = 2 * X_test + 10

print("X_test:{}".format(X_test))
print("Y_test:{}".format(Y_test))

y_mean = Y.numpy().mean()


def predict_mean(X):
    y_hat = [y_mean] * len(X)
    return y_hat

Y_hat = predict_mean(X_test)
```
2. Linear regression loss_mse function
```
def loss_mse(X, Y, w0, w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)
```
3. `tf.GradientTable()` to get loss from linear regression
```
def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape: # Record operations for automatic differentiation.
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dw0, dw1 = compute_gradients(X, Y, w0, w1)
print("dw0:", dw0.numpy())
print("dw1", dw1.numpy())

# TODO 3
STEPS = 1000
LEARNING_RATE = .02
MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"


w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)


for step in range(0, STEPS + 1):

    dw0, dw1 = compute_gradients(X, Y, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)

    if step % 100 == 0: # % is modulus. To find out if a year is a leap year or not, you can divide it by four and if the remainder is zero, it is a leap year.
        loss = loss_mse(X, Y, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))

loss = loss_mse(X_test, Y_test, w0, w1)
loss.numpy()
```
4. Modelling non linear function is captured at end of the notebook

### load_diff_filedata.ipynb
[load_diff_filedata](./load_diff_filedata.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/load_diff_filedata.ipynb)
1. Make numpy values easier to read.
```
np.set_printoptions(precision=3, suppress=True)
```
2. Download file from website using `tf.keras.utils.get_file`
```
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
```
3. Head csv using bash
```
!head {train_file_path}
```
4. Get and view dataset using `show_batch`
```
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size = 5, # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value='?',
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

show_batch(raw_train_data)
```
5. Data normalisation. **Continous data should always be normalised**
```
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
print(MEAN,STD)

def normalize_numeric_data(data, mean, std):
    return (data-mean)/std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column
```
6. Dealing with categorical data
```
categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])
```
7. Combined preprocessing layer
```
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
```
### feat.cols_tf.data.ipynb
[feat.cols_tf.data](./feat.cols_tf.data.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/feat.cols_tf.data.ipynb)
1. Split the dataframe into train, validation, and test
```
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataframe, test_size = 0.2)
train, val = train_test_split(train, test_size = 0.2)

print(len(dataframe), 'total')
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
```
2. create a `tf.data` dataset from a Pandas Dataframe
```
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
```
3. `tf.feature_column`
```
age = feature_column.numeric_column("age")
tf.feature_column.numeric_column
print(age)
```
4. Bucketized columns
```
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)
```
5. Categorical columns
```
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = tf.feature_column.indicator_column(thal)
demo(thal_one_hot)
```
6. Embedding columns
```
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)
```
7. Hashed feature columns
```
thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(thal_hashed))
```
8. Crossed feature columns
```
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(crossed_feature))
```
9. Combining all the feature columns
```
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
```
10. Input feature columns to a keras model
```
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
print(len(train_ds), "train_ds")
print(len(val_ds), "val_ds")
print(len(test_ds), "test_ds")
```
11. create, compile and train the model
```
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
```
12. Visualise the model loss curve
```
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
    
    

plot_curves(history, ['loss', 'accuracy'])
```
13. ## Key Point
You will typically see best results with deep learning with much larger and more complex datasets. When working with a small dataset like this one, we recommend using a decision tree or random forest as a strong baseline. The goal of this tutorial is not to train an accurate model, but to demonstrate the mechanics of working with structured data, so you have code to use as a starting point when working with your own datasets in the future.
### tfrecord-tf.example.ipynb
[tfrecord-tf.example.ipynb](./tfrecord-tf.example.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/tfrecord-tf.example.ipynb)
1. `tf.train.Feature`
Fundamentally, a `tf.Example` is a `{"string": tf.train.Feature}` mapping.

The `tf.train.Feature` message type can accept one of the following three types (See the [`.proto` file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) for reference). Most other generic types can be coerced into one of these:

a. `tf.train.BytesList` (the following types can be coerced)

  - `string`
  - `byte`

b. `tf.train.FloatList` (the following types can be coerced)

  - `float` (`float32`)
  - `double` (`float64`)

c. `tf.train.Int64List` (the following types can be coerced)

  - `bool`
  - `enum`
  - `int32`
  - `uint32`
  - `int64`
  - `uint64`
2. `serialize_example`
```
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
```
3. `tf.train.Example.FromString`
```
example_proto = tf.train.Example.FromString(serialized_example)
example_proto
```
4. `tf_serialize_example`
```
 # TODO 2a
def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

  tf_serialize_example(f0,f1,f2,f3)

# `.map` function maps across the elements of the dataset.
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset
```
5. `tf.data.TFRecordDataset`
```
 # TODO 2c
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```
6. Use the `.take` method to pull ten examples from the dataset.
```
for raw_record in raw_dataset.take(10):
  print(repr(raw_record))
```
7. Create a description of the features
```
# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)
```
8. `tf.io.TFRecordWriter`
```
# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())
```
### 2_dataset_api.ipynb
[2_dataset_api.ipynb](./2_dataset_api.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/2_dataset_api.ipynb)

1. how to use tf.data to read data from memory
```
# Lets define the create_dataset() procedure
def create_dataset(X, Y, epochs, batch_size):
    # Using the tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset
```
2. test function by iterating twice over our dataset in batches of 3 datapoints
```
BATCH_SIZE = 3
EPOCH = 2

dataset = create_dataset(X, Y, epochs=EPOCH, batch_size=BATCH_SIZE)

for i, (x, y) in enumerate(dataset):
    # You can convert a native TF tensor to a NumpPy array ising .numpy() method
    # Let's output the value of x & y
    print("x:", x.numpy(), "y:", y.numpy())
    assert len(x) == BATCH_SIZE
    assert len(y) == BATCH_SIZE
```
3. how to use tf.data in a training loop
```
EPOCHS = 250
BATCH_SIZE = 2
LEARNING_RATE = .02

MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dataset = create_dataset(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

for step, (X_batch, Y_batch) in enumerate(dataset):

    dw0, dw1 = compute_gradients(X_batch, Y_batch, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)

    if step % 100 == 0:
        loss = loss_mse(X_batch, Y_batch, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))
        
assert loss < 0.0001
assert abs(w0 - 2) < 0.001
assert abs(w1 - 10) < 0.001
```
4. tf.data to read the CSV files
The first step is to define 

- the feature names into a list `CSV_COLUMNS`
- their default values into a list `DEFAULTS`

```
CSV_COLUMNS = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'key'
]
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], ['na'], [0.0], [0.0], [0.0], [0.0], [0.0], ['na']]
```
5. how to use tf.data to read data from disk
```
def create_dataset(pattern):
    # the tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset
    return tf.data.experimental.make_csv_dataset(pattern, 1, CSV_COLUMNS, DEFAULTS)

tempds = create_dataset('../data/taxi-train*')
print(tempds)
```

Note that this is a prefetched dataset, where each element is an `OrderedDict` whose keys are the feature names and whose values are tensors of shape `(1,)` (i.e. vectors).

Let's iterate over the two first element of this dataset using `dataset.take(2)` and let's convert them ordinary Python dictionary with numpy array as values for more readability:
```
for data in tempds.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")
```
6. Transforming the features
```
UNWANTED_COLS = ['pickup_datetime', 'key']

# Lets define the features_and_labels() method
def features_and_labels(row_data):
    # The .pop() method will return item and drop from frame
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label
```
7. Let's iterate over 2 examples from our `tempds` dataset and apply our `feature_and_labels`
function to each of the examples to make sure it's working:
```
for row_data in tempds.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")
    assert UNWANTED_COLS[0] not in features.keys()
    assert UNWANTED_COLS[1] not in features.keys()
    assert label.shape == [1]
```
8. how to write production input pipelines with feature engineering (batching, shuffling, etc.)
```
# Let's define the create_dataset() method
def create_dataset(pattern, batch_size):
    # the tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    return dataset.map(features_and_labels)
```
9. test that our batches are of the right size
```
BATCH_SIZE = 2

tempds = create_dataset('../data/taxi-train*', batch_size=2)

for X_batch, Y_batch in tempds.take(2):
    pprint({k: v.numpy() for k, v in X_batch.items()})
    print(Y_batch.numpy(), "\n")
    assert len(Y_batch) == BATCH_SIZE
```
10. Shuffling

When training a deep learning model in batches over multiple workers, it is helpful if we shuffle the data. That way, different workers will be working on different parts of the input file at the same time, and so averaging gradients across workers will help. Also, during training, we will need to read the data indefinitely.
```
def create_dataset(pattern, batch_size=1, mode='eval'):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    # The map() function executes a specified function for each item in an interable.
    # The item is sent to the function as a parameter
    
    dataset = dataset.map(features_and_labels).cache()

    if mode == 'train':
        dataset = dataset.shuffle(1000).repeat()

    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    
    return dataset
```
11. Let's check that our function works well in both models
```
tempds = create_dataset('../data/taxi-train*', 2, 'train')
print(list(tempds.take(1)))
```
### adv_tfdv_facets.ipynb
[adv_tfdv_facets](./adv_tfdv_facets.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/adv_tfdv_facets.ipynb)
1. Use TFRecords to load record-oriented binary format data
```
download_original_data = False #@param {type:"boolean"}

# Downloads a file from a URL if it is not already in cache using the `tf.keras.utils.get_file()` function
if download_original_data:
    train_tf_file = tf.keras.utils.get_file('train_tf.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf.tfrecord')
    validate_tf_file = tf.keras.utils.get_file('validate_tf.tfrecord',
                                              'https://storage.googleapis.com/civil_comments_dataset/validate_tf.tfrecord')
    # The identity terms list will be grouped together by their categories
    # (see 'IDENTITY_COLUMNS') on threshould 0.5. Only the identity term column,
    # text column and label column will be kept after processing.
    train_tf_file = util.convert_comments_data(train_tf_file)
    validate_tf_file = util.convert_comments_data(validate_tf_file)
    
else:
    train_tf_file = tf.keras.utils.get_file('train_tf_processed.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf_processed.tfrecord')
    validate_tf_file = tf.keras.utils.get_file('validate_tf_processed.tfrecord',
                                              'https://storage.googleapis.com/civil_comments_dataset/validate_tf_processed.tfrecord')
```
2. `tfdv.generate_statistics_from_tfrecord` to generate statistics and Facets to visualize the data
```
 The computation of statistics using TFDV.  The returned value is a DatasetFeatureStatisticsList protocol buffer. 
stats = tfdv.generate_statistics_from_tfrecord(data_location=train_tf_file)

# A visualization of the statistics using Facets Overview.
tfdv.visualize_statistics(stats)
```
3. Analyze label distribution for subset groups
```
#@title Calculate label distribution for gender-related examples
raw_dataset = tf.data.TFRecordDataset(train_tf_file)

toxic_gender_examples = 0
nontoxic_gender_examples = 0

# There are 1,082,924 examples in the dataset
# The `take()` method returns the specified number of elements starting from the first element
for raw_record in raw_dataset.take(1082924):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    if str(example.features.feature["gender"].bytes_list.value) != "[]":
        if str(example.features.feature["toxicity"].float_list.value) == "[1.0]":
            toxic_gender_examples += 1
        else: nontoxic_gender_examples += 1
            
print("Toxic Gender Examples: %s" % toxic_gender_examples)
print("Nontoxic Gender Examples %s" % nontoxic_gender_examples)
```
### 3_keras_sequential_api.ipynb
[3_keras_sequential_api](./3_keras_sequential_api.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/3_keras_sequential_api.ipynb)
1. change the ownership of the repo to the user
```
!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst
```
2. finding if have permission to datasets
```
# using -l parameter will list the files with assigned permissions
!ls -l ../data/*.csv
```
3. Use tf.data to read the CSV files
```
# feature names into list
CSV_COLUMNS = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'key'
]
LABEL_COLUMN = 'fare_amount'
# listing default values
DEFAULTS = [[0.0], ['na'], [0.0], [0.0], [0.0], [0.0], [0.0], ['na']]
UNWANTED_COLS = ['pickup_datetime', 'key']


def features_and_labels(row_data):
    # the pop method will return item and drop from frame
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label


def create_dataset(pattern, batch_size=1, mode='eval'):
    # the tf.data.experimental.make_csv_dataset() method reads CSV files into dataset
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    # the map() function executes a specified function for each item in an iterable
    # the item is sent to the function as a parameter
    dataset = dataset.map(features_and_labels)

    if mode == 'train':
    # the shuffle() method takes a sequence (list, string, tuple) and re organise the order of the items
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    return dataset
```
4. In our case we won't do any feature engineering. However, we still need to create a list of feature columns to specify the numeric values which will be passed on to our model. To do this, we use `tf.feature_column.numeric_column()`

We use a python dictionary comprehension to create the feature columns for our model, which is just an elegant alternative to a for loop.
```
INPUT_COLS = [
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
]

# Create input layer of feature columns
feature_columns = {
    colname: tf.feature_column.numeric_column(colname)
    for colname in INPUT_COLS
}
```
5. Create a deep neural network using Keras's Sequential API.
```
# Build a keras DNN model using Sequential API
model = Sequential([
    DenseFeatures(feature_columns=feature_columns.values()),
    Dense(units=32, activation='relu', name='h1'),
    Dense(units=8, activation='relu', name='h2'),
    Dense(units=1, activation='relu', name='output')
])
```
6. Create a custom loss function called `rmse` which computes the root mean squared error between `y_true` and `y_pred`. Pass this function to the model as an evaluation metric. 
```
# Create a custom evalution metric
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Compile the keras model
model.compile(optimizer= 'adam', loss= 'mse', metrics= [rmse,'mse'])
```
7. ## Train the model

To train your model, Keras provides three functions that can be used:
 1. `.fit()` for training a model for a fixed number of epochs (iterations on a dataset).
 2. `.fit_generator()` for training a model on data yielded batch-by-batch by a generator
 3. `.train_on_batch()` runs a single gradient update on a single batch of data. 
 
The `.fit()` function works well for small datasets which can fit entirely in memory. However, for large datasets (or if you need to manipulate the training data on the fly via data augmentation, etc) you will need to use `.fit_generator()` instead. The `.train_on_batch()` method is for more fine-grained control over training and accepts only a single batch of data.

The taxifare dataset we sampled is small enough to fit in memory, so can we could use `.fit` to train our model. Our `create_dataset` function above generates batches of training examples, so we could also use `.fit_generator`. In fact, when calling `.fit` the method inspects the data, and if it's a generator (as our dataset is) it will invoke automatically `.fit_generator` for training. 

We start by setting up some parameters for our training job and create the data generators for the training and validation data.

We refer you the the blog post [ML Design Pattern #3: Virtual Epochs](https://medium.com/google-cloud/ml-design-pattern-3-virtual-epochs-f842296de730) for further details on why express the training in terms of `NUM_TRAIN_EXAMPLES` and `NUM_EVALS` and why, in this training code, the number of epochs is really equal to the number of evaluations we perform.
```
TRAIN_BATCH_SIZE = 1000
NUM_TRAIN_EXAMPLES = 10000 * 5  # training dataset will repeat, wrap around
NUM_EVALS = 50  # how many times to evaluate
NUM_EVAL_EXAMPLES = 10000  # enough to get a reasonable sample

trainds = create_dataset(
    pattern='../data/taxi-train*',
    batch_size=TRAIN_BATCH_SIZE,
    mode='train')

evalds = create_dataset(
    pattern='../data/taxi-valid*',
    batch_size=1000,
    mode='eval').take(NUM_EVAL_EXAMPLES//1000) # // operator to do integer division (i.e., quotient without remainder)
```
8. High-level model evaluation using `.summary`
```
model.summary()
```
9. Running .fit (or .fit_generator) returns a History object which collects all the events recorded during training. Similar to Tensorboard, we can plot the training and validation curves for the model loss and rmse by accessing these elements of the History object.
```
RMSE_COLS = ['rmse', 'val_rmse']

pd.DataFrame(history.history)[RMSE_COLS].plot()

LOSS_COLS = ['loss', 'val_loss']

pd.DataFrame(history.history)[LOSS_COLS].plot()
```
10. Making predictions with our model
```
# The predict() method will predict the response for model
# Using tf.convert_to_tensor() we will convert the given value to a Tensor

model.predict(x={"pickup_longitude": tf.convert_to_tensor([-73.982683]),
                 "pickup_latitude": tf.convert_to_tensor([40.742104]),
                 "dropoff_longitude": tf.convert_to_tensor([-73.983766]),
                 "dropoff_latitude": tf.convert_to_tensor([40.755174]),
                 "passenger_count": tf.convert_to_tensor([3.0])},
              steps=1)
```
11. ## Export and deploy our model
Use `tf.saved_model.save` to export the trained model to a Tensorflow SavedModel format. Reference the [documentation for `tf.saved_model.save`](https://www.tensorflow.org/api_docs/python/tf/saved_model/save) as you fill in the code for the cell below.

Next, print the signature of your saved model using the SavedModel Command Line Interface command `saved_model_cli`. You can read more about the command line interface and the `show` and `run` commands it supports in the [documentation here](https://www.tensorflow.org/guide/saved_model#overview_of_commands). 
```
OUTPUT_DIR = "./export/savedmodel"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
EXPORT_PATH = os.path.join(OUTPUT_DIR,
                           datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

tf.saved_model.save(model, EXPORT_PATH) # with default serving function
```
12. ## Export the model to a TensorFlow SavedModel format
```
!saved_model_cli show \
 --tag_set serve \
 --signature_def serving_default \
 --dir {export_path}

!find {EXPORT_PATH}
os.environ['EXPORT_PATH'] = EXPORT_PATH
```
13. ## Deploy our model to AI Platform
```
%%bash
gcloud config set compute/region us-east1
gcloud config set ai_platform/region global
```
Below cell will take around 10 minutes to complete.
```
%%bash

PROJECT=qwiklabs-gcp-00-94c0998b6fe5
BUCKET=${PROJECT}
REGION=us-east1
MODEL_NAME=taxifare
VERSION_NAME=dnn

# Create GCS bucket if it doesn't exist already...
exists=$(gsutil ls -d | grep -w gs://${BUCKET}/)

if [ -n "$exists" ]; then
    echo -e "Bucket exists, let's not recreate it."
else
    echo "Creating a new GCS bucket."
    gsutil mb -l ${REGION} gs://${BUCKET}
    echo "Here are your current buckets:"
    gsutil ls
fi

if [[ $(gcloud ai-platform models list --format='value(name)' --region=$REGION | grep $MODEL_NAME) ]]; then
    echo "$MODEL_NAME already exists"
else
    echo "Creating $MODEL_NAME"
    gcloud ai-platform models create --region=$REGION $MODEL_NAME
fi

if [[ $(gcloud ai-platform versions list --model $MODEL_NAME --region=$REGION --format='value(name)' | grep $VERSION_NAME) ]]; then
    echo "Deleting already existing $MODEL_NAME:$VERSION_NAME ... "
    echo yes | gcloud ai-platform versions delete --model=$MODEL_NAME $VERSION_NAME --region=$REGION
    echo "Please run this cell again if you don't see a Creating message ... "
    sleep 2
fi

echo "Creating $MODEL_NAME:$VERSION_NAME"
gcloud ai-platform versions create --model=$MODEL_NAME $VERSION_NAME \
       --framework=tensorflow --python-version=3.7 --runtime-version=2.1 \
       --origin=$EXPORT_PATH --staging-bucket=gs://$BUCKET --region=$REGION
```
14. Creating example to test prediction with
```
%%writefile input.json
{"pickup_longitude": -73.982683, "pickup_latitude": 40.742104,"dropoff_longitude": -73.983766,"dropoff_latitude": 40.755174,"passenger_count": 3.0}  
```
15. the `gcloud ai-platform predict` sends a prediction request to AI platform for the given instances
```
!gcloud ai-platform predict \
    --model taxifare \
    --json-instances input.json \
    --version dnn \
    --region us-east1
```
### basic_intro_logistic_regression.ipynb
[basic_intro_logistic_regression](./basic_intro_logistic_regression.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/basic_intro_logistic_regression.ipynb)
1. Download dataset
```
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

# Download the training dataset file using the `tf.keras.utils.get_file` function. This returns the file path of the downloaded file.
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
```
2. Create feature and label names
```
# Column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

# Let's output the value of `Features` and `Label`
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))
```
3. Create `tf.data.Dataset`
```
batch_size = 32

# The `tf.data.experimental.make_csv_dataset()` method reads CSV files into a dataset
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
```
4. Creating DNN model in keras
```
# Here `tf.keras.Sequential` used to sequentially groups a linear stack of layers into a tf.keras.Model.
model = tf.keras.Sequential([
# `tf.keras.layers.Dense` is inherited from: `Layer`
# `tf.keras.layers.Dense` is your regular densely-connected NN layer.
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
```
5. Let's have a quick look at what this model does to a batch of features. **Returns a logit for each class**. [More about logit](https://developers.google.com/machine-learning/glossary#logits)
```
predictions = model(features)
predictions[:5]
```
6. To convert these logits to a probability for each class, use the softmax function. [More about softmax](https://developers.google.com/machine-learning/glossary#softmax). **However the results from this cell won't be accurate as model has not been trained**
```
# `tf.nn.softmax()` will compute softmax activations.
tf.nn.softmax(predictions[:5])
```
7. ## Train the model

[Training](https://developers.google.com/machine-learning/crash-course/glossary#training) is the stage of machine learning when the model is gradually optimized, or the model *learns* the dataset. The goal is to learn enough about the structure of the training dataset to make predictions about unseen data. If you learn *too much* about the training dataset, then the predictions only work for the data it has seen and will not be generalizable. This problem is called [overfitting](https://developers.google.com/machine-learning/crash-course/glossary#overfitting)—it's like memorizing the answers instead of understanding how to solve a problem.

The Iris classification problem is an example of [supervised machine learning](https://developers.google.com/machine-learning/glossary/#supervised_machine_learning): the model is trained from examples that contain labels. In [unsupervised machine learning](https://developers.google.com/machine-learning/glossary/#unsupervised_machine_learning), the examples don't contain labels. Instead, the model typically finds patterns among the features.

### Define the loss and gradient function

Both training and evaluation stages need to calculate the model's [loss](https://developers.google.com/machine-learning/crash-course/glossary#loss). This measures how off a model's predictions are from the desired label, in other words, how bad the model is performing. We want to minimize, or optimize, this value.

Our model will calculate its loss using the `tf.keras.losses.SparseCategoricalCrossentropy` function which takes the model's class probability predictions and the desired label, and returns the average loss across the examples.

```
# `tf.keras.losses.SparseCategoricalCrossentropy()` will computes the crossentropy loss between the labels and predictions.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
```
def loss(model, x, y, training):
  # TODO 2
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))
```

Use the `tf.GradientTape` context to calculate the [gradients](https://developers.google.com/machine-learning/crash-course/glossary#gradient) used to optimize your model:

```
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
```
8. ### Create an optimizer

An [optimizer](https://developers.google.com/machine-learning/crash-course/glossary#optimizer) applies the computed gradients to the model's variables to minimize the `loss` function. You can think of the loss function as a curved surface (see Figure 3) and we want to find its lowest point by walking around. The gradients point in the direction of steepest ascent—so we'll travel the opposite way and move down the hill. By iteratively calculating the loss and gradient for each batch, we'll adjust the model during training. Gradually, the model will find the best combination of weights and bias to minimize loss. And the lower the loss, the better the model's predictions.

<table>
  <tr><td>
    <img src="https://cs231n.github.io/assets/nn3/opt1.gif" width="70%"
         alt="Optimization algorithms visualized over time in 3D space.">
  </td></tr>
  <tr><td align="center">
    <b>Figure 3.</b> Optimization algorithms visualized over time in 3D space.<br/>(Source: <a href="http://cs231n.github.io/neural-networks-3/">Stanford class CS231n</a>, MIT License, Image credit: <a href="https://twitter.com/alecrad">Alec Radford</a>)
  </td></tr>
</table>

TensorFlow has many optimization algorithms available for training. This model uses the `tf.keras.optimizers.SGD` that implements the [stochastic gradient descent](https://developers.google.com/machine-learning/crash-course/glossary#gradient_descent) (SGD) algorithm. The `learning_rate` sets the step size to take for each iteration down the hill. This is a *hyperparameter* that you'll commonly adjust to achieve better results.

```
# `tf.keras.optimizers.SGD()` will Gradient descent (with momentum) optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
```

We'll use this to calculate a single optimization step:

```
loss_value, grads = grad(model, features, labels)

# Let's output the value of `Initial Loss` at `step 0`
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Let's output the value of `Loss` at `step 1`
print("Step: {},Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))
```
9. ### Training loop

With all the pieces in place, the model is ready for training! A training loop feeds the dataset examples into the model to help it make better predictions. The following code block sets up these training steps:

1. Iterate each *epoch*. An epoch is one pass through the dataset.
2. Within an epoch, iterate over each example in the training `Dataset` grabbing its *features* (`x`) and *label* (`y`).
3. Using the example's features, make a prediction and compare it with the label. Measure the inaccuracy of the prediction and use that to calculate the model's loss and gradients.
4. Use an `optimizer` to update the model's variables.
5. Keep track of some stats for visualization.
6. Repeat for each epoch.

The `num_epochs` variable is the number of times to loop over the dataset collection. Counter-intuitively, training a model longer does not guarantee a better model. `num_epochs` is a [hyperparameter](https://developers.google.com/machine-learning/glossary/#hyperparameter) that you can tune. Choosing the right number usually requires both experience and experimentation:

```
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
```
10. ### Visualize the loss function over time
While it's helpful to print out the model's training progress, it's often *more* helpful to see this progress. [TensorBoard](https://www.tensorflow.org/tensorboard) is a nice visualization tool that is packaged with TensorFlow, but we can create basic charts using the `matplotlib` module.

Interpreting these charts takes some experience, but you really want to see the *loss* go down and the *accuracy* go up:
```
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
# `plt.show()` will display a figure
plt.show()
```
11. ## Evaluate the model's effectiveness

Now that the model is trained, we can get some statistics on its performance.

*Evaluating* means determining how effectively the model makes predictions. To determine the model's effectiveness at Iris classification, pass some sepal and petal measurements to the model and ask the model to predict what Iris species they represent. Then compare the model's predictions against the actual label.  For example, a model that picked the correct species on half the input examples has an [accuracy](https://developers.google.com/machine-learning/glossary/#accuracy) of `0.5`. Figure 4 shows a slightly more effective model, getting 4 out of 5 predictions correct at 80% accuracy:

<table cellpadding="8" border="0">
  <colgroup>
    <col span="4" >
    <col span="1" bgcolor="lightblue">
    <col span="1" bgcolor="lightgreen">
  </colgroup>
  <tr bgcolor="lightgray">
    <th colspan="4">Example features</th>
    <th colspan="1">Label</th>
    <th colspan="1" >Model prediction</th>
  </tr>
  <tr>
    <td>5.9</td><td>3.0</td><td>4.3</td><td>1.5</td><td align="center">1</td><td align="center">1</td>
  </tr>
  <tr>
    <td>6.9</td><td>3.1</td><td>5.4</td><td>2.1</td><td align="center">2</td><td align="center">2</td>
  </tr>
  <tr>
    <td>5.1</td><td>3.3</td><td>1.7</td><td>0.5</td><td align="center">0</td><td align="center">0</td>
  </tr>
  <tr>
    <td>6.0</td> <td>3.4</td> <td>4.5</td> <td>1.6</td> <td align="center">1</td><td align="center" bgcolor="red">2</td>
  </tr>
  <tr>
    <td>5.5</td><td>2.5</td><td>4.0</td><td>1.3</td><td align="center">1</td><td align="center">1</td>
  </tr>
  <tr><td align="center" colspan="6">
    <b>Figure 4.</b> An Iris classifier that is 80% accurate.<br/>&nbsp;
  </td></tr>
</table>

12. ### Setup the test dataset

Evaluating the model is similar to training the model. The biggest difference is the examples come from a separate [test set](https://developers.google.com/machine-learning/crash-course/glossary#test_set) rather than the training set. To fairly assess a model's effectiveness, the examples used to evaluate a model must be different from the examples used to train the model.

The setup for the test `Dataset` is similar to the setup for training `Dataset`. Download the CSV text file and parse that values, then give it a little shuffle:
```
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

# The `tf.keras.utils.get_file` will downloads a file from a URL if it not already in the cache.
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
```
```
# The `tf.data.experimental.make_csv_dataset()` method reads CSV files into a dataset
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

# The `map()` method will pack the `features` into the training dataset:
test_dataset = test_dataset.map(pack_features_vector)
```
13. ### Evaluate the model on the test dataset

Unlike the training stage, the model only evaluates a single [epoch](https://developers.google.com/machine-learning/glossary/#epoch) of the test data. In the following code cell, we iterate over each example in the test set and compare the model's prediction against the actual label. This is used to measure the model's accuracy across the entire test set:
```
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
```
14. ## Use the trained model to make predictions

We've trained a model and "proven" that it's good—but not perfect—at classifying Iris species. Now let's use the trained model to make some predictions on [unlabeled examples](https://developers.google.com/machine-learning/glossary/#unlabeled_example); that is, on examples that contain features but not a label.

In real-life, the unlabeled examples could come from lots of different sources including apps, CSV files, and data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels. Recall, the label numbers are mapped to a named representation as:

* `0`: Iris setosa
* `1`: Iris versicolor
* `2`: Iris virginica
```
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
```
### adv_logistic_reg_TF2.0.ipynb
[adv_logistic_reg_TF2.0](./adv_logistic_reg_TF2.0.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/adv_logistic_reg_TF2.0.ipynb)

## Learning Objectives

- Load a CSV file using Pandas
- Create train, validation, and test sets
- Define and train a model using Keras (including setting class weights)
- Evaluate the model using various metrics (including precision and recall)
- Try common techniques for dealing with imbalanced data like:
    Class weighting and
    Oversampling

## Introduction 
This lab how to classify a highly imbalanced dataset in which the number of examples in one class greatly outnumbers the examples in another. You will work with the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

1. Examining class imbalance
```
# numpy bincount() method is used to obtain the frequency of each element provided inside a numpy array
neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
```
2. ### Clean, split and normalize the data

The raw data has a few issues. First the `Time` and `Amount` columns are too variable to use directly. **Drop the `Time` column (since it's not clear what it means) and take the log of the `Amount` column to reduce its range.**
```
cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)
```
3. Splitting train, test, validation sets and from numpy arrays of labels and features
```
# Use a utility from sklearn to split and shuffle our dataset.
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(cleaned_df, test_size = 0.2)
train_df, val_df = train_test_split(train_df, test_size = 0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
```
4. `liam-functions` To get an idea about arrays, especually numpy arrays. Unable to do pd.describe()
```
def array_info(array_name, name= "array"):
    print(f"{name} ndim: ", array_name.ndim)
    print(f"{name} shape:", array_name.shape)
    print(f"{name} size: ", array_name.size)
    print(f"{name} dtype: ", array_name.dtype)
    print(f"{name} itemsize: ", array_name.itemsize, "bytes")
    print(f"{name} nbytes: ", array_name.nbytes, "bytes")
```
5. Normalize the input features using the sklearn StandardScaler.
This will set the mean to 0 and standard deviation to 1.

Note: The `StandardScaler` is only fit using the `train_features` to be sure the model is not peeking at the validation or test sets. 
```
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# `np.clip` clip (limit) the values in the array
train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)
```
6. ### Look at the data distribution

Next compare the distributions of the positive and negative examples over a few features. Good questions to ask yourself at this point are:

* Do these distributions make sense? 
    * Yes. You've normalized the input and these are mostly concentrated in the `+/- 2` range.
* Can you see the difference between the distributions?
    * Yes the positive examples contain a much higher rate of extreme values.
```
# pandas dataframe is a two dimensional size mutable, potentially heterogeneous tabluar data structure with 
# labelled axis (rows and columns)
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

# seaborns jointplot displays a relationship between 2 variables (bivariate) as well as 
sns.jointplot(pos_df['V5'], pos_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
# The suptitle() function in pyplot module of the matplotlib library is used to add a title to the figure
plt.suptitle("Positive distribution")

sns.jointplot(neg_df['V5'], neg_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
_ = plt.suptitle("Negative distribution")
```
7. ## Define the model and metrics

Define a function that creates a simple neural network with a densly connected hidden layer, a [dropout](https://developers.google.com/machine-learning/glossary/#dropout_regularization) layer to reduce overfitting, and an output sigmoid layer that returns the probability of a transaction being fraudulent: 
```
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
    if output_bias is not None:
    # `tf.keras.initialisers.Constant()` generates tensors with constant values
        output_bias = tf.keras.initializers.Constant(output_bias)
    # creating a sequential model
    model = keras.Sequential([
      tf.keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),  # input shape required
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
# compile 
    model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    return model
```
8. 7b Notice that the model is fit using a larger than default batch size of 2048, this is important to ensure that each batch has a decent chance of containing a few positive samples. If the batch size was too small, they would likely have no fraudulent transactions to learn from.

**Note: this model will not handle the class imbalance well. You will improve it later in this tutorial.**

**stop training when a monitored metric has stopped improving**
```
EPOCHS = 100
BATCH_SIZE = 2048

# stop training when a monitored metric has stopped improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
```
9. display a model summary
```
model = make_model()
model.summary()
```
10. ### Optional: Set the correct initial bias.
These are initial guesses are not great. You know the dataset is imbalanced. Set the output layer's bias to reflect that (See: [A Recipe for Training Neural Networks: "init well"](http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines)). This can help with initial convergence.

#### More notes from website about `init well`

Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.

With the default bias initialization the loss should be about `math.log(2) = 0.69314` 
```
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
```
The correct bias to set can be derived from:

$$ p_0 = pos/(pos + neg) = 1/(1+e^{-b_0}) $$
$$ b_0 = -log_e(1/p_0 - 1) $$
$$ b_0 = log_e(pos/neg)$$
```
# np.log() is a mathematical function that is used to calculate the natural logarithm
initial_bias = np.log([pos/neg])
initial_bias
```
Set that as the initial bias, and the model will give much more reasonable initial guesses. 

It should be near: `pos/total = 0.0018`
```
model = make_model(output_bias = initial_bias)
model.predict(train_features[:10])
```
11. Plot loss function
```
def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()
```
```
plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
```
12. ### Plot the ROC

Now plot the [ROC](https://developers.google.com/machine-learning/glossary#ROC). This plot is useful because it shows, at a glance, the range of performance the model can reach just by tuning the output threshold.

ROC function and example
```
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
```
```
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
```
13. ### Calculate class weights

The goal is to identify fradulent transactions, but you don't have very many of those positive samples to work with, so you would want to have the classifier heavily weight the few examples that are available. You can do this by passing Keras weights for each class through a parameter. **These will cause the model to "pay more attention" to examples from an under-represented class.**

```
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2
weight_for_1 = (1 / pos)*(total)/2

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
```
14. ### Train a model with class weights

Now try re-training and evaluating the model with class weights to see how that affects the predictions.

Note: Using `class_weights` changes the range of the loss. This may affect the stability of the training depending on the optimizer. Optimizers whose step size is dependent on the magnitude of the gradient, like `optimizers.SGD`, may fail. The optimizer used here, `optimizers.Adam`, is unaffected by the scaling change. Also note that because of the weighting, the total losses are not comparable between the two models.
```
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight) 
```
15. Evaluate metrics
```
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
```
```
weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)
```
16. ### Oversample the minority class

A related approach would be to resample the dataset by oversampling the minority class.
```
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]
```
#### Using NumPy

You can balance the dataset manually by choosing the right number of random 
indices from the positive examples:
```
# np.arrange() return evenly spaced values within a given interval
ids = np.arange(len(pos_features))
# choice method(), you can get the random samples of 1 dimensional array and return the random samples of numpy array
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape
```
```
# numpy.concatenate() function concatenate a sequence of arrays along an existing axis
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))

# numpy.random.shuffle() modify a sequence in-place by shuffling its contents
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape
```
17. ### Plotting baseline, resampled and weighted

```
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])
plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')
plt.legend(loc='lower right')
```
# Readings Notes
[In this tutorial](https://machinelearningmastery.com/keras-functional-api-deep-learning/), you discovered how to use the functional API in Keras for defining simple and complex deep learning models.

Specifically, you learned:

    The difference between the Sequential and Functional APIs.
    How to define simple Multilayer Perceptron, Convolutional Neural Network, and Recurrent Neural Network models using the functional API.
    How to define more complex models with shared layers and multiple inputs and outputs.

### Best Practices

In this section, I want to give you some tips to get the most out of the functional API when you are defining your own models.

    Consistent Variable Names. Use the same variable name for the input (visible) and output layers (output) and perhaps even the hidden layers (hidden1, hidden2). It will help to connect things together correctly.
    Review Layer Summary. Always print the model summary and review the layer outputs to ensure that the model was connected together as you expected.
    Review Graph Plots. Always create a plot of the model graph and review it to ensure that everything was put together as you intended.
    Name the layers. You can assign names to layers that are used when reviewing summaries and plots of the model graph. For example: Dense(1, name=’hidden1′).
    Separate Submodels. Consider separating out the development of submodels and combine the submodels together at the end.

### 4_keras_functional_api.ipynb
[4_keras_functional_api](./4_keras_functional_api.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/introduction_to_tensorflow/labs/4_keras_functional_api.ipynb)

1. Processes features and labels function
```
def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
        
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label
```
2. `create_dataset` splits into training dataset function
```
def create_dataset(pattern, batch_size=1, mode='eval'):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    dataset = dataset.map(features_and_labels)
    
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    return dataset
```
3. Create Feature columns for Wide and Deep model. 
```
from tensorflow import feature_column as fc

# 1. Bucketize latitudes and longitudes
NBUCKETS = 16
latbuckets = np.linspace(start=38.0, stop=42.0, num=NBUCKETS).tolist()
lonbuckets = np.linspace(start=-76.0, stop=-72.0, num=NBUCKETS).tolist()

fc_bucketized_plat = fc.bucketized_column(source_column=fc.numeric_column('pickup_latitude'), 
                                                         boundaries=latbuckets)
fc_bucketized_plon = fc.bucketized_column(source_column=fc.numeric_column('pickup_longitude'), 
                                                         boundaries=lonbuckets)
fc_bucketized_dlat = fc.bucketized_column(source_column=fc.numeric_column('dropoff_latitude'),
                                                         boundaries=latbuckets)
fc_bucketized_dlon = fc.bucketized_column(source_column=fc.numeric_column('dropoff_longitude'),
                                                         boundaries=lonbuckets)

# 2. Cross features for locations
fc_crossed_dloc = fc.crossed_column([fc_bucketized_dlat,fc_bucketized_dlon],
                                                   hash_bucket_size=NBUCKETS * NBUCKETS)
fc_crossed_ploc = fc.crossed_column([fc_bucketized_plat,fc_bucketized_plon],
                                                   hash_bucket_size=NBUCKETS * NBUCKETS)
fc_crossed_pd_pair = fc.crossed_column([fc_crossed_dloc,fc_crossed_ploc],
                                                     hash_bucket_size=NBUCKETS**4) # to the power of

# 3. Create embedding columns for the crossed columns
# not sure how dimension is calculated
fc_pd_pair = fc.embedding_column(categorical_column=fc_crossed_pd_pair, dimension=3) 
fc_dloc = fc.embedding_column(categorical_column=fc_crossed_dloc, dimension=3)
fc_ploc = fc.embedding_column(categorical_column=fc_crossed_ploc, dimension=3)
```
4. Collect the wide and deep columns into two separate lists
```
wide_columns = [
    # One-hot encoded feature crosses
    fc.indicator_column(fc_crossed_dloc),
    fc.indicator_column(fc_crossed_ploc), 
    fc.indicator_column(fc_crossed_pd_pair)
]

deep_columns = [
    # Embedding_column to "group" together ...
    # not sure how dimension is calculated
    fc.embedding_column(fc_crossed_pd_pair, dimension=10),

    # Numeric columns
    fc.numeric_column('pickup_latitude'),
    fc.numeric_column('pickup_longitude'),
    fc.numeric_column('dropoff_latitude'),
    fc.numeric_column('dropoff_longitude')
]
```
5. Build the model
```def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def build_model(dnn_hidden_units):
    # Create the deep part of model
    # A layer that produces a dense Tensor based on given feature_columns.
    deep = DenseFeatures(deep_columns, name='deep_inputs')(inputs)
    for num_nodes in dnn_hidden_units:
        deep = Dense(num_nodes, activation='relu')(deep)
    
    # Create the wide part of model
    wide = DenseFeatures(wide_columns, name='wide_inputs')(inputs)

    # Combine deep and wide parts of the model
    combined = concatenate(inputs=[deep, wide], name='combined')

    # Map the combined outputs into a single prediction value
    output = Dense(units=1, activation=None, name='prediction')(combined)
    
    # Finalize the model
    model = Model(inputs=list(inputs.values()), outputs=output)

    # Compile the keras model
    model.compile(optimizer='adam', loss="mse", metrics=[rmse, 'mse'])
    
    return model
```
6. Plot model diagram
```
HIDDEN_UNITS = [10,10]

model = build_model(dnn_hidden_units=HIDDEN_UNITS)

tf.keras.utils.plot_model(model, show_shapes=False, rankdir='LR')
```
7. We'll set up our training variables, create our datasets for training and validation, and train our model.

(We refer you the the blog post [ML Design Pattern #3: Virtual Epochs](https://medium.com/google-cloud/ml-design-pattern-3-virtual-epochs-f842296de730) for further details on why express the training in terms of `NUM_TRAIN_EXAMPLES` and `NUM_EVALS` and why, in this training code, the number of epochs is really equal to the number of evaluations we perform.)
```
BATCH_SIZE = 1000
NUM_TRAIN_EXAMPLES = 10000 * 5  # training dataset will repeat, wrap around
NUM_EVALS = 50  # how many times to evaluate
NUM_EVAL_EXAMPLES = 10000  # enough to get a reasonable sample

trainds = create_dataset(
    pattern='../data/taxi-train*',
    batch_size=BATCH_SIZE,
    mode='train')

evalds = create_dataset(
    pattern='../data/taxi-valid*',
    batch_size=BATCH_SIZE,
    mode='eval').take(NUM_EVAL_EXAMPLES//1000)
```
```
%%time
steps_per_epoch = NUM_TRAIN_EXAMPLES // (BATCH_SIZE * NUM_EVALS)

OUTDIR = "./taxi_trained"
shutil.rmtree(path=OUTDIR, ignore_errors=True) # start fresh each time

history = model.fit(x=trainds,
                    steps_per_epoch=steps_per_epoch,
                    epochs=NUM_EVALS,
                    validation_data=evalds,
                    callbacks=[TensorBoard(OUTDIR)])
```
8. Graph of `MSE` and `RSME`
```
RMSE_COLS = ['rmse', 'val_rmse']

pd.DataFrame(history.history)[RMSE_COLS].plot()
```