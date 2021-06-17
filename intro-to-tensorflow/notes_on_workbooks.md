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
2. create a tf.data dataset from a Pandas Dataframe
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
3. tf.feature_column
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