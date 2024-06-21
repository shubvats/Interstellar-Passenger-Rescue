# Importing necessary libraries
import numpy as np
import pandas as pd
import os

# Input data files directory
input_dir = '/kaggle/input/spaceship-titanic'

# Listing all files in the input directory
for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Loading the training dataset
df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
df = pd.DataFrame(df)

# Displaying the first 5 rows of the dataset
df.head(5)

# Checking the shape of the dataset
df.shape

# Counting total missing values in the dataset
total_missing_value = df.isnull().sum().sum()
total_missing_value

# Checking data types of columns
df.dtypes

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

# Importing visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Importing TensorFlow and TensorFlow Decision Forests
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Checking TensorFlow and TensorFlow Decision Forests versions
print("TensorFlow version:", tf.__version__)
print("TensorFlow Decision Forests version:", tfdf.__version__)

# Creating a bar chart to visualize the distribution of 'Transported' column
barChart_Transported = df.Transported.value_counts()
barChart_Transported.plot(kind="bar")
plt.xlabel('Transported')
plt.ylabel('Count')
plt.title('Distribution of Transported Passengers')
plt.show()

# Dropping columns 'PassengerId' and 'Name' from the dataset
df = df.drop(['PassengerId', 'Name'], axis=1)
df.head(3)

# Handling missing values by filling NaNs with 0 for certain columns
df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
df.isnull().sum().sort_values(ascending=False)

# Converting boolean columns to integer type
df['VIP'] = df['VIP'].astype(int)
df['CryoSleep'] = df['CryoSleep'].astype(int)

# Creating new features 'Deck', 'Cabin_num', and 'Side' from 'Cabin' column
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
df = df.drop('Cabin', axis=1)
df.dtypes

# Splitting the dataset into training and validation sets
def split_dataset(dataset, test_ratio=0.20):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

train_ds_pd.head(2)
valid_ds_pd.head(2)

# Converting Pandas DataFrame to TensorFlow Datasets format
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label='Transported')
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label='Transported')

train_ds

# Listing available models in TensorFlow Decision Forests
tfdf.keras.get_all_models()

# Selecting RandomForestModel
Randforest_trdf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")

# Compiling the model
Randforest_trdf.compile(metrics=["accuracy"])

# Training the model
Randforest_trdf.fit(x=train_ds)

# Evaluating the model on training dataset
logs = Randforest_trdf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.title("Training Progress")
plt.show()

# Evaluating the model on validation dataset
evaluation = Randforest_trdf.evaluate(x=valid_ds, return_dict=True)
print("Validation Accuracy:", evaluation['accuracy'])

# Retrieving model inspector to analyze variable importances
inspector = Randforest_trdf.make_inspector()
print("Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

# Preparing CSV for submission
# Loading the test dataset
test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'))
submission_id = test_df.PassengerId

# Handling missing values and creating new features as done with training dataset
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Converting test Pandas DataFrame to TensorFlow Datasets format
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Getting predictions for test dataset
predictions = Randforest_trdf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

output.head()

# Creating a sample submission file
sample_submission_df = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()
