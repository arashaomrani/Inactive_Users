import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

def data_preprocessing(data, churn_threshold=7, training=True):
  """
  Load the csv file from data argumrnet and preprocess it for training.
  """
  df = pd.read_csv(data, index_col=0)
  most_recent_date = datetime.strptime(df['last_activity_dt'].max(), '%Y-%m-%d')
  df['last_activity_dt'] = pd.to_datetime(df['last_activity_dt'])
  df['first_activity_dt'] = pd.to_datetime(df['first_activity_dt'])
  df['duration'] = (df['last_activity_dt'] - df['first_activity_dt']).dt.days

  # Creating 'is_churned' column
  if training:
    days_since_last_activity = (most_recent_date - df['last_activity_dt']).dt.days
    y = (days_since_last_activity > churn_threshold).astype(int)


  df.drop(['player_id', 'first_activity_dt', 'last_activity_dt'], axis=1, inplace=True)

  cat_features = df.select_dtypes(include=['object']).columns
  # Changing the categorical features to one-hot-encoding
  if training:
    enc = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    encoded = enc.fit_transform(df[list(cat_features)])
    joblib.dump(enc, 'encoder.joblib')
  else:
    enc = joblib.load('encoder.joblib')
    encoded = enc.transform(df[list(cat_features)])

  # Dropping the categorical features
  df.drop(list(cat_features), axis=1, inplace=True)
  df = pd.concat([df, pd.DataFrame(encoded)], axis=1)

  # Filling the NaN with mean of the column
  df.columns = df.columns.astype(str)
  if training:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df)
    joblib.dump(imp_mean,'imp.joblib')
  else:
    imp_mean = joblib.load('imp.joblib')
  imp_mean.set_output(transform="pandas")
  df_fill = imp_mean.transform(df)

  # Changing the rage of numerical featire to (0, 1)
  if training:
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(df_fill.iloc[:,:8])
    joblib.dump(scaler, 'scaler.joblib')
    df_fill_scale = pd.concat([pd.DataFrame(temp), df_fill.iloc[:,8:]], axis=1)

    # Defining the training features and the target
    # Split the dataset to train and test set for training
    X_train, X_test, y_train, y_test = train_test_split(df_fill_scale, y, test_size=0.1, random_state=2, shuffle=True)

    return df, df_fill, df_fill_scale, X_train, X_test, y_train, y_test
  
  else:
    scaler = joblib.load('scaler.joblib')
    temp = scaler.transform(df_fill.iloc[:,:8])
    df_fill_scale = pd.concat([pd.DataFrame(temp), df_fill.iloc[:,8:]], axis=1)
    return df_fill_scale

def model(X_train, oprimizer='adam'):
  """
  Build the Keras model and compile it.
  """
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),
      keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
  ])

  model.compile(optimizer=oprimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

def train(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32, path='model.keras'):
  """
  Train and save the model and return the history.
  """
  def scheduler(epoch, lr):
    if epoch < 20:
      return lr
    else:
      return lr * tf.math.exp(-0.1)

  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

  history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[callback],
                      epochs=epochs, batch_size=batch_size)
  model.save(path)
  return history
  
def predict(model, data, output):
  model = tf.keras.models.load_model(model)
  df_fill_scale = data_preprocessing(data, training=False)
  p = model.predict(df_fill_scale)
  pd.DataFrame(np.concatenate([p,np.where(p > 0.5, 1,0)], axis=1),
               columns=['prediction', 'class']).to_csv(output)
  return p, np.where(p > 0.5, 1,0)

