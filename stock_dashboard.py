import streamlit as st ,pandas as pd,numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt

st.title('Stock Dashboard')

stocks = ("GOOG","APPL","MSFT","TESLA")
selected_stocks = st.selectbox("Select Stock for prediction",stocks)
ticker=st.sidebar.text_input('Ticker')
start_date=st.sidebar.date_input('Start Date')
end_date=st.sidebar.date_input('End Date')

uploaded_file = st.file_uploader("Choose a dataset_train file")
if uploaded_file is not None:
    dataset_train= pd.read_csv(uploaded_file)
    st.write(dataset_train)
    
fig= px.line(dataset_train , x= dataset_train.Date ,y =dataset_train['Close'], title= ticker)
st.plotly_chart(fig)

training_set = dataset_train.iloc[:, 1:2].values
print(training_set)
print(training_set.shape)

# normalizing the dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_training_set = scaler.fit_transform(training_set)
print(scaled_training_set)
X_train = []
Y_train = []
for i in range(60, 1259):
    X_train.append(scaled_training_set[i-60:i, 0])
    Y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)
# reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
from tensorflow.keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences= True, name='lstm_1' ,input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True, name='lstm_2' ))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True , name='lstm_3' ))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50 , name='lstm_4' ))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, name='lstm_5' ))

# fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, epochs =10 , batch_size =32)

uploaded_file = st.file_uploader("Choose a dataset_test file")
if uploaded_file is not None:
    dataset_test= pd.read_csv(uploaded_file)
    st.write(dataset_test)
actual_stock_price = dataset_test.iloc[:, 1:2].values

# preparing input for the model
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, 180):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

fig1= px.line(dataset_test,x= dataset_test.Date,y=dataset_test['Close'],title= ticker)
st.plotly_chart(fig1)



# Generate some example data
x=dataset_test.Date
actual_price=st.write(actual_stock_price)
predicted_price=st.write(predicted_stock_price)
# Plot the line graphs using matplotlib
fig, ax = plt.subplots()
# Plot the data
ax.plot(x,actual_price,label='Actual Price')
ax.plot(x,predicted_price,label='Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Close_Price')
ax.set_title('Price Comparison')
ax.legend()
# Display the plot using Streamlit
st.pyplot(fig)


pricing_data = st.tabs["Pricing Data"]

with pricing_data:
    st.header('Price Movements')
    data2=dataset_test
    data2['% change'] = dataset_test['Adj Close']/dataset_test['Adj Close'].shift(1)-1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% change'].mean()*252*100
    st.write('Annual Return is',annual_return,'%')
    stdev = np.std(data2['% change'])*np.sqrt(252)
    st.write('Standard Deviation is',stdev*100,'%')
    st.write('Risk Adj Return is',annual_return/stdev*100)