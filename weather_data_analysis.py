# Importing required libraries
import pandas as pd                   # For data Manipulation and analysis
import matplotlib.pyplot as plt       # For creating static Visualizations    
import seaborn as sns                 # For statistical data visualization
from sklearn.model_selection import train_test_split    # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression       # For implementing linear regression
from sklearn.metrics import mean_squared_error          # for calculating model performance
  
# Loading the dataset
df = pd.read_csv('weather.csv')  Reads the CSV file into pandas DataFrame


print(df.head())
print(df.info())
print(df.describe())


sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()





df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()


plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show() 



X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')



highest_rainfall_month = monthly_avg_max_temp.idxmax()
lowest_rainfall_month = monthly_avg_max_temp.idxmin()
print(f'Highest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')
