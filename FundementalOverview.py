import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime as datetime
import statsmodels.api as sm
from IPython.display import display



class ReadDataframes:
    def __init__(self, csv_file1, csv_file2, csv_file3, csv_file4):
        self.netIncome = pd.read_csv(csv_file1)
        self.netIncome = self.format_nI(self.netIncome)

        self.Oil_price = pd.read_csv(csv_file2)
        self.Oil_price = self.format_oil(self.Oil_price)

        self.Natgas_price = pd.read_csv(csv_file3)
        self.Natgas_price = self.format_natgas(self.Natgas_price)

    
        self.Stock_price = pd.read_csv(csv_file4)
        self.Stock_price = self.format_stock(self.Stock_price)

        #above we are importating all the dataframes and formatting them to be used in the program
        #below we are creating a datetime index (allows us to plot the dataframes on the same graph)
        #then we filter the dataframe limits so that the min is the minimum for the dataset
        #that has the closest min to today. This would be net income.
        #then we scale the data so that it is all on the same scale 
        #and can be plotted on the same graph

        self.Oil_price, self.netIncome, self.Natgas_price, self.Stock_price = self.create_datetime(self.netIncome, self.Oil_price, self.Natgas_price, self.Stock_price)

        self.Oil_price_filtered, self.Natgas_price_filtered, self.Stock_price_filtered = self.filter_data(self.Oil_price, self.netIncome, self.Natgas_price, self.Stock_price)

        self.Oil_price_scaled, self.netIncome_scaled, self.Natgas_price_scaled, self.Stock_price_scaled = self.scale_data(self.netIncome, self.Oil_price_filtered, self.Natgas_price_filtered, self.Stock_price_filtered)

        self.modela, self.modelb = self.create_regression(self.Oil_price_scaled, self.Natgas_price_scaled, self.Stock_price_scaled)

        self.plot_multiple_oil_types(self.Oil_price_scaled, self.netIncome_scaled, self.Natgas_price_scaled, self.Stock_price_scaled, self.modela, self.modelb)

    def format_nI(self, netIncome):
        netIncome.set_index('fiscalDateEnding', inplace=True)
        return netIncome['netIncome']
    
    def format_oil(self, Oil_price):
        Oil_price.set_index('date', inplace=True)
        return Oil_price['value']
    
    def format_natgas(self, Natgas_price):
        Natgas_price.set_index('date', inplace=True)
        return Natgas_price['value']
    
    #these three functions above are used to format the dataframes 
    #so that they can be used in the program
    
   
    def format_stock(self, Stock_price):
        Stock_price.set_index('date', inplace=True)
        Stock_price = Stock_price[['1. open','5. adjusted close']] # [[]] -> selecting a list of columns
        Stock_price.loc[:, 'Average'] = Stock_price[['1. open', '5. adjusted close']].mean(axis=1)
        return Stock_price['Average']
    
    #meanwhile this last function is a lot more complicated, but because stocks have a open and 
    #close price we need to take the average of the two to get a single price for the week
    
    def create_datetime(self, netIncome, Oil_price, Natgas_price, Stock_price):
        # convert the index to datetime
        netIncome.index = pd.to_datetime(netIncome.index)
        Oil_price.index = pd.to_datetime(Oil_price.index)
        Natgas_price.index = pd.to_datetime(Natgas_price.index)
        Stock_price.index = pd.to_datetime(Stock_price.index)
        
        return Oil_price, netIncome, Natgas_price, Stock_price
    
    #creating datetime index is a crucial step, luckily the dates in thes data sets
    #are already in the correct format so we just need to convert them to datetime
    
    def filter_data(self, Oil_price, netIncome, Natgas_price, Stock_price):
        # Filter the oil_prices so only the dates that are in the netIncome 
        #dataframe are included creating limits for the oil price data 
        #net income is by far the smallest dataset so we use it as the min and max
        start_date = netIncome.index.min()
        end_date = netIncome.index.max()
        Oil_price_filtered = Oil_price[(Oil_price.index >= start_date) & (Oil_price.index <= end_date)] 
        Natgas_price_filtered = Natgas_price[(Natgas_price.index >= start_date) & (Natgas_price.index <= end_date)]
        Stock_price_filtered = Stock_price[(Stock_price.index >= start_date) & (Stock_price.index <= end_date)]

        return Oil_price_filtered, Natgas_price_filtered, Stock_price_filtered
    

    def scale_data(self, netIncome, Oil_price_filtered, Natgas_price_filtered, Stock_price_filtered):
        scaler_oil = MinMaxScaler()
        scaler_net_income = MinMaxScaler()
        scaler_natgas = MinMaxScaler()
        scaler_stock = MinMaxScaler()

        # Scale the data and convert back to DataFrame to retain the index
        Oil_price_scaled = pd.DataFrame(scaler_oil.fit_transform(Oil_price_filtered.values.reshape(-1, 1)), index=Oil_price_filtered.index, columns=['Scaled Oil Price'])
        netIncome_scaled = pd.DataFrame(scaler_net_income.fit_transform(netIncome.values.reshape(-1, 1)), index=netIncome.index, columns=['Scaled Net Income'])
        Natgas_price_scaled = pd.DataFrame(scaler_natgas.fit_transform(Natgas_price_filtered.values.reshape(-1, 1)), index=Natgas_price_filtered.index, columns=['Scaled Natgas Price'])
        Stock_price_scaled = pd.DataFrame(scaler_stock.fit_transform(Stock_price_filtered.values.reshape(-1, 1)), index=Stock_price_filtered.index, columns=['Scaled Stock Price'])
        return Oil_price_scaled, netIncome_scaled, Natgas_price_scaled, Stock_price_scaled
    
    
        #this scale function offers a lot of flexibility, we can scale the 
        #data to any range we want and interpret data in a proportionate way
        #this is all due to the functionality of the MinMaxScaler() function
        #(-1,1) <-- this is telling the function to scale the data between -1 and 1
        #where negative one represents whatever nessisary hence the flexibility
    
    def create_regression(self, Oil_price_scaled, Natgas_price_scaled, Stock_price_scaled):
        # Align the indices of the scaled dataframes
        aligned_data = pd.concat([Oil_price_scaled, Natgas_price_scaled, Stock_price_scaled], axis=1, join='inner')


        # Create a regression model using the statsmodels library
        # The dependent variable is the net income and the independent variables are the oil price, natural gas price, and stock price
        # We add a constant to the independent variables to get an intercept
        a = aligned_data['Scaled Oil Price']
        b = aligned_data['Scaled Natgas Price']
        y = aligned_data['Scaled Stock Price']
        ax = sm.add_constant(a)
        bx = sm.add_constant(b)

        modela = sm.OLS(y, ax).fit()  # Regression model for oil price
        modelb = sm.OLS(y, bx).fit()  # Regression model for natural gas price
     # Regression model for stock price
        return modela, modelb
        
        
    def plot_multiple_oil_types(self, Oil_price_scaled, netIncome_scaled, Natgas_price_scaled, Stock_price_scaled, modela, modelb):
        plt.figure(figsize=(12, 6))
        
        # Plot using the index of each DataFrame for the x-axis and the scalued values 
        #for the y-axis,  Oil_price_scaled['Scaled Oil Price'], although this series is not 
        #nessisary as it is the only column in the dataframe, it is good practice to to create
        #the flexabilty to add more columns to the dataframe in the future
        plt.scatter(Oil_price_scaled.index, Oil_price_scaled['Scaled Oil Price'], label='WTI Oil Price (Scaled)')
        plt.scatter(netIncome_scaled.index, netIncome_scaled['Scaled Net Income'], label='Company Net Income (Scaled)', color='orange', linestyle='--')
        plt.scatter(Natgas_price_scaled.index, Natgas_price_scaled['Scaled Natgas Price'], label='Natgas Price (Scaled)', color='green', linestyle='--')
        plt.scatter(Stock_price_scaled.index, Stock_price_scaled['Scaled Stock Price'], label='Stock Price (Scaled)', color='red', linestyle='--')
        # Ensure the predictions are aligned with the dates
        common_dates = Oil_price_scaled.index.intersection(Natgas_price_scaled.index).intersection(Stock_price_scaled.index)

        plt.plot(common_dates, modela.predict(sm.add_constant(Oil_price_scaled.loc[common_dates, 'Scaled Oil Price'])), label='Regression Model (Oil)', color='blue')
        plt.plot(common_dates, modelb.predict(sm.add_constant(Natgas_price_scaled.loc[common_dates, 'Scaled Natgas Price'])), label='Regression Model (Natgas)', color='green')

        plt.title('WTI Oil Price vs. Company Net Income')
        plt.xlabel('Date')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.show()


#example usage
#note this function is very flexible for other types of data as long
#as the structure is formatted correctly
csv_file1 = 'netIncome.csv'
csv_file2 = 'crudeOil.csv'
csv_file3 = 'NATURAL_GAS.csv'
csv_file4 = 'stock_data.csv'

run = ReadDataframes(csv_file1, csv_file2, csv_file3, csv_file4)




