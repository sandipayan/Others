import math
import numpy as np
import datetime as dt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
 
max_lag = 250
max_trades = 3
trade_list = [[], #company 1
              [], #company 2
              [], #original price spread
              [], #original price spread average
              [], #stock_x quantity
              []] #stock_y quantity
 
all_pairs = [] #{[(ticker_x, ticker_y), (sid_x, sid_y), is_cointegrated, dif_list, ave, stdev]}
 
# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    global all_pairs    
 
    #dictionary of our stock universe
    #(stock sid, stock history with length max_lag)
    context.stocks = {"CAG": (sid(1228), []),
                      "MO": (sid(5885), []),
                      "SRE": (sid(24778), []),
                      "VTR": (sid(18821), []),
                      "YUM": (sid(17787), []),
                      "NE": (sid(5249), []),}
   
    tuple_list = [('CAG', 'MO'),('SRE', 'VTR'),('YUM', 'NE')]  
   
    for t in tuple_list:
        ticker_x = t[0]
        ticker_y = t[1]
       
        sid_x = (context.stocks)[ticker_x][0]
        sid_y = (context.stocks)[ticker_y][0]
        #get difference
        dif_list = []
        stdev = 0
        #get average
        ave = 0
        #get cointegration
        is_cointegrated = False
        #append pair
        all_pairs.append( [(ticker_x, ticker_y),
                          (sid_x, sid_y),
                          is_cointegrated,
                          dif_list,
                          ave,
                          stdev])
   
#conduct augmented dickey fuller or array x with a default
#level of 10%
def is_stationary(x, p = 10):
   
    x = np.array(x)
    result = ts.adfuller(x, regression='ctt')
    #1% level
    if p == 1:
        #if DFStat <= critical value
        if result[0] >= result[4]['1%']:        #DFstat is less negative
            #is stationary
            return True
        else:
            #is nonstationary
            return False
    #5% level
    if p == 5:
        #if DFStat <= critical value
        if result[0] >= result[4]['5%']:        #DFstat is less negative
            #is stationary
            return True
        else:
            #is nonstationary
            return False
    #10% level
    if p == 10:
        #if DFStat <= critical value
        if result[0] >= result[4]['10%']:        #DFstat is less negative
            #is stationary
            return True
        else:
            #is nonstationary
            return False
   
   
#Engle-Granger test for cointegration for array x and array y
def are_cointegrated(x, y):
 
    #check x is I(1) via Augmented Dickey Fuller
    x_is_I1 = not(is_stationary(x))
    #check y is I(1) via Augmented Dickey Fuller
    y_is_I1 = not(is_stationary(y))
    #if x and y are no stationary        
    if x_is_I1 and y_is_I1:
        X = sm.add_constant(x)
        #regress x on y
        model = sm.OLS(np.array(y), np.array(X))
        results = model.fit()
        const = results.params[1]
        beta_1 = results.params[0]
        #solve for ut_hat
        u_hat = []
        for i in range(0, len(y)):
            u_hat.append(y[i] - x[i] * beta_1 - const)    
        #check ut_hat is I(0) via Augmented Dickey Fuller
        u_hat_is_I0 = is_stationary(u_hat)
        #if ut_hat is I(0)
        if u_hat_is_I0:
            #x and y are cointegrated
            return True
        else:
            #x and y are not cointegrated
            return False
    #if x or y are nonstationary they are not cointegrated
    else:
        return False
       
#update all pairs with new information      
def update_all_pairs(context):
    #pair = [(ticker_x, ticker_y), (sid_x, sid_y), is_cointegrated, dif_list, ave, stdev]
    global all_pairs
   
    #for each pair    
    for p in range(0, len(all_pairs)):
        ticker_x = all_pairs[p][0][0]
        ticker_y = all_pairs[p][0][1]
 
        #get history
        x_history = (context.stocks)[ticker_x][1]
        y_history = (context.stocks)[ticker_y][1]
        #get difference
        dif_list = []
        for i in range (0, len(x_history)):
            dif_list.append(x_history[i] - y_history[i])
        #get stdev
        stdev = np.std(dif_list)
        #get average
        ave = np.average(dif_list)
        #get cointegration
        is_cointegrated = are_cointegrated(x_history, y_history)
        #update information
        all_pairs[p] = [all_pairs[p][0], all_pairs[p][1], is_cointegrated, dif_list, ave, stdev]
        log.info(str(all_pairs[p]))
 
#buy signal
def buy_signal(context, data, pair_index):
    global max_trades
    global all_pairs
    global trade_list
    global max_lag
   
    #allocate cash for each trade
    cash_per_trade = (context.portfolio.cash)/(2*max_trades)
    #get stock_x current information
    stock_x_root = (context.stocks)[all_pairs[pair_index][0][0]]
    stock_x = stock_x_root[0]
    stock_x_data = data.current(stock_x,'price')
    stock_x_price = math.log10(stock_x_data)
    shares_x = int(cash_per_trade/stock_x_price)
    #get stock_y current information
    stock_y_root = (context.stocks)[all_pairs[pair_index][0][1]]
    stock_y = stock_y_root[0]
    stock_y_data = data.current(stock_y,'price')
    stock_y_price = math.log10(stock_y_data)
    shares_y = int(cash_per_trade/stock_y_price)
    #compare the price difference in stock_x and stock_y
    ave = all_pairs[pair_index][4]
    stdev = all_pairs[pair_index][5]
    cointegrated = all_pairs[pair_index][2]
       
    #if there is enough price data in our stock's history
    if len(stock_x_root[1]) == max_lag and len(stock_y_root[1]) == max_lag:
        #if the stocks are cointegrated
        if cointegrated:
            #if the difference in the normalized price is greater than 2 historical stdevs
            if (abs(stock_x_price - stock_y_price) >= (abs(ave)+stdev)) and (len(trade_list[0]) < max_trades):
                #is stock_x is above its relative price or stock_y below its relative price
                if (stock_x_price - stock_y_price) > ave:
                    #sell x, buy y
                    order(stock_x, -shares_x)
                    order(stock_y, shares_y)
                    trade_list[0].append(stock_x)
                    trade_list[1].append(stock_y)
                    trade_list[2].append(stock_x_price - stock_y_price)
                    trade_list[3].append(ave)
                    trade_list[4].append(shares_x)
                    trade_list[5].append(shares_y)                    
         
                #if stock_x is trading below its relative price or stock_y above its relative price
                else:
                    #sell stock_y and buy stock_x
                    order(stock_y, -shares_y)
                    order(stock_x, shares_x)
                    trade_list[0].append(stock_x)
                    trade_list[1].append(stock_y)
                    trade_list[2].append(stock_x_price - stock_y_price)
                    trade_list[3].append(ave)
                    trade_list[4].append(shares_x)
                    trade_list[5].append(shares_y)                  
   
   
#sell signal
def sell_signal(context, data, pair_index, trade_index):
    global all_pairs
    global trade_list
    #get stock_x current information
    stock_x = all_pairs[pair_index][1][0]
    stock_x_data = data.current(stock_x,'price')
    stock_x_price = math.log10(stock_x_data)
    shares_x = trade_list[4][trade_index]
    #get stock_y current information
    stock_y = all_pairs[pair_index][1][1]
    stock_y_data = data.current(stock_y,'price')
    stock_y_price = math.log10(stock_y_data)
    shares_y = trade_list[5][trade_index]
    #compare the price difference in stock_x and stock_y
    ave = all_pairs[pair_index][4]
    old_ave = trade_list[3][trade_index]
   
    #if the original difference > old average
    if trade_list[2][trade_index] > old_ave:
        #sell if the current difference < current_ave    (crossover)
        if (stock_x_price - stock_y_price) < ave:
            order(stock_x, shares_x)
            order(stock_y, -shares_y)
            trade_list[0].pop(trade_index)
            trade_list[1].pop(trade_index)
            trade_list[2].pop(trade_index)
            trade_list[3].pop(trade_index)
            trade_list[4].pop(trade_index)
            trade_list[5].pop(trade_index)
           
    #if the orignal difference < old average
    else:
        #sell if the current difference > current average    (crossover)
        if (stock_x_price - stock_y_price) > ave:
            order(stock_y, shares_y)
            order(stock_x, -shares_x)
            trade_list[0].pop(trade_index)
            trade_list[1].pop(trade_index)
            trade_list[2].pop(trade_index)
            trade_list[3].pop(trade_index)
            trade_list[4].pop(trade_index)
            trade_list[5].pop(trade_index)    
 
   
   
# Will be called on every trade event for the securities you specify.
def handle_data(context, data):
    global all_pairs
    global trade_list
    global max_trades
    global max_lag
    global data_collected
   
    #get stock data
    ticker_list = (context.stocks).keys()
    stock = (context.stocks)[ticker_list[0]][0]
    stock_data = data.current(stock,'price')
    #get the current time
    time_list = ((str(get_datetime()).split(" "))[1]).split(":")
    (hour, minute) = (int(time_list[0]), int(time_list[1]))
    hour = hour - 4        #adjust for time difference
    #update all pairs if the market just opened
    if hour == 9 and minute == 31:
        if len((context.stocks)[ticker_list[0]][1]) == max_lag:
            update_all_pairs(context)      
    #append price data if the market just closed
    if hour == 16 and minute == 0:
        ticker_list = (context.stocks).keys()
        for ticker in ticker_list:
            #get ticker data
            stock = (context.stocks)[ticker][0]
            stock_data = data.current(stock,'price')
            stock_price = math.log10(stock_data)
            #if data count < max_lags
            if len((context.stocks)[ticker][1]) == max_lag:
                #pop first element
                ((context.stocks)[ticker][1]).pop(0)
                #append new data to end
                ((context.stocks)[ticker][1]).append(stock_price)
                #print "POP/Appended to " + ticker
            else:
                #append price
                ((context.stocks)[ticker][1]).append(stock_price)
    #for each pair
    pair_index = 0
    for pair in all_pairs:
        stock_x = pair[1][0]
        stock_y = pair[1][1]
 
        #if this trade is open
        trade_exists = False
        for n in range (0, len(trade_list[0])):
            if  ( (trade_list[0][n] == stock_x and trade_list[1][n] == stock_y) or (trade_list[0][n] == stock_y and trade_list[1][n] == stock_x)):
                trade_exists = True
                #check to see if it needs to be closed
                sell_signal(context, data, pair_index, n)
                break    
        #if this trade is not open
        if not trade_exists and len(trade_list[0]) < max_trades:    
            #look to see if meets criteria and if so buy
            buy_signal(context, data, pair_index)
        pair_index = pair_index + 1