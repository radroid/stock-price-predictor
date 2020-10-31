import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import os
from statsmodels.tsa.stattools import adfuller


def plot_data(df: pd.DataFrame, stock_name: str = 'Stock', columns: list = None,
              ax: object = None, title: str = None,
              ylabel: str = 'Close Price', legend_loc=0,
              **kwargs_plot):
    """Plot stocks data.

    Args:
        df (pd.DataFrame): DataFrame containing all the data.
        stock_name (str, optional): name of the stock whose data is being plotted.
                                    Defaults to 'Stock'.
        columns (list, optional): list of columns in the DataFrame to be plotted.
                                  Defaults to None. All columns in the DataFrame
                                  are used if it is None.
        ax (object, optional): a Matplotlib axis object on which the plot
                               is to be plotted on. If None, a new axis
                               will be created and returned.
                               Defaults to None.
        title (str, optional): Specific title for the plot. If None, a
                               predefined title with the stock name will be
                               used. Defaults to None.
        ylabel (str, optional): Specific ylabel for the plot.
                                Defaults to 'Close Price'.
        legend_loc (int, optional): Specific location of the legend on the
                                    plot. Defaults to 0. (0 = 'Best')

    Returns:
        plt.axes: axis on which the plot is created.
    """
    df = df.copy()
    
    if columns is None:
        columns = list(df.columns)

    if ax is None:
        ax = df[columns].plot(**kwargs_plot)
    else:
        df[columns].plot(ax=ax, **kwargs_plot)

    if title is None:
        title = f"Variation in {stock_name}"

    ax.set_xlabel("Timestamp",
                  fontdict={"size": 16})
    ax.set_ylabel(ylabel,
                  fontdict={"size": 16})
    ax.set_title(title,
                 fontdict={"fontsize": 24, "fontweight": "bold"},
                 pad=2)

    leg = ax.legend(fontsize=16, frameon=True, loc=legend_loc)
    leg.get_frame().set_color('#F2F2F2')
    leg.get_frame().set_edgecolor('black')
    leg.set_title("Legend", prop={"size": 20, "weight": 'bold'})

    return ax


def stationarity_stats(time_series: pd.Series, window_size: int = 10):
    """The function plots and prints the necessary statistics needed to determine if a
    series can be assumed to be stationary.
    
    Args:
        time_series (pd.Series): series that is to be tested for stationarity.
        window_size (int): rolling average and standard deviation window size. Defaults to 10.
    """
    
    #Determing rolling statistics
    ts_moving_avg = time_series.rolling(window_size).mean()
    ts_moving_std = time_series.rolling(window_size).std()

    #Plot statistics data
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.style.use('seaborn')
    
    time_series.plot(color='blue',
                     ax=ax,
                     label='Time Series')
    ts_moving_avg.plot(color='yellow',
                       ax=ax,
                       label='Moving Average')
    ts_moving_std.plot(color='green',
                       ax=ax,
                       label='Standard Deviation')
    plt.legend(loc='best')
    ax.set_title('Time Series with Moving Avearage & Rolling Standard Deviation',
                 fontdict={'size': 20, 'weight': 'bold'})
    ax.set_xlabel('Timestamp', fontdict={"size": 14})
    ax.set_ylabel('Price ($)', fontdict={"size": 14})
    
    leg = ax.legend(fontsize=16, frameon=True)
    leg.get_frame().set_color('#F2F2F2')
    leg.get_frame().set_edgecolor('black')
    leg.set_title("Legend", prop={"size": 20, "weight": 'bold'})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    
    #Calculate resutls of the Dickey-Fuller test
    test_results = adfuller(time_series, autolag='AIC')
    data = ['Test Statistic',
            'p-value',
            'Number of Lags Used',
            'Number of Observations Used']
    for name, value in zip(data, test_results[:4]):
        print(f'{name}: {value: .4f}')

    for key, value in test_results[4].items():
        print(f'{key} Critical Value: {value: .3f}')
    
    return test_results


def plot_acf_pacf(data_acf: list, data_pacf: list):
    # plot data
    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(20,8))

    plot_titles = ['Autocorreleration Function',
                   'Partial Autocorreleration Function']
    # ax_pacf.plot(data_pacf)

    for ax, title, plot_data in zip(axes, plot_titles, (data_acf, data_pacf)):
        ax.bar(x=range(len(plot_data)), height=plot_data, width=0.1)
        ax.axhline(y=0,linestyle='--',color='gray')
        ax.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='-',color='gray')
        ax.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='-',color='gray')
        ax.set_title(title, fontdict={'size': 20, 'weight': 'bold'})
        ax.set_xlabel('Lag Value', fontdict={"size": 14})

    fig.suptitle("Autocorrelation Plots of Log of Time Series and First Degree of Differencing",
                  fontsize=24, fontweight= 'bold');
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def evaluate_model(train_ts, y_true, y_preds):
    """The function plots the results obtained on a graph, calculates metric values and returns it.
    
    Args:
        train_ts (pd.Series): the series the model was trained on.
        y_true (pd.Series): true values of the time series.
        y_preds (pd.Series): predicted values of the time series.
        
    Returns:
        dict: with the values of all the metrics printed.
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    y_true.plot(ax=ax,
                color='green',
                label='True Series')
    y_preds.plot(ax=ax,
                 color='blue',
                 label='Predicted Series')
    
    plt.legend(loc='best')
    ax.set_title('Time Series Predictions',
                 fontdict={'size': 20, 'weight': 'bold'})
    ax.set_xlabel('Timestamp', fontdict={"size": 14})
    ax.set_ylabel('Price ($)', fontdict={"size": 14})
    
    leg = ax.legend(fontsize=16, frameon=True)
    leg.get_frame().set_color('#F2F2F2')
    leg.get_frame().set_edgecolor('black')
    leg.set_title("Legend", prop={"size": 20, "weight": 'bold'})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # report performance
    performance_dict = {
        "mae": mean_absolute_error(y_true, y_preds),
        "mse": mean_squared_error(y_true, y_preds),
        "rmse": math.sqrt(mean_squared_error(y_true, y_preds)),
        "mape": np.mean(np.abs((y_preds - y_true)/y_true))
    }
    
    print("######### Metric Scores ##########\n"
          "Mean Absolute Error:            {mae: .2f}\n"
          "Mean Squared Error:             {mse: .2f}\n"
          "Root Mean Squared Error:        {rmse: .2f}\n"
          "Mean Absolute Percentage Error: {mape: 0.2%}\n".format(**performance_dict))

    return performance_dict


def create_time_series(data: pd.Series, series_length_years: int, 
                       start_date: datetime.datetime = None,
                       last_date: datetime.datetime = None,
                       equal_series: bool = True):
    """Creates a list of time series of the given length from the data provided.
    Args:
        data (pd.Series): pandas series with all the data, indexed with the timestamp.
        series_length_years (int): length of each series in the list to be created.
        start_date (datetime.datetime, optional): 
            Date the first time series should start from. If None, '2002-01-01' is used.
            Defaults to None.
        last_date (datetime.datetime, optional):
            Date the last time series should end on. If None, last date found in `data` will be used.
            Defaults to None.
        equal_series (bool): if True, all series created will be of equal length. last series created
                             will be removed if it is shorter than the others.
    
    Returns:
        (list of pd.Series): python list containing all the time series created.
    
    """
    # Updating dictionaries.
    if start_date is None:
        start_date = datetime.datetime(2002, 1, 1)
    if last_date is None:
        last_date = max(data.index.date)
    else:
        last_date = last_date.date()
    
    start_date = start_date.date()
    time_series_list = []

    while start_date < last_date:
        end_date = start_date + pd.DateOffset(years=series_length_years) - pd.DateOffset(days=1)
        time_series_list.append(data.loc[start_date:end_date])
        start_date = end_date + pd.DateOffset(days=1)
        
    is_last_equal = str(max(time_series_list[0].index.date))[5:10] == str(max(time_series_list[-1].index.date))[5:10]
    
    if equal_series and not is_last_equal:
        time_series_list = time_series_list[:-1]
    
    print(f'Number of series created: {len(time_series_list)}')
    print(f'Last series removed: {not is_last_equal}')
    print(f'Last series end date: {max(time_series_list[-1].index.date)}')
    return time_series_list


def create_training_series(time_series_list: list,
                           prediction_length_months: int):
    """Create a training series using the prediction length provided in months.
    
    Args:
        time_series_list (list): list of pandas series each of equal length.
        prediciton_length_months (int): number of months a prediction is to be made. This will be
                                        used to created a training series, i.e. context length for
                                        DeepAR algorithm to train on.
    
    Returns:
        list of pd.Series: python list containing the training time series.
    """
    training_series_list = []
    
    for ts in time_series_list:
        end = max(ts.index.date) - pd.DateOffset(months=prediction_length_months)
        training_series_list.append(ts[:end])

    print(f'Number of series updated: {len(training_series_list)}')
    return training_series_list