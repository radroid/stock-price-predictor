import pandas as pd


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