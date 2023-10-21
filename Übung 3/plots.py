import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from mlxtend.preprocessing import minmax_scaling


class Plots:
    """
    A class for plotting data of a pandas dataframe.
    ------------------------------------------------

    Parameters:
    -----------
    - dataframe (pd.DataFrame): the dataset to be plotted

    Methods:
    --------
    * single_boxplot(): single boxplot
    * mutli_boxplot(): multiple grouped boxplots
    * line_plot(): line plot
    * scatter_plot(): scatter plot
    * single_histogram(): single histogram of one entry
    * correlation_heatmap(): linear correlation heatmap across all entries
    * correlation_data(): correlation data set
    * group_correlation(): grouped linear correlation heatmap
    * lmplot(): linear regression plot trough a scatter plot
    * scatter_heatmap(): scatter plot with a thrid dimensional entry to create a heatmap

    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _set_plot_style(self, **kwargs):
        """
        Set common styling elements for the plots

        Parameters:
        -----------
            **kwargs: Keyword arguments to set the style elements of the plot.
                      Supported arguments: title, title_fontsize, x_title, x_fontsize,
                      xticks_fontsize, y_title, y_fontsize, yticks_fontsize, length, height,
                      outliers (bool): Whether to show outliers in the plot.
        """
        plt.figure(figsize=(kwargs.get("length", 6), kwargs.get("height", 6)))
        plt.title(kwargs.get("title", ""))
        plt.xlabel(kwargs.get("x_title", ""))
        plt.ylabel(kwargs.get("y_title", ""))
        sns.set_theme()
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.despine(top=False, right=False, left=False, bottom=False)
        plt.tight_layout()

    def single_boxplot(self, x_value: str, outliers: bool = True, **kwargs):
        """
        Create a single box plot

        Parameters:
        -----------
            * x_value (str): The column name or variable to be plotted on the x-axis
            * outliers (bool, optional): Show outliers of the boxplot. Default is True
        """

        self._set_plot_style(**kwargs)
        sns.boxplot(self.dataframe[x_value], orient="v", showfliers=outliers)
        plt.show()

    def multi_boxplot(self, x_value: str, y_value: str, hue: str = None, outliers: bool = False, **kwargs):
        """
        Generate a multi-group boxplot.

        Parameters:
        -----------
            * x_value (str): The name of the column in the dataframe to be used as the x-axis variable
            * y_value (str): The name of the column in the dataframe to be used as the y-axis variable
            * hue (str, optional):  The name of the column in the dataframe used to group the data to create boxplots
            * outliers (bool, optional): Determines whether to show outliers in the boxplots. Defaults to False
        """
        self._set_plot_style(**kwargs)
        sns.boxplot(x=x_value, y=y_value, hue=hue, data=self.dataframe, showfliers=outliers)
        plt.ylim(kwargs.get("y_bot"), kwargs.get("y_top"))
        plt.show()

    def line_plot(self, x_value: str, y_value: str, **kwargs):
        """
        Create a line plot using the Seaborn library

        Parameters:
        -----------
            * x_value (str): The column name or variable to be plotted on the x-axis
            * y_value (str): The column name or variable to be plotted on the y-axis
        """
        self._set_plot_style(**kwargs)
        sns.lineplot(x=x_value, y=y_value, data=self.dataframe)
        plt.ylim(kwargs.get("y_bot"), kwargs.get("y_top"))
        plt.xlim(kwargs.get("x_bot"), kwargs.get("x_top"))
        plt.show()

    def scatter_plot(self, x_value: str, y_value: str, hue: str = None, **kwargs):
        """
        Create a scatter plot.

        Parameters:
        -----------
            * x_value (str): The column name or variable to be plotted on the x-axis
            * y_value (str): The column name or variable to be plotted on the y-axis
            * hue (str, optional): The column name or variable used for creating a legend. Defaults to None
        """

        self._set_plot_style(**kwargs)
        sns.scatterplot(x=x_value, y=y_value, data=self.dataframe, hue=hue)
        plt.ylim(kwargs.get("y_bot"), kwargs.get("y_top"))
        plt.xlim(kwargs.get("x_bot"), kwargs.get("x_top"))
        plt.show()

    def single_histogram(self, x_value: str, **kwargs):
        """
        Create a single histogram

        Parameters:
        -----------
            * x_value (str): The column name or variable to be plotted on the x-axis
        """

        self._set_plot_style(**kwargs)
        sns.histplot(x=x_value, data=self.dataframe)
        plt.xlim(kwargs.get("x_bot"), kwargs.get("x_top"))
        plt.show()

    def correlation_heatmap(self, corr_method: str = "pearson", round_factor: int = 2, **kwargs):
        """
        Create a linear correlation heatmap

        Parameters:
        -----------
            * corr_method (str, optional): correlation method. Defaults to "pearson"
            * round_factor (int, optional): round factor. Defaults to 2
        """
        data = self.dataframe.select_dtypes(exclude=object)
        cols = data.columns
        fit_data = minmax_scaling(data, columns=cols)
        data_corr = abs(fit_data.corr(method=corr_method).round(round_factor))

        self._set_plot_style(**kwargs)
        mask = np.triu(np.ones_like(fit_data.corr()))
        sns.heatmap(data_corr, annot=True, cmap="crest", mask=mask)
        plt.show()

    def correlation_data(self, corr_method: str = "pearson", round_factor: int = 2) -> pd.DataFrame:
        """
        Create a correlation dataframe

        Parameters:
        -----------
            * corr_method (str, optional): correlation method. Defaults to "pearson"
            * round_factor (int, optional): round factor. Defaults to 2

        Returns:
        --------
            pd.DataFrame: correlation data as a pandas dataframe
        """
        raw_data = self.dataframe
        data_columns = raw_data.columns
        fit_data = minmax_scaling(raw_data, columns=data_columns)
        data_corr = abs(fit_data.corr(method=corr_method).round(round_factor))
        return data_corr

    def group_correlation(self, columns: list, rows: list, corr_method="pearson", round_factor: int = 2, **kwargs):
        """
        Compute and visualize the correlation matrix for a specific group of columns

        Parameters:
        -----------
            * columns (list): The list of column names to include in the correlation matrix.
            * rows (list): The list of row names to exclude from the correlation matrix.
            * corr_method (str, optional): The method used to compute the correlation. Default is 'pearson'.
            * round_factor (int, optional): The number of decimal places to round the correlation values. Default is 2.
        """

        data = self.dataframe.filter(items=columns + rows)
        fit_data = minmax_scaling(data, columns=data.columns)
        data_corr = abs(fit_data.corr(method=corr_method).round(round_factor))

        data_corr = data_corr.drop(columns=rows)
        data_corr = data_corr[len(columns) :]
        self._set_plot_style(**kwargs)
        sns.heatmap(data_corr, annot=True, cmap="crest")
        plt.show()

    def lmplot(self, x_value: str, y_value: str, hue: str = None, **kwargs):
        """
        Scatter plot with regression line

        Parameters:
        -----------
            * x_value (str): Name of the column for the x-axis variable
            * y_value (str): Name of the column for the y-axis variable
            * hue (str, optional): Name of the column for the hue variable. Defaults to None

        """

        length = kwargs.get("length", 6)
        height = kwargs.get("height", 6)
        _, ax = plt.subplots(figsize=(length, height))

        # scatter plot
        if hue is not None:
            unique_hues = self.dataframe[hue].unique()
            for i in unique_hues:
                df_subset = self.dataframe.loc[self.dataframe[hue] == i]
                ax.scatter(df_subset[x_value], df_subset[y_value], label=i, s=kwargs.get("dot_size", 10))
        else:
            ax.scatter(self.dataframe[x_value], self.dataframe[y_value], s=kwargs.get("dot_size", 10))

        # regression line
        x = self.dataframe[x_value]
        y = self.dataframe[y_value]
        slope, intercept, r_value, _, _ = scipy.stats.linregress(x, y)
        ax.plot(x, slope * x + intercept, color="r")

        ax.set_xlabel(x_value)
        ax.set_ylabel(y_value)
        plt.ylim(kwargs.get("y_bot"), kwargs.get("y_top"))
        plt.xlim(kwargs.get("x_bot"), kwargs.get("x_top"))
        plt.tight_layout()

        if hue is not None:
            ax.legend()

        title = f"Linear fit: y = {slope:.2f}x + {intercept:.2f} | R-Value: {r_value:.2f}"
        plt.title(title)
        plt.show()

    def scatter_heatmap(self, x_value: str, y_value: str, z_value: str = None, **kwargs):
        """
        create scatter heatmap

        Parameters:
        -----------
            * x_value (str): Name of the column for the x-axis variable
            * y_value (str): Name of the column for the y-axis variable
            * z_value (str, optional): The name of the column representing the values for the heatmap. Defaults to None.
        """
        x = self.dataframe[x_value]
        y = self.dataframe[y_value]
        length = kwargs.get("length", 6)
        height = kwargs.get("height", 6)

        if z_value is None:
            fig, ax = plt.subplots(figsize=(length, height))
            scatter = ax.scatter(x, y, s=kwargs.get("scatter_size", 1), c="black")
        else:
            z = self.dataframe[z_value]
            fig, ax = plt.subplots(figsize=(length, height))
            scatter = ax.scatter(x, y, s=kwargs.get("heatmap_size", 30), c=z, cmap=kwargs.get("cmap", "YlOrRd"))
            cbar = fig.colorbar(scatter)
            cbar.set_label(z_value)

        ax.set_xlabel(x_value)
        ax.set_ylabel(y_value)
        ax.set_title(kwargs.get("title", ""))

        plt.show()
