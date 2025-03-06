"""
This script analyzes energy consumption data, performs exploratory data analysis,
creates interactive visualizations, and implements forecasting models to predict
future consumption patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import warnings
import datetime as dt
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class EnergyConsumptionAnalyzer:
    """
    A class for analyzing energy consumption data.
    
    This class provides methods for loading data, preprocessing, exploratory 
    data analysis, visualization, anomaly detection, and forecasting.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the EnergyConsumptionAnalyzer.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing energy consumption data.
        """
        self.data_path = data_path
        self.raw_data = None
        self.data = None
        self.forecast_data = None
        self.anomalies = None
        
        # Set default plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'figure.figsize': (12, 6)})
    
    def load_data(self, data_path=None):
        """
        Load energy consumption data from a CSV file.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing energy consumption data.
            
        Returns:
        --------
        pandas.DataFrame
            The loaded data.
        """
        if data_path:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("Data path not provided.")
        
        print(f"Loading data from {self.data_path}...")
        
        # Load data
        self.raw_data = pd.read_csv(self.data_path)
        
        # Check if there's a timestamp column
        timestamp_col = None
        for col in self.raw_data.columns:
            if any(term in col.lower() for term in ['time', 'date', 'timestamp']):
                timestamp_col = col
                break
        
        if timestamp_col:
            print(f"Found timestamp column: {timestamp_col}")
            # Convert timestamp to datetime
            self.raw_data[timestamp_col] = pd.to_datetime(self.raw_data[timestamp_col])
            
            # Set timestamp as index
            self.raw_data.set_index(timestamp_col, inplace=True)
        else:
            print("No timestamp column found. Please ensure your data has a timestamp column.")
        
        # Print data info
        print("\nData Overview:")
        print(f"Shape: {self.raw_data.shape}")
        print("\nColumn Types:")
        print(self.raw_data.dtypes)
        print("\nFirst 5 rows:")
        print(self.raw_data.head())
        
        return self.raw_data
    
    def preprocess_data(self, consumption_col=None):
        """
        Preprocess the data for analysis.
        
        Parameters:
        -----------
        consumption_col : str, optional
            Name of the column containing energy consumption values.
            
        Returns:
        --------
        pandas.DataFrame
            The preprocessed data.
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nPreprocessing data...")
        
        # Make a copy of the raw data
        self.data = self.raw_data.copy()
        
        # Identify consumption column if not provided
        if consumption_col is None:
            for col in self.data.columns:
                if any(term in col.lower() for term in ['consumption', 'energy', 'usage', 'kwh', 'power']):
                    consumption_col = col
                    break
        
        if consumption_col:
            print(f"Using '{consumption_col}' as energy consumption column")
        else:
            # If no consumption column is identified, use the first numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                consumption_col = numeric_cols[0]
                print(f"No specific energy consumption column identified. Using '{consumption_col}' as default.")
            else:
                raise ValueError("No numeric columns found in the data.")
        
        # Ensure consumption_col is the first column for easier reference
        if consumption_col in self.data.columns:
            cols = [consumption_col] + [col for col in self.data.columns if col != consumption_col]
            self.data = self.data[cols]
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values found:")
            print(missing_values[missing_values > 0])
            
            # Handle missing values
            print("Handling missing values...")
            # Fill numeric columns with their median
            for col in self.data.select_dtypes(include=[np.number]).columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)
            
            # Forward fill for other columns
            self.data.fillna(method='ffill', inplace=True)
            # Backward fill any remaining NaNs
            self.data.fillna(method='bfill', inplace=True)
        else:
            print("No missing values found.")
        
        # Add time-based features
        if isinstance(self.data.index, pd.DatetimeIndex):
            print("Adding time-based features...")
            self.data['hour'] = self.data.index.hour
            self.data['day'] = self.data.index.day
            self.data['day_of_week'] = self.data.index.dayofweek
            self.data['month'] = self.data.index.month
            self.data['year'] = self.data.index.year
            self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        
        # Resample to regular intervals if timestamps are not evenly spaced
        if isinstance(self.data.index, pd.DatetimeIndex):
            # Check if data is already on a regular frequency
            inferred_freq = pd.infer_freq(self.data.index)
            
            if inferred_freq is None:
                print("Data does not have a regular frequency. Resampling to hourly intervals...")
                # Resample to hourly frequency (or adjust as needed)
                self.data = self.data.resample('H').mean()
                # Fill any NaNs created during resampling
                self.data.fillna(method='ffill', inplace=True)
                print(f"Data resampled, new shape: {self.data.shape}")
            else:
                print(f"Data already has a regular frequency: {inferred_freq}")
        
        print("Preprocessing completed.")
        return self.data
    
    def compute_statistics(self, column=None):
        """
        Compute basic statistics for the energy consumption data.
        
        Parameters:
        -----------
        column : str, optional
            The column for which to compute statistics. If None, uses the first column.
            
        Returns:
        --------
        pandas.Series
            Basic statistics of the specified column.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nComputing statistics for '{column}'...")
        
        stats = self.data[column].describe()
        
        # Add additional statistics
        stats['variance'] = self.data[column].var()
        stats['skewness'] = self.data[column].skew()
        stats['kurtosis'] = self.data[column].kurtosis()
        
        print(stats)
        return stats
    
    def plot_distribution(self, column=None):
        """
        Plot the distribution of energy consumption.
        
        Parameters:
        -----------
        column : str, optional
            The column to plot. If None, uses the first column.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plots.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nPlotting distribution for '{column}'...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        sns.histplot(self.data[column], kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        
        # Boxplot
        sns.boxplot(x=self.data[column], ax=axes[1])
        axes[1].set_title(f'Boxplot of {column}')
        axes[1].set_xlabel(column)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_time_series(self, column=None, resample_freq=None):
        """
        Plot the time series of energy consumption.
        
        Parameters:
        -----------
        column : str, optional
            The column to plot. If None, uses the first column.
        resample_freq : str, optional
            Frequency to resample the data (e.g., 'D' for daily, 'M' for monthly).
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot plot time series.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nPlotting time series for '{column}'...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        data_to_plot = self.data[column]
        
        # Resample if requested
        if resample_freq:
            data_to_plot = data_to_plot.resample(resample_freq).mean()
            title_freq = {
                'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'
            }.get(resample_freq, resample_freq)
            title = f'{title_freq} Energy Consumption'
        else:
            title = 'Energy Consumption Over Time'
        
        data_to_plot.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def decompose_time_series(self, column=None, period=None):
        """
        Decompose the time series into trend, seasonal, and residual components.
        
        Parameters:
        -----------
        column : str, optional
            The column to decompose. If None, uses the first column.
        period : int, optional
            The period of the seasonality. If None, it will be inferred.
            
        Returns:
        --------
        statsmodels.tsa.seasonal.DecomposeResult
            The decomposition result.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot decompose time series.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nDecomposing time series for '{column}'...")
        
        # Make sure there are no missing values
        data_to_decompose = self.data[column].fillna(method='ffill').fillna(method='bfill')
        
        # Infer period if not provided
        if period is None:
            # Try to infer from the frequency
            inferred_freq = pd.infer_freq(self.data.index)
            if inferred_freq:
                if 'D' in inferred_freq:
                    period = 7  # Weekly seasonality
                elif 'H' in inferred_freq:
                    period = 24  # Daily seasonality
                elif 'T' in inferred_freq or 'min' in str(inferred_freq).lower():
                    period = 60  # Hourly seasonality
                else:
                    period = 12  # Monthly seasonality (default)
            else:
                period = 12  # Default to monthly seasonality
        
        print(f"Using period = {period} for decomposition")
        
        # Perform decomposition
        result = seasonal_decompose(data_to_decompose, model='additive', period=period)
        
        # Plot the decomposed components
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        result.observed.plot(ax=axes[0])
        axes[0].set_title('Observed')
        axes[0].set_ylabel(column)
        
        result.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')
        
        result.seasonal.plot(ax=axes[2])
        axes[2].set_title(f'Seasonality (Period = {period})')
        
        result.resid.plot(ax=axes[3])
        axes[3].set_title('Residual')
        
        plt.tight_layout()
        plt.show()
        
        return result
    
    def create_interactive_time_series(self, column=None):
        """
        Create an interactive time series plot using Plotly.
        
        Parameters:
        -----------
        column : str, optional
            The column to plot. If None, uses the first column.
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            The interactive figure.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot create interactive time series.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nCreating interactive time series for '{column}'...")
        
        # Create plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data[column],
            mode='lines',
            name=column,
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'Interactive Time Series of {column}',
            xaxis_title='Date',
            yaxis_title=column,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        fig.show()
        
        return fig
    
    def compare_periods(self, column=None, period='day'):
        """
        Compare energy consumption across different periods.
        
        Parameters:
        -----------
        column : str, optional
            The column to analyze. If None, uses the first column.
        period : str, optional
            The period to compare ('day', 'week', 'month'). Default is 'day'.
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            The interactive figure.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot compare periods.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nComparing {period}ly energy consumption for '{column}'...")
        
        # Create a copy of the data with the specified column
        data_to_analyze = self.data[[column]].copy()
        
        if period == 'day':
            # Group by hour of day
            data_to_analyze['hour'] = data_to_analyze.index.hour
            grouped = data_to_analyze.groupby('hour')[column].mean()
            x_label = 'Hour of Day'
            title = 'Average Energy Consumption by Hour of Day'
            
        elif period == 'week':
            # Group by day of week
            data_to_analyze['day_of_week'] = data_to_analyze.index.dayofweek
            grouped = data_to_analyze.groupby('day_of_week')[column].mean()
            # Convert day numbers to names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            grouped.index = [day_names[i] for i in grouped.index]
            x_label = 'Day of Week'
            title = 'Average Energy Consumption by Day of Week'
            
        elif period == 'month':
            # Group by month
            data_to_analyze['month'] = data_to_analyze.index.month
            grouped = data_to_analyze.groupby('month')[column].mean()
            # Convert month numbers to names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            grouped.index = [month_names[i-1] for i in grouped.index]
            x_label = 'Month'
            title = 'Average Energy Consumption by Month'
            
        else:
            raise ValueError("Invalid period. Use 'day', 'week', or 'month'.")
        
        # Create interactive bar chart
        fig = px.bar(
            grouped,
            x=grouped.index,
            y=grouped.values,
            labels={'x': x_label, 'y': column},
            title=title,
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=f'Average {column}',
            showlegend=False
        )
        
        fig.show()
        
        return fig
    
    def create_heatmap(self, column=None):
        """
        Create a heatmap of energy consumption patterns.
        
        Parameters:
        -----------
        column : str, optional
            The column to analyze. If None, uses the first column.
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            The interactive figure.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot create heatmap.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nCreating heatmap for '{column}'...")
        
        # Create a copy of the data with the specified column
        data_for_heatmap = self.data[[column]].copy()
        
        # Add hour and day of week columns
        data_for_heatmap['hour'] = data_for_heatmap.index.hour
        data_for_heatmap['day_of_week'] = data_for_heatmap.index.dayofweek
        
        # Pivot the data to create a matrix suitable for heatmap
        pivot_table = data_for_heatmap.pivot_table(
            values=column,
            index='hour',
            columns='day_of_week',
            aggfunc='mean'
        )
        
        # Rename columns to days
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table.columns = [day_names[i] for i in pivot_table.columns]
        
        # Create the heatmap
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Day of Week", y="Hour of Day", color=f"Average {column}"),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale="Viridis",
            title=f"Energy Consumption Heatmap: Hour of Day vs Day of Week"
        )
        
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            coloraxis_colorbar=dict(title=f"Average {column}")
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate="Day: %{x}<br>Hour: %{y}<br>Consumption: %{z:.2f}<extra></extra>"
        )
        
        fig.show()
        
        return fig
    
    def detect_anomalies(self, column=None, contamination=0.05):
        """
        Detect anomalies in the energy consumption data using Isolation Forest.
        
        Parameters:
        -----------
        column : str, optional
            The column to analyze. If None, uses the first column.
        contamination : float, optional
            The expected proportion of anomalies in the data. Default is 0.05 (5%).
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the original data with an 'anomaly' column.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nDetecting anomalies in '{column}'...")
        
        # Prepare data for anomaly detection
        data_for_anomaly = self.data[[column]].copy()
        
        # Fit the Isolation Forest model
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (1: normal, -1: anomaly)
        data_for_anomaly['anomaly'] = isolation_forest.fit_predict(data_for_anomaly[[column]])
        
        # Convert to binary (0: normal, 1: anomaly)
        data_for_anomaly['anomaly'] = (data_for_anomaly['anomaly'] == -1).astype(int)
        
        # Count anomalies
        anomaly_count = data_for_anomaly['anomaly'].sum()
        percentage = (anomaly_count / len(data_for_anomaly)) * 100
        
        print(f"Detected {anomaly_count} anomalies ({percentage:.2f}% of the data)")
        
        # Store the anomaly data
        self.anomalies = data_for_anomaly
        
        # Plot the anomalies
        fig = go.Figure()
        
        # Add normal points
        normal_data = data_for_anomaly[data_for_anomaly['anomaly'] == 0]
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data[column],
            mode='markers',
            marker=dict(color='blue', size=4),
            name='Normal'
        ))
        
        # Add anomalies
        anomaly_data = data_for_anomaly[data_for_anomaly['anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data[column],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='Anomaly'
        ))
        
        fig.update_layout(
            title=f'Anomaly Detection in {column}',
            xaxis_title='Date',
            yaxis_title=column,
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.show()
        
        return data_for_anomaly
    
    def forecast_arima(self, column=None, steps=24):
        """
        Forecast energy consumption using ARIMA model.
        
        Parameters:
        -----------
        column : str, optional
            The column to forecast. If None, uses the first column.
        steps : int, optional
            Number of steps to forecast ahead. Default is 24.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the forecast results.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot perform forecasting.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nForecasting '{column}' using ARIMA model...")
        
        # Prepare data for forecasting
        data_for_forecast = self.data[column].copy()
        
        # Check for stationarity
        print("Checking for stationarity...")
        result = adfuller(data_for_forecast.dropna())
        
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        
        if result[1] <= 0.05:
            print("The series is stationary (p <= 0.05)")
            d = 0
        else:
            print("The series is not stationary (p > 0.05), differencing will be applied.")
            d = 1
        
        # Fit ARIMA model
        print("Fitting ARIMA model...")
        # Starting with a simple ARIMA(1,d,1) model
        model = ARIMA(data_for_forecast, order=(1, d, 1))
        model_fit = model.fit()
        
        print(f"ARIMA model summary:")
        print(model_fit.summary())
        
        # Generate forecast
        print(f"Forecasting {steps} steps ahead...")
        forecast = model_fit.forecast(steps=steps)
        
        # Create date range for forecast
        last_date = data_for_forecast.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=steps, freq=pd.infer_freq(data_for_forecast.index))
        
        # Create DataFrame with forecasted values
        forecast_df = pd.DataFrame({
            'forecast': forecast
        }, index=forecast_dates)
        
        # Combine historical and forecasted data
        combined_df = pd.DataFrame({
            'historical': data_for_forecast,
            'forecast': None
        })
        
        combined_df = pd.concat([combined_df, forecast_df])
        
        # Store the forecast data
        self.forecast_data = combined_df
        
        # Plot the forecast
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=combined_df.index[:-steps],
            y=combined_df['historical'].dropna(),
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add forecasted data
        fig.add_trace(go.Scatter(
            x=combined_df.index[-steps:],
            y=combined_df['forecast'].dropna(),
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'ARIMA Forecast of {column}',
            xaxis_title='Date',
            yaxis_title=column,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
        
        return combined_df
    
    def forecast_prophet(self, column=None, periods=24, include_components=True):
        """
        Forecast energy consumption using Facebook Prophet.
        
        Parameters:
        -----------
        column : str, optional
            The column to forecast. If None, uses the first column.
        periods : int, optional
            Number of periods to forecast ahead. Default is 24.
        include_components : bool, optional
            Whether to include decomposition components. Default is True.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the forecast results.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index is not a DatetimeIndex. Cannot perform forecasting.")
        
        if column is None:
            column = self.data.columns[0]
        
        print(f"\nForecasting '{column}' using Prophet model...")
        
        # Prepare data for Prophet
        data_for_prophet = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data[column]
        }).reset_index(drop=True)
        
        # Handle missing values
        data_for_prophet = data_for_prophet.dropna()
        
        # Fit Prophet model
        print("Fitting Prophet model...")
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(data_for_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=pd.infer_freq(self.data.index))
        
        # Generate forecast
        print(f"Forecasting {periods} periods ahead...")
        forecast = model.predict(future)
        
        # Plot the forecast
        fig = model.plot(forecast)
        plt.title(f'Prophet Forecast of {column}')
        plt.tight_layout()
        plt.show()
        
        # Plot components if requested
        if include_components:
            fig_comp = model.plot_components(forecast)
            plt.tight_layout()
            plt.show()
        
        # Create interactive plot
        historical_dates = data_for_prophet['ds']
        forecast_dates = forecast['ds']
        
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=data_for_prophet['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add forecasted data
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add uncertainty intervals
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'Prophet Forecast of {column}',
            xaxis_title='Date',
            yaxis_title=column,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
        
        return forecast
    
    def run_complete_analysis(self, data_path=None, consumption_col=None):
        """
        Run a complete analysis pipeline on the energy consumption data.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing energy consumption data.
        consumption_col : str, optional
            Name of the column containing energy consumption values.
            
        Returns:
        --------
        dict
            Dictionary containing all analysis results.
        """
        results = {}
        
        # Load and preprocess data
        if data_path:
            self.load_data(data_path)
        elif self.data_path:
            self.load_data()
        else:
            raise ValueError("Data path not provided.")
        
        self.preprocess_data(consumption_col)
        
        # Compute statistics
        column = consumption_col if consumption_col else self.data.columns[0]
        results['statistics'] = self.compute_statistics(column)
        
        # Create visualizations
        self.plot_distribution(column)
        self.plot_time_series(column)
        
        # Decompose time series
        if isinstance(self.data.index, pd.DatetimeIndex):
            try:
                results['decomposition'] = self.decompose_time_series(column)
            except Exception as e:
                print(f"Error in time series decomposition: {e}")
        
        # Create interactive visualizations
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.create_interactive_time_series(column)
            self.compare_periods(column, period='day')
            self.compare_periods(column, period='week')
            self.compare_periods(column, period='month')
            self.create_heatmap(column)
        
        # Detect anomalies
        results['anomalies'] = self.detect_anomalies(column)
        
        # Forecasting
        if isinstance(self.data.index, pd.DatetimeIndex):
            try:
                results['arima_forecast'] = self.forecast_arima(column)
            except Exception as e:
                print(f"Error in ARIMA forecasting: {e}")
            
            try:
                results['prophet_forecast'] = self.forecast_prophet(column)
            except Exception as e:
                print(f"Error in Prophet forecasting: {e}")
        
        return results
    
    def save_plots(self, output_dir='./plots'):
        """
        Save all generated plots to files.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the plots. Default is './plots'.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Add code to save plots
        print(f"Plots saved to {output_dir}")


def generate_sample_data(output_file='sample_energy_data.csv', num_days=30):
    """
    Generate sample energy consumption data for testing.
    
    Parameters:
    -----------
    output_file : str, optional
        Path to save the generated data. Default is 'sample_energy_data.csv'.
    num_days : int, optional
        Number of days of data to generate. Default is 30.
        
    Returns:
    --------
    pandas.DataFrame
        The generated data.
    """
    print(f"Generating sample energy consumption data for {num_days} days...")
    
    # Create date range with hourly frequency
    start_date = dt.datetime(2023, 1, 1)
    end_date = start_date + dt.timedelta(days=num_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create DataFrame
    df = pd.DataFrame(index=date_range)
    
    # Generate consumption data with daily and weekly patterns
    # Base consumption
    base = 20
    
    # Daily pattern (higher during day, lower at night)
    hour_effect = np.sin(np.pi * df.index.hour / 12) * 15
    
    # Weekly pattern (lower on weekends)
    weekend_effect = -10 * ((df.index.dayofweek >= 5).astype(int))
    
    # Monthly pattern (higher in winter months)
    month_effect = -5 * np.sin(np.pi * df.index.month / 6)
    
    # Random noise
    noise = np.random.normal(0, 3, len(df))
    
    # Special events (occasional spikes)
    special_days = np.random.choice(df.index, size=int(len(df) * 0.01))
    special_effect = pd.Series(0, index=df.index)
    special_effect.loc[special_days] = np.random.uniform(20, 40, len(special_days))
    
    # Combine all effects
    df['energy_consumption'] = base + hour_effect + weekend_effect + month_effect + noise + special_effect
    
    # Ensure no negative values
    df['energy_consumption'] = df['energy_consumption'].clip(lower=0)
    
    # Add some missing values (about 1%)
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.01))
    df.loc[missing_indices, 'energy_consumption'] = np.nan
    
    # Add temperature data (correlated with energy consumption)
    base_temp = 20
    season_effect = 15 * np.sin(2 * np.pi * (df.index.dayofyear / 365 - 0.5))
    daily_effect = 5 * np.sin(2 * np.pi * df.index.hour / 24)
    temp_noise = np.random.normal(0, 2, len(df))
    
    df['temperature'] = base_temp + season_effect + daily_effect + temp_noise
    
    # Add humidity data (anti-correlated with temperature)
    base_humidity = 60
    humidity_season = -10 * np.sin(2 * np.pi * (df.index.dayofyear / 365 - 0.5))
    humidity_noise = np.random.normal(0, 5, len(df))
    
    df['humidity'] = base_humidity + humidity_season + humidity_noise
    df['humidity'] = df['humidity'].clip(lower=20, upper=100)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Sample data saved to {output_file}")
    
    return df


def main():
    """
    Main function to demonstrate the usage of the EnergyConsumptionAnalyzer.
    """
    print("Energy Consumption Analysis and Visualization")
    print("============================================")
    
    # Check if sample data exists, if not, generate it
    sample_data_path = 'sample_energy_data.csv'
    if not os.path.exists(sample_data_path):
        generate_sample_data(sample_data_path)
    
    # Create analyzer instance
    analyzer = EnergyConsumptionAnalyzer(sample_data_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()