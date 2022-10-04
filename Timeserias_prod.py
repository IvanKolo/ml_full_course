# Import necessary packages
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import earthpy as et

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Use white grid plot background from seaborn
sns.set(font_scale=1.5, style="whitegrid")

# Download the data
data = et.data.get_data('colorado-flood')

# Set working directory
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

# Define relative path to file with hourly precip
file_path = os.path.join("data", "colorado-flood",
                         "precipitation",
                         "805325-precip-daily-2003-2013.csv")

# Import data using datetime and no data value
precip_2003_2013_hourly = pd.read_csv(file_path,
                                      parse_dates=['DATE'],
                                      index_col=['DATE'],
                                      na_values=['999.99'])

# View first few rows
first_rows = precip_2003_2013_hourly.head()
print(first_rows)

# View dataframe info
info_by_data = precip_2003_2013_hourly.info()

print(info_by_data)


# View summary statistics
summary_stat = precip_2003_2013_hourly.describe()

print(summary_stat)

# View index values of dataframe
index = precip_2003_2013_hourly.index

# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(precip_2003_2013_hourly.index.values,
           precip_2003_2013_hourly['HPCP'],
           color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitation (inches)",
       title="Hourly Precipitation - Boulder Station\n 2003-2013")

plt.show()

# Resample to daily precip sum and save as new dataframe
precip_2003_2013_daily = precip_2003_2013_hourly.resample('D').sum()

print(precip_2003_2013_daily)


# # Set date column as index
# precip_hourly_index = precip_hourly.set_index('DATE')
#
# # Resample to daily sum of precip
# precip_daily = precip_hourly_index.resample('D').sum()


# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(precip_2003_2013_daily.index.values,
           precip_2003_2013_daily['HPCP'],
           color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitation (inches)",
       title="Daily Precipitation - Boulder Station\n 2003-2013")

plt.show()

# Resample to monthly precip sum and save as new dataframe
precip_2003_2013_monthly = precip_2003_2013_daily.resample('M').sum()

print(precip_2003_2013_monthly)

# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(precip_2003_2013_monthly.index.values,
           precip_2003_2013_monthly['HPCP'],
           color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitation (inches)",
       title="Monthly Precipitation - Boulder Station\n 2003-2013")

plt.show()

# Resample to monthly precip sum and save as new dataframe
precip_2003_2013_yearly = precip_2003_2013_monthly.resample('Y').sum()
# precip_2003_2013_yearly = precip_2003_2013_hourly.resample('Y').sum()
print(precip_2003_2013_yearly)


# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.scatter(precip_2003_2013_hourly.resample('Y').sum().index.values,
           precip_2003_2013_hourly.resample('Y').sum()['HPCP'],
           color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Precipitation (inches)",
       title="Yearly Precipitation - Boulder Station\n 2003-2013")

plt.show()