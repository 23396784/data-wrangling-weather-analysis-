# 🌤️ Data Wrangling & Meteorological Analysis: NYC Airport Weather Patterns

A comprehensive data wrangling and statistical analysis project examining hourly meteorological data from New York City's three major airports throughout 2013.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Analysis-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project demonstrates advanced **data wrangling** techniques applied to real-world meteorological data from the `nycflights13` dataset. The analysis covers 26,130 hourly weather observations from JFK, LaGuardia (LGA), and Newark (EWR) airports, focusing on wind patterns, seasonal variations, and cross-airport comparisons.

### Key Skills Demonstrated:
- 🔧 **Data Extraction** - Decompressing .gz files and parsing CSV
- 🧹 **Data Cleaning** - Handling missing values, unit conversions
- 📊 **Outlier Detection** - IQR method for wind speed anomalies
- 📈 **Time Series Analysis** - Monthly aggregations and trends
- 🔄 **Unit Conversion** - Imperial to metric transformations
- 📊 **Multi-location Comparison** - Cross-airport statistical analysis

## 🎯 Project Objectives

1. Extract and parse compressed meteorological data
2. Clean data by removing invalid entries and handling NaN values
3. Convert units from Imperial to Metric system
4. Detect and analyze wind speed outliers
5. Calculate monthly mean wind speeds per airport
6. Compare seasonal patterns across locations
7. Provide actionable insights for airline operations

## 📊 Dataset Information

### Source
**nycflights13 Weather Dataset**
- Period: January - December 2013
- Frequency: Hourly observations
- Locations: 3 major NYC airports

### Data Dimensions

| Metric | Value |
|--------|-------|
| Total Observations | 26,130 |
| Variables | 15 |
| Airports | 3 (JFK, LGA, EWR) |
| Time Span | 8,760 hours/airport |

### Variables (Columns)

| Column | Description | Original Unit | Converted Unit |
|--------|-------------|---------------|----------------|
| origin | Airport code | - | - |
| year | Year of recording | - | - |
| month | Month (1-12) | - | - |
| day | Day of month | - | - |
| hour | Hour (0-23) | - | - |
| temp | Temperature | °F | °C |
| dewp | Dew point | °F | °C |
| humid | Relative humidity | % | % |
| wind_dir | Wind direction | degrees | degrees |
| wind_speed | Wind speed | mph | m/s |
| wind_gust | Wind gust speed | mph | m/s |
| precip | Precipitation | inches | mm |
| pressure | Sea level pressure | millibars | millibars |
| visib | Visibility | miles | meters |

### Airport Information

| Code | Airport Name | Location |
|------|--------------|----------|
| **JFK** | John F. Kennedy International | Queens, NY |
| **LGA** | LaGuardia Airport | Queens, NY |
| **EWR** | Newark Liberty International | Newark, NJ |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/data-wrangling-weather-analysis.git
cd data-wrangling-weather-analysis
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
import gzip
import pandas as pd

# Extract compressed data
with gzip.open('nycflights13_weather.csv.gz', 'rb') as f_in:
    with open('nycflights13_weather.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load data
weather = np.genfromtxt('nycflights13_weather.csv', 
                        delimiter=',',
                        skip_header=1)

print(f"Loaded {weather.shape[0]} observations")
```

## 📁 Project Structure

```
data-wrangling-weather-analysis/
├── README.md
├── requirements.txt
├── weather_analysis.py           # Main analysis module
├── nyc_weather_analysis.ipynb    # Jupyter notebook
├── data/
│   └── nycflights13_weather.csv.gz
└── visualizations/
    ├── monthly_wind_speeds.png
    └── airport_comparison.png
```

## 🔬 Analysis Components

### 1. Data Extraction & Loading

```python
import gzip
import shutil
import numpy as np

# Extract .gz file
with gzip.open('nycflights13_weather.csv.gz', 'rb') as f_in:
    with open('nycflights13_weather.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load with NumPy
weather = np.genfromtxt('nycflights13_weather.csv', 
                        delimiter=',',
                        skip_header=1,
                        dtype=float)
```

### 2. Data Cleaning

```python
# Remove rows where all values are NaN
weather = weather[~np.isnan(weather).all(axis=1)]

# Check for remaining NaN values
nan_count = np.isnan(weather).sum()
print(f"Remaining NaN values: {nan_count}")
```

### 3. Unit Conversion

```python
def convert_to_metric(data):
    """Convert Imperial units to Metric."""
    # Temperature: Fahrenheit to Celsius
    temp_c = (data['temp'] - 32) * 5/9
    
    # Wind speed: mph to m/s
    wind_ms = data['wind_speed'] * 0.44704
    
    # Visibility: miles to meters
    visib_m = data['visib'] * 1609.34
    
    return temp_c, wind_ms, visib_m
```

### 4. Outlier Detection (IQR Method)

```python
def detect_outliers(data, column):
    """Detect outliers using IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | 
                    (data[column] > upper_bound)]
    return outliers
```

### 5. Monthly Aggregation

```python
# Calculate monthly mean wind speeds by airport
monthly_means = df.groupby(['month', 'origin'])['wind_speed'].mean()
monthly_means = monthly_means.unstack()
```

## 📈 Key Findings

### Monthly Mean Wind Speeds (m/s)

| Month | EWR | JFK | LGA |
|-------|-----|-----|-----|
| January | 4.72 | 5.82 | 5.60 |
| February | 5.21 | 6.30 | 5.95 |
| **March** | **5.77** | **6.83** | **6.53** |
| April | 4.90 | 6.11 | 5.60 |
| May | 4.16 | 5.05 | 4.78 |
| June | 4.80 | 5.60 | 5.07 |
| July | 4.63 | 5.17 | 4.81 |
| **August** | **3.85** | **4.94** | **4.30** |
| September | 4.11 | 5.01 | 4.53 |
| October | 4.19 | 5.26 | 5.25 |
| November | 5.16 | 6.21 | 5.95 |
| December | 4.50 | 5.57 | 5.18 |

### Key Insights

1. **Seasonal Patterns**:
   - **March peak**: Highest wind speeds across all airports
   - **August trough**: Lowest wind speeds of the year
   - **November secondary peak**: Pre-winter wind increase

2. **Airport Rankings** (Highest to Lowest):
   - JFK consistently highest winds
   - LGA middle range
   - EWR lowest winds

3. **Outlier Analysis**:
   - 387 outlier observations detected (1.5%)
   - Outliers distributed across all months
   - Most outliers in winter/spring months

4. **Operational Implications**:
   - Spring operations most affected by wind
   - Summer offers most stable conditions
   - JFK requires most wind-related planning

## 📊 Visualizations

### Monthly Wind Speed Trends
- Line plot comparing all three airports
- X-axis: Months (Jan-Dec)
- Y-axis: Wind Speed (m/s)
- Clear seasonal pattern visualization

### Key Visual Insights
- Parallel trends across airports
- JFK consistently elevated
- Summer dip clearly visible
- March peak prominent

## 🔧 Data Wrangling Techniques

| Technique | Implementation |
|-----------|----------------|
| **File Decompression** | `gzip.open()`, `shutil.copyfileobj()` |
| **CSV Parsing** | `np.genfromtxt()`, `pd.read_csv()` |
| **NaN Handling** | `np.isnan()`, boolean indexing |
| **Unit Conversion** | Custom conversion functions |
| **Outlier Detection** | IQR method with 1.5× multiplier |
| **Aggregation** | `groupby()`, `mean()`, `unstack()` |
| **Time Series** | Monthly grouping and analysis |

## 💼 Practical Recommendations

### For Airlines:

**Operational Planning**
- Schedule more maintenance in winter/spring
- Build buffer time into March schedules
- Optimize fuel loads based on seasonal patterns

**Airport-Specific Strategy**
- Route wind-sensitive aircraft through EWR when possible
- Extra fuel reserves for JFK operations
- Adjust schedules during peak wind seasons

**Cost Management**
- Budget for higher winter operating costs
- Take advantage of efficient summer conditions
- Plan maintenance during low-wind periods

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **Data Extraction** - Working with compressed files
2. **NumPy & Pandas** - Array and DataFrame manipulation
3. **Data Cleaning** - Handling real-world messy data
4. **Unit Conversion** - Imperial to Metric transformations
5. **Statistical Analysis** - Outlier detection, aggregations
6. **Time Series** - Temporal pattern analysis
7. **Visualization** - Multi-series line plots
8. **Domain Knowledge** - Meteorological data interpretation

## 👨‍💼 Author

**Victor Prefa**
- Medical Doctor & Data Scientist
- MSc Data Science & Business Analytics, Deakin University
- Student ID: 225187913

## 📚 References

1. nycflights13 Dataset - https://github.com/hadley/nycflights13
2. NumPy Documentation - https://numpy.org/doc/
3. Pandas Documentation - https://pandas.pydata.org/docs/
4. NOAA Weather Data - https://www.weather.gov/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project was developed as part of the Data Science coursework at Deakin University, demonstrating practical data wrangling skills applied to real-world meteorological data.*
