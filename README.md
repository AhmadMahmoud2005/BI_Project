# BI_Project

=========================================
      WALMART DATASET ANALYSIS PROJECT     
=========================================

Loading data...

--- Data Exploration & Business Insights ---
=== Answers to Business Questions ===
1. Product category with highest sales: Technology
2. Customer segment with highest sales: Consumer
3. Region with highest sales: West | Highest profit: West
4. State with highest sales: California | Highest profit: California
5. City with highest sales: New York City | Highest profit: New York City
=====================================

   Row ID        Order ID  Order Date   Ship Date  ...     Sales Quantity Discount    Profit
0       1  CA-2016-152156  11/08/2016  11/11/2016  ...  261.9600        2     0.00   41.9136
1       2  CA-2016-152156  11/08/2016  11/11/2016  ...  731.9400        3     0.00  219.5820
2       3  CA-2016-138688  06/12/2016  06/16/2016  ...   14.6200        2     0.00    6.8714
3       4  US-2015-108966  10/11/2015  10/18/2015  ...  957.5775        5     0.45 -383.0310
4       5  US-2015-108966  10/11/2015  10/18/2015  ...   22.3680        2     0.20    2.5164

[5 rows x 21 columns]
<class 'pandas.DataFrame'>
RangeIndex: 9994 entries, 0 to 9993
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Row ID         9994 non-null   int64  
 1   Order ID       9994 non-null   str    
 2   Order Date     9994 non-null   str    
 3   Ship Date      9994 non-null   str    
 4   Ship Mode      9971 non-null   str    
 5   Customer ID    9994 non-null   str    
 6   Customer Name  9994 non-null   str    
 7   Segment        9994 non-null   str    
 8   Country        9994 non-null   str    
 9   City           9994 non-null   str    
 10  State          9994 non-null   str    
 11  Postal Code    9994 non-null   int64  
 12  Region         9994 non-null   str    
 13  Product ID     9994 non-null   str    
 14  Category       9994 non-null   str    
 15  Sub-Category   9994 non-null   str    
 16  Product Name   9994 non-null   str    
 17  Sales          9994 non-null   float64
 18  Quantity       9994 non-null   int64  
 19  Discount       9945 non-null   float64
 20  Profit         9994 non-null   float64
dtypes: float64(3), int64(3), str(15)
memory usage: 1.6 MB
None
            Row ID   Postal Code         Sales     Quantity     Discount       Profit
count  9994.000000   9994.000000   9994.000000  9994.000000  9945.000000  9994.000000
mean   4997.500000  55190.379428    229.858001     3.789574     0.156972    28.656896
std    2885.163629  32063.693350    623.245101     2.225110     0.206668   234.260108
min       1.000000   1040.000000      0.444000     1.000000     0.000000 -6599.978000
25%    2499.250000  23223.000000     17.280000     2.000000     0.000000     1.728750
50%    4997.500000  56430.500000     54.490000     3.000000     0.200000     8.666500
75%    7495.750000  90008.000000    209.940000     5.000000     0.200000    29.364000
max    9994.000000  99301.000000  22638.480000    14.000000     0.800000  8399.976000
sum of null in each column: 
Row ID            0
Order ID          0
Order Date        0
Ship Date         0
Ship Mode        23
Customer ID       0
Customer Name     0
Segment           0
Country           0
City              0
State             0
Postal Code       0
Region            0
Product ID        0
Category          0
Sub-Category      0
Product Name      0
Sales             0
Quantity          0
Discount         49
Profit            0
dtype: int64
duplicated data: 
0

--- Data Cleaning & Transformation ---
Before duplicates removal: (9994, 21)
After duplicates removal: (9994, 21)
sum of null in each column before drop:
Row ID            0
Order ID          0
Order Date        0
Ship Date         0
Ship Mode        23
Customer ID       0
Customer Name     0
Segment           0
Country           0
City              0
State             0
Postal Code       0
Region            0
Product ID        0
Category          0
Sub-Category      0
Product Name      0
Sales             0
Quantity          0
Discount         49
Profit            0
dtype: int64
Final shape: (6706, 21)
            Row ID                  Order Date                   Ship Date   Postal Code        Sales     Quantity     Discount       Profit
count  6706.000000                        6706                        6706   6706.000000  6706.000000  6706.000000  6706.000000  6706.000000
mean   5010.543692  2016-05-01 20:20:06.799880  2016-05-05 19:38:27.306889  55806.552789    68.111362     3.374143     0.096897    11.655695
min       1.000000         2014-01-03 00:00:00         2014-01-07 00:00:00   1453.000000     0.990000     1.000000     0.000000   -26.875800
25%    2542.250000         2015-05-25 00:00:00         2015-05-28 00:00:00  22304.000000    14.940000     2.000000     0.000000     3.246300
50%    5042.500000         2016-07-03 00:00:00         2016-07-05 00:00:00  56430.500000    34.795000     3.000000     0.000000     7.927400
75%    7467.750000         2017-05-14 00:00:00         2017-05-19 00:00:00  90036.000000    84.039000     4.000000     0.200000    17.667300
max    9993.000000         2017-12-30 00:00:00         2018-01-05 00:00:00  99301.000000   496.860000     9.000000     0.500000    53.270400
std    2869.054674                         NaN                         NaN  32783.818962    85.895680     1.867508     0.105226    13.509949
-> Saved cleaned dataset as 'cleaned_walmart.csv' for PowerBI visualization.

--- Integrated Time Series Analysis ---
-> Saved rolling mean/std plot to Images/monthly_rolling_mean_std.png
ADF test p-value (d=0): 0.01069
Selected differencing order d = 0
-> Saved ACF plot to Images/acf_stationary.png
-> Saved PACF plot to Images/pacf_stationary.png
Selected p=1, q=1 (threshold=0.2 heuristic)
Train periods: 36, Test periods: 12
ARIMA(1,0,1) RMSE on test: 6507.362
-> Saved forecast plot to Images/arima_forecast.png
Integrated ARIMA result: order=(1,0,1), RMSE=6507.362

--- Machine Learning Model ---
Training Random Forest Regressor...
Model Evaluation Metrics:
Root Mean Squared Error (RMSE): 70.96
R-squared (R2): 0.2574
-> Feature importance plot saved as 'Images/feature_importance.png'