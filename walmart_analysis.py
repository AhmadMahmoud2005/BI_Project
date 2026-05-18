import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def save_plot(filename):
    os.makedirs("Images", exist_ok=True)
    plt.savefig(os.path.join("Images", filename), bbox_inches="tight")
    plt.close()

def main():
    print("=========================================")
    print("      WALMART DATASET ANALYSIS PROJECT     ")
    print("=========================================\n")
    
    # Load the dataset
    print("Loading data...")
    df = pd.read_csv("Walmart Dataset.csv", encoding='latin1')
    
    # Data Exploration & Answering Questions
    print("\n--- Data Exploration & Business Insights ---")
    
    highest_sales_category = df.groupby('Category')['Sales'].sum().idxmax()
    highest_sales_segment = df.groupby('Segment')['Sales'].sum().idxmax()
    
    region_sales = df.groupby('Region')['Sales'].sum()
    region_profit = df.groupby('Region')['Profit'].sum()
    
    state_sales = df.groupby('State')['Sales'].sum()
    state_profit = df.groupby('State')['Profit'].sum()
    
    city_sales = df.groupby('City')['Sales'].sum()
    city_profit = df.groupby('City')['Profit'].sum()
    
    print("=== Answers to Business Questions ===")
    print(f"1. Product category with highest sales: {highest_sales_category}")
    print(f"2. Customer segment with highest sales: {highest_sales_segment}")
    print(f"3. Region with highest sales: {region_sales.idxmax()} | Highest profit: {region_profit.idxmax()}")
    print(f"4. State with highest sales: {state_sales.idxmax()} | Highest profit: {state_profit.idxmax()}")
    print(f"5. City with highest sales: {city_sales.idxmax()} | Highest profit: {city_profit.idxmax()}")
    print("=====================================\n")

    print(df.head())
    print(df.info())
    print(df.describe())
    print("sum of null in each column: ")
    print(df.isnull().sum())
    print("duplicated data: ")
    print(df.duplicated().sum())

    topTenStateSales=df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10)
    topTenStateSales.plot(kind='bar',title="Top 10 states by sales")
    save_plot("top_10_states_by_sales.png")

    correlation=df[["Sales","Profit","Discount","Quantity"]].corr()
    sns.heatmap(correlation,annot=True,cmap="coolwarm")
    plt.title("correlation matrix")
    save_plot("correlation_matrix.png")

    df.plot(kind="scatter",x="Discount",y="Profit")
    save_plot("discount_vs_profit.png")

    df.plot(kind="scatter",x="Quantity",y="Profit")
    save_plot("quantity_vs_profit.png")

    plt.hist(df["Sales"], bins=30,color='blue')
    plt.xlabel("Sales")
    plt.ylabel("count")
    plt.title("sales distribution")
    save_plot("sales_distribution.png")

    sns.boxplot(data=df,x="Sales")
    save_plot("sales_boxplot.png")

    sns.boxplot(data=df,x="Profit")
    save_plot("profit_boxplot.png")
    
    # Data Cleaning & Transformation
    print("\n--- Data Cleaning & Transformation ---")
    print("Before duplicates removal:", df.shape)

    df = df.drop_duplicates()
    print("After duplicates removal:", df.shape)

    print("sum of null in each column before drop:")
    print(df.isnull().sum())
    
    df = df.dropna()

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print("Final shape:", df.shape)
    print(df.describe())

    df.to_csv("cleaned_walmart.csv", index=False)
    print(f"-> Saved cleaned dataset as 'cleaned_walmart.csv' for PowerBI visualization.")

    print("\n--- Integrated Time Series Analysis ---")
    try:
        from timeseries_arima import run_arima_analysis

        arima_result = run_arima_analysis("cleaned_walmart.csv")
        if arima_result is not None:
            p, d, q = arima_result["order"]
            print(f"Integrated ARIMA result: order=({p},{d},{q}), RMSE={arima_result['rmse']:.3f}")
    except Exception as e:
        print(f"Skipping integrated ARIMA analysis: {e}")
    
    # Machine Learning Model
    print("\n--- Machine Learning Model ---")
    ml_df = df.copy()
    
    features = ['Category', 'Sub-Category', 'Region', 'Quantity', 'Discount', 'Segment']
    target = 'Sales'
    
    X = ml_df[features].copy()
    y = ml_df[target]
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['Category', 'Sub-Category', 'Region', 'Segment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Sales Prediction')
    plt.tight_layout()
    save_plot('feature_importance.png')
    print("-> Feature importance plot saved as 'Images/feature_importance.png'")
    
    print("\nProject pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
