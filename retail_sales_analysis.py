import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

df = pd.read_csv("Superstore.csv", encoding="latin1")
df.head()

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("="*50)
print("COLUMN NAMES")
print(df.columns)

print("="*50)
print("FIRST 5 ROWS")
print(df.head())

print("="*50)
print("DATASET INFO")
print(df.info())

print("="*50)
print("STATISTICAL SUMMARY")
print(df.describe())

print("="*50)
print("MISSING VALUES")
print(df.isnull().sum())

print("="*50)
print("DUPLICATES")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'])

print("="*50)
print("SALES BY CATEGORY")

sales_category = df.groupby("Category")["Sales"].sum()

print(sales_category)

plt.figure(figsize=(8,5))
sales_category.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("sales_by_category.png")
plt.show()

print("="*50)
print("PROFIT BY CATEGORY")

profit_category = df.groupby("Category")["Profit"].sum()

print(profit_category)

plt.figure(figsize=(8,5))
sns.barplot(x=profit_category.index,y=profit_category.values)
plt.title("Profit by Category")
plt.tight_layout()
plt.savefig("profit_by_category.png")
plt.show()

print("="*50)
print("REGIONAL SALES")

region_sales = df.groupby("Region")["Sales"].sum()

print(region_sales)

plt.figure(figsize=(8,8))
plt.pie(region_sales.values,
        labels=region_sales.index,
        autopct="%1.1f%%")
plt.title("Regional Sales Distribution")
plt.savefig("regional_sales.png")
plt.show()

print("="*50)
print("TOP 10 PRODUCTS")

top_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_products)

plt.figure(figsize=(10,6))
top_products.plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.tight_layout()
plt.savefig("top_products.png")
plt.show()

df["Month"] = df["Order Date"].dt.to_period("M")

monthly_sales = df.groupby("Month")["Sales"].sum()

print("="*50)
print("MONTHLY SALES TREND")

print(monthly_sales)

plt.figure(figsize=(12,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.show()

print("="*50)
print("TOP 10 CUSTOMERS")

top_customers = (
    df.groupby("Customer Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print(top_customers)

plt.figure(figsize=(10,6))
top_customers.plot(kind="bar")
plt.title("Top Customers by Sales")
plt.tight_layout()
plt.savefig("top_customers.png")
plt.show()

print("="*50)
print("CORRELATION MATRIX")

correlation = df[["Sales","Profit","Quantity","Discount"]].corr()

print(correlation)

plt.figure(figsize=(8,6))
sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

print("="*50)
print("KEY BUSINESS INSIGHTS")

print("1. Highest sales generating category:")
print(sales_category.idxmax())

print("2. Most profitable category:")
print(profit_category.idxmax())

print("3. Highest revenue region:")
print(region_sales.idxmax())

print("4. Best selling product:")
print(top_products.index[0])

print("5. Top customer:")
print(top_customers.index[0])

print("="*50)
print("PROJECT COMPLETED SUCCESSFULLY")
