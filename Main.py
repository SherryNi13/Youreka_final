import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the cleaned disease and climate data
disease_data_path = "Coccidioidomycosis_Data_2014-2022.xlsx"
climate_data_path = "New climate data.xlsx"

# Load datasets
disease_df = pd.read_excel(disease_data_path)
climate_df = pd.read_excel(climate_data_path)

# Aggregate disease data by Reporting Area and MMWR Year (mean for duplicates)
disease_df_grouped = disease_df.groupby(["Reporting Area", "MMWR Year"], as_index=False).mean(numeric_only=True)

# Standardize state names (uppercase) for both datasets
disease_df_grouped["Reporting Area"] = disease_df_grouped["Reporting Area"].str.upper()
climate_df["Reporting Area"] = climate_df["Reporting Area"].str.upper()

# Merge datasets on Reporting Area and MMWR Year
merged_df = pd.merge(disease_df_grouped, climate_df, on=["Reporting Area", "MMWR Year"], how="inner")

# Drop unnecessary columns from climate data
merged_df = merged_df[[
    "Reporting Area", "MMWR Year",
    "Coccidioidomycosis, Cum Current Year",
    "Coccidioidomycosis, Cum Previous Year",
    "TAVG", "PRCP"
]]

# Drop rows with missing values for analysis
analysis_df = merged_df.dropna(subset=[
    "Coccidioidomycosis, Cum Current Year", "TAVG", "PRCP"
])

# Build the OLS regression model
X = analysis_df[["TAVG", "PRCP"]]
y = analysis_df["Coccidioidomycosis, Cum Current Year"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Extract regression results
intercept = model.params['const']
coef_tavg = model.params['TAVG']
coef_prcp = model.params['PRCP']

# Create regression equation string
equation = f"Cases = {intercept:.3f} + ({coef_tavg:.3f} × TAVG) + ({coef_prcp:.3f} × PRCP) + ε"

# Print the equation and regression summary
print("Regression Equation:")
print(equation)
print(model.summary())

# Plotting the regression line graph for TAVG vs Coccidioidomycosis
plt.figure(figsize=(10,6))

# Scatter plot of actual vs predicted cases
plt.scatter(analysis_df["TAVG"], y, color='blue', label="Actual cases")
plt.plot(analysis_df["TAVG"], model.fittedvalues, color='red', label="Regression line")

# Labels and title
plt.xlabel("Average Temperature (TAVG)", fontsize=12)
plt.ylabel("Coccidioidomycosis Cumulative Cases", fontsize=12)
plt.title("Regression Line: TAVG vs Coccidioidomycosis", fontsize=14)

# Show legend and plot
plt.legend()
plt.show()

