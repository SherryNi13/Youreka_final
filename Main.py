# Import necessary libraries
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import ace_tools as tools

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

# Streamlit display
st.title("Regression Analysis of Coccidioidomycosis Cases")

st.write("## Regression Equation")
st.write(equation)

st.write("## Regression Summary")
st.text(model.summary())

# Plotting the regression line graph for TAVG vs Coccidioidomycosis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of actual vs predicted cases
ax.scatter(analysis_df["TAVG"], y, color='blue', label="Actual cases")
ax.plot(analysis_df["TAVG"], model.fittedvalues, color='red', label="Regression line")

# Labels and title
ax.set_xlabel("Average Temperature (TAVG)", fontsize=12)
ax.set_ylabel("Coccidioidomycosis Cumulative Cases", fontsize=12)
ax.set_title("Regression Line: TAVG vs Coccidioidomycosis", fontsize=14)

# Show legend
ax.legend()

# Display the plot
st.pyplot(fig)

# Load the merged disease and climate dataset
merged_data_path = "path/to/Merged_Disease_and_Climate_Data.csv"
merged_df = pd.read_csv(merged_data_path)

# Simple Regression: TAVG vs Coccidioidomycosis cases
X_tavg = merged_df[["TAVG"]]
y = merged_df["Coccidioidomycosis, Cum Current Year"]

# Add constant for intercept
X_tavg = sm.add_constant(X_tavg)

# Fit the regression model
model_tavg = sm.OLS(y, X_tavg).fit()

# Simple Regression: PRCP vs Coccidioidomycosis cases
X_prcp = merged_df[["PRCP"]]
y = merged_df["Coccidioidomycosis, Cum Current Year"]

# Add constant for intercept
X_prcp = sm.add_constant(X_prcp)

# Fit the regression model
model_prcp = sm.OLS(y, X_prcp).fit()

# Multiple Regression: TAVG and PRCP vs Coccidioidomycosis cases
X_multiple = merged_df[["TAVG", "PRCP"]]
y = merged_df["Coccidioidomycosis, Cum Current Year"]

# Add constant for intercept
X_multiple = sm.add_constant(X_multiple)

# Fit the regression model
model_multiple = sm.OLS(y, X_multiple).fit()

# Get the regression summary for TAVG and PRCP
model_tavg_summary = model_tavg.summary()
model_prcp_summary = model_prcp.summary()
model_multiple_summary = model_multiple.summary()

# Display the regression results
tools.display_dataframe_to_user(name="TAVG Regression Coefficients", dataframe=model_tavg.summary2().tables[1])
tools.display_dataframe_to_user(name="PRCP Regression Coefficients", dataframe=model_prcp.summary2().tables[1])
tools.display_dataframe_to_user(name="Multiple Regression Coefficients", dataframe=model_multiple.summary2().tables[1])

# Plotting the regression line for TAVG vs Coccidioidomycosis cases
plt.figure(figsize=(10, 6))
plt.scatter(merged_df["TAVG"], y, color='blue', label="Actual cases")
plt.plot(merged_df["TAVG"], model_tavg.fittedvalues, color='red', label="Regression line")
plt.xlabel("Average Temperature (TAVG)", fontsize=12)
plt.ylabel("Coccidioidomycosis Cumulative Cases", fontsize=12)
plt.title("Simple Regression: TAVG vs Coccidioidomycosis", fontsize=14)
plt.legend()
plt.show()

# Plotting the regression line for PRCP vs Coccidioidomycosis cases
plt.figure(figsize=(10, 6))
plt.scatter(merged_df["PRCP"], y, color='green', label="Actual cases")
plt.plot(merged_df["PRCP"], model_prcp.fittedvalues, color='red', label="Regression line")
plt.xlabel("Precipitation (PRCP)", fontsize=12)
plt.ylabel("Coccidioidomycosis Cumulative Cases", fontsize=12)
plt.title("Simple Regression: PRCP vs Coccidioidomycosis", fontsize=14)
plt.legend()
plt.show()

# Print the regression summaries
print("TAVG Regression Summary:\n", model_tavg_summary)
print("PRCP Regression Summary:\n", model_prcp_summary)
print("Multiple Regression Summary:\n", model_multiple_summary)
