import pandas as pd
import statsmodels.api as sm
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the climate and disease data (you should have already uploaded these in the app)
@st.cache
def load_climate_data():
    climate_data = pd.read_excel('New_climate_data.xlsx')
    return climate_data[['MMWR Year', 'Reporting Area', 'TAVG', 'PRCP']]

@st.cache
def load_disease_data():
    # Load the disease data (ensure the path is correct)
    disease_data = pd.read_csv('2014_NNDSS_Table_II.csv')
    disease_columns = ['Reporting Area', 'MMWR Year', 
                       'Coccidioidomycosis, Previous 52 weeks Max', 
                       'Coccidioidomycosis, Previous 52 weeks Med', 
                       'Coccidioidomycosis, Cum 2014']
    return disease_data[disease_columns]

# Load the data
climate_df = load_climate_data()
disease_df = load_disease_data()

# Merge the climate and disease datasets
merged_df = pd.merge(climate_df, disease_df, on=['MMWR Year', 'Reporting Area'], how='inner')

# Regression analysis for temperature (TAVG) and precipitation (PRCP)
X = merged_df[['TAVG', 'PRCP']]  # Independent variables (temperature and precipitation)
y = merged_df['Coccidioidomycosis, Previous 52 weeks Max']  # Dependent variable (disease cases)

# Add constant (intercept) to the model
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Show the regression results in Streamlit
st.title("Regression and Correlation Analysis")

st.subheader("Regression Analysis Results")
st.write(model.summary())

# Plot the regression line for TAVG (Temperature)
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='TAVG', y='Coccidioidomycosis, Previous 52 weeks Max', data=merged_df, ax=ax)
ax.set_title('Regression Line: Temperature vs Coccidioidomycosis Cases')
ax.set_xlabel('Temperature (°F)')
ax.set_ylabel('Coccidioidomycosis Cases')
st.pyplot(fig)

# Plot the regression line for PRCP (Precipitation)
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='PRCP', y='Coccidioidomycosis, Previous 52 weeks Max', data=merged_df, ax=ax)
ax.set_title('Regression Line: Precipitation vs Coccidioidomycosis Cases')
ax.set_xlabel('Precipitation (inches)')
ax.set_ylabel('Coccidioidomycosis Cases')
st.pyplot(fig)

# Show the correlation matrix between temperature, precipitation, and disease cases
correlations = merged_df[['TAVG', 'PRCP', 'Coccidioidomycosis, Previous 52 weeks Max', 
                          'Coccidioidomycosis, Previous 52 weeks Med', 'Coccidioidomycosis, Cum 2014']].corr()

st.subheader("Correlation Matrix")
st.write(correlations)

# Display the correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

