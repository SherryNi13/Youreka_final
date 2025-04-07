import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st

# Function to load the merged dataset from the GitHub repository (ensure the file is in the same directory as the script or provide the path)
@st.cache
def load_data():
    # Load the CSV file from the repository (ensure it's in the correct directory or path)
    df = pd.read_csv("Coccidioidomycosis_Regression_Analysis.csv")
    return df

# Load the data
df_merged = load_data()

# Perform the regression analysis: TAVG, PRCP vs Coccidioidomycosis (Previous 52 Weeks Max)
X = df_merged[['TAVG', 'PRCP']]  # Independent variables
y = df_merged['Coccidioidomycosis, Previous 52 weeks Max†']  # Dependent variable

# Add constant to the model (intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Get the summary of the regression model
model_summary = model.summary()

# Streamlit UI
st.title("Coccidioidomycosis and Climate Analysis")
st.markdown("""
This app performs regression analysis on Coccidioidomycosis cases, 
considering temperature (TAVG) and precipitation (PRCP) from 2014 to 2022.
""")

# Show the regression results summary
st.subheader("Regression Analysis Summary")
st.text(model_summary)

# Plot the regression line for TAVG vs Coccidioidomycosis
st.subheader('Regression Line: Temperature vs Coccidioidomycosis')
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='TAVG', y='Coccidioidomycosis, Previous 52 weeks Max†', data=df_merged, ax=ax, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
ax.set_title('Regression Line: Temperature vs Coccidioidomycosis (Previous 52 Weeks Max)', fontsize=14)
ax.set_xlabel('Average Temperature (TAVG)', fontsize=12)
ax.set_ylabel('Coccidioidomycosis (Previous 52 Weeks Max)', fontsize=12)
st.pyplot(fig)

# Plot the p-values for TAVG and PRCP
st.subheader('P-Values for TAVG and PRCP')
p_values = {'TAVG': model.pvalues['TAVG'], 'PRCP': model.pvalues['PRCP']}
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Variable', 'P-Value'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Variable', y='P-Value', data=p_values_df, ax=ax2)
ax2.set_title('P-Value for TAVG and PRCP', fontsize=14)
ax2.set_ylabel('P-Value', fontsize=12)
st.pyplot(fig2)

# Show a dataframe of the merged data (optional)
st.subheader("Merged Data Sample")
st.dataframe(df_merged.head())
