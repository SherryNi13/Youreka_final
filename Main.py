# Generate a regression line graph for TAVG vs Coccidioidomycosis cases
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the regression line for TAVG vs Coccidioidomycosis
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='TAVG', y='Coccidioidomycosis, Previous 52 weeks Max†', data=df_merged, ax=ax, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
ax.set_title('Regression Line: Temperature vs Coccidioidomycosis (Previous 52 Weeks Max)', fontsize=14)
ax.set_xlabel('Average Temperature (TAVG)', fontsize=12)
ax.set_ylabel('Coccidioidomycosis (Previous 52 Weeks Max)', fontsize=12)

# Display the graph
st.pyplot(fig)

# Generate p-value bar graph for TAVG and PRCP
p_values = {'TAVG': model.pvalues['TAVG'], 'PRCP': model.pvalues['PRCP']}
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Variable', 'P-Value'])

# Plot p-values
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Variable', y='P-Value', data=p_values_df, ax=ax2)
ax2.set_title('P-Value for TAVG and PRCP', fontsize=14)
ax2.set_ylabel('P-Value', fontsize=12)

# Display the p-value graph
st.pyplot(fig2)
