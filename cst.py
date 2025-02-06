import pandas as pd
import statsmodels.api as sm


file_path = 'cust2.csv' 

df = pd.read_csv(file_path)


print("First few rows of the dataset:")
print(df.head())

# Prepare the data for logistic regression
# Define independent variables (X) and dependent variable (y)
X = df[['Age', 'Income']]
y = df['Made_Purchase']


X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(y, X).fit()


print(model.summary())

# Step: Make predictions on new data
new_data = pd.DataFrame({
    'const': [1]*3,
    'Age': [24,38,45],
    'Income': [50000,85000,40000]
})

# Make predictions using the model
predictions = model.predict(new_data)

# (threshold of >0.5)
predicted_classes = [1 if p >0.5 else 0 for p in predictions]

# Display predictions
for i in range(len(new_data)):
    print(f"Customer {i+1}: Predicted Purchase = {'Yes' if predicted_classes[i] ==1 else'No'} (Probability: {predictions[i]:.2f})")
