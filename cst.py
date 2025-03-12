import pandas as pd  # Import pandas for data handling
import statsmodels.api as sm  # Import statsmodels for logistic regression

# Define file path to the dataset
file_path = 'cust2.csv'  

# Read  CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display  first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())

#  Prepare the data for logistic regression 

# Select the independent variables (Age and Income) 
X = df[['Age', 'Income']]

# Select the dependent variable (Made_Purchase), which indicates whether a purchase was made (1) or not (0)
y = df['Made_Purchase']

# Add a constant term to the independent variables 
X = sm.add_constant(X)

# Fit a logistic regression model using the Logit function (for binary classification)
model = sm.Logit(y, X).fit()

# Print out the summary of the logistic regression results
print(model.summary())

# Make predictions on new customer data 

# Create a new DataFrame with customer details for prediction
new_data = pd.DataFrame({
    'const': [1]*3,  # Add a constant term for the intercept
    'Age': [24, 38, 45],  # Input new customer ages
    'Income': [50000, 85000, 40000]  # Input new customer incomes
})

# Use the trained model to predict purchase probabilities
predictions = model.predict(new_data)

# Convert probabilities into class predictions using a threshold of 0.5
predicted_classes = [1 if p > 0.5 else 0 for p in predictions]

# Display the predictions
for i in range(len(new_data)):
    print(f"Customer {i+1}: Predicted Purchase = {'Yes' if predicted_classes[i] == 1 else 'No'} (Probability: {predictions[i]:.2f})")
