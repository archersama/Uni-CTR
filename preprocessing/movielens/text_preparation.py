import pandas as pd

# Assuming your data is in a CSV file, replace 'your_data.csv' with your actual data file path
df = pd.read_csv('../../datasets/movielens/ml-1m/merged_movielens_data.dat', sep='\t', engine='python')

# Define mappings
age_mapping = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

occupation_mapping = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

gender_mapping = {
    'F': "female",
    'M': "male"
}

# Apply mappings
df['Age'] = df['Age'].map(age_mapping)
df['Occupation'] = df['Occupation'].map(occupation_mapping)
df['Gender'] = df['Gender'].map(gender_mapping)

# Save the updated DataFrame back to CSV or view it
df.to_csv('../../datasets/movielens/ml-1m/merged_movielens_text_prep.csv', sep='\t', index=False)

# Or view the DataFrame
# print(df)
