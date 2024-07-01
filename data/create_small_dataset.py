import pandas as pd

# Load the original CSV file
df = pd.read_csv('zinc.csv')

# Randomly select 500 rows
df_small = df.sample(n=10000)

# Save the smaller dataframe to a new CSV file
df_small.to_csv('zinc_small.csv', index=False)