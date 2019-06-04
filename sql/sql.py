import sqlite3
import pandas as pd

conn = sqlite3.connect('population_data.db')

# all data
pd.read_sql('SELECT * FROM population_data', conn)

# Write a query that finds the change in population from 1960 to 1961 in Aruba
print(pd.read_sql('SELECT country_name, "1960", "1961" FROM population_data WHERE country_name="Aruba" ', conn))

# Write a query that finds the population of Belgium and also Luxembourg in 1975. The output should have two rows.
print(pd.read_sql('SELECT country_name, "1975" FROM population_data WHERE country_name="Belgium" OR country_name="Luxembourg"', conn))
