import pandas as pd



# Importing dataset
dataset = pd.read_csv('/data_fouling_without_processing.csv', sep=';', decimal=',')
dataset

# Deleting values less than 1
dataset.drop(dataset[
                 (dataset.Fdw < 1) | (dataset.Tdw < 1) | (dataset.Ta < 1) | (dataset.Pa < 1) | (dataset.Thwi < 1) | (
                             dataset.Thwo < 1)].index, inplace=True)
dataset.reset_index(drop=True)

# Convert Time column to datetime format
dataset['DATE TIME'] = pd.to_datetime(dataset['DATE TIME'])
dataset

# Deleting rows that have same date time
same_time = dataset[~dataset['DATE TIME'].dt.round('min').duplicated()]
same_time

# Reindexing
re_ind = same_time.reset_index(drop=True)
re_ind

# Averaging
re_ind.groupby(re_ind.index // 5).agg(
    {'Fc': 'mean', 'Td': 'mean', 'Ta': 'mean', 'Pa': 'mean', 'Thi': 'mean', 'Thwo': 'mean'})
