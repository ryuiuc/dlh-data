import pandas as pd

# Load the data
data = pd.read_csv('listfile.csv')

# Extract the stay ID from the 'stay' column
data['stay_id'] = data['stay'].apply(lambda x: x.split('_episode')[0])

# Extract the episode number and convert it to integer for accurate comparison
data['episode_number'] = data['stay'].apply(lambda x: int(x.split('_episode')[1].split('_timeseries.csv')[0]))

# Sort data by stay_id and episode_number to ensure the ordering
data.sort_values(by=['stay_id', 'episode_number'], inplace=True)

# Get the first episode for each stay_id
first_episodes = data.drop_duplicates(subset=['stay_id'], keep='first')

# Count total episodes
total_episodes = len(data)
print(f'Total number of episodes: {total_episodes}')

# Calculate mortality rate based on episodes
episode_mortality_rate = data['y_true'].mean()
print(f'Mortality rate based on episodes: {episode_mortality_rate:.2%}')

# Count unique stay IDs
unique_stay_ids = data['stay_id'].nunique()
print(f'Number of unique stay IDs: {unique_stay_ids}')

# Calculate the maximum y_true for each stay ID to determine mortality
data_grouped = data.groupby('stay_id')['y_true'].max()  # This gives 1 if any episode is 1, otherwise 0

# Calculate mortality rate based on unique stay IDs
unique_stay_mortality_rate = data_grouped.mean()
print(f'Mortality rate based on unique stay IDs: {unique_stay_mortality_rate:.2%}')

# Counts of mortality and survival based on episodes
episode_mortality_counts = data['y_true'].value_counts()
print('Counts of Mortality and Survival based on episodes:')
print(episode_mortality_counts)

# Counts of mortality and survival based on unique stays
unique_stay_mortality_counts = data_grouped.value_counts()
print('Counts of Mortality and Survival based on unique stay IDs:')
print(unique_stay_mortality_counts)

# Mortality based on the first episode of each stay
first_episode_mortality_counts = first_episodes['y_true'].value_counts()
print('Counts of Mortality and Survival based on the first episode of each stay:')
print(first_episode_mortality_counts)

# Mortality rate based on the first episode of each stay
first_episode_mortality_rate = first_episodes['y_true'].mean()
print(f'Mortality rate based on the first episode of each stay: {first_episode_mortality_rate:.2%}')