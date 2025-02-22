Final Project Submission¶
Please fill out:

Student name: Tabby Mirara

Student pace: full time - Remote

Scheduled project review date/time:

Instructor name: Lucille Kaleha

Blog post URL:

# Aviation Accidents Dataset


## Project Title: 
# Risk Analysis of Aircraft Types for Safer Business Operations






### 1. Introduction

In this project we will be working with Aviation Accident Dataset to solve our business problem.

The purpose of this project is to analyse data and determine which aircraft types are at the lowest risk of accidents. Using the  Aviation dataset, we perform Exploratory Data Analysis(EDA) to find out the partterns in accidents by aircraft type, weather conditions and the most affected countries. 






#### Business Problem

The business problem is, the company is interested in purchasing and operating airplanes for commercial and private enterprises, but do not know anything about the potential risks of aircraft. So we need to analyse the data and find out which  aircraft Make is suitable.   

#### Objectives
1. Cleaning the dataset for accurate for analysis.

2. Analyzing the dataset using Exploratory Data Analysis in order to identify patterns, trends and the risk factors associated with different make of aircraft.

3. Creating visualizations and interpreting our findings using our dataset to assist the company in identifying the safest aircraft for its new business endeavor. 
  


### 2. Data Understanding

Load the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Rename the data df 

df = pd.read_csv('Aviation_Data.csv')
df

Examine the structure of your dataset

# Know the type, columns and number of rows
df.info()

# summary statistics for numerical columns
df.describe()

# check the number of rows and columns
df.shape

# check unique values for categorical columns
df.nunique()

### 3. Data Cleaning and Analysis
 This involves tasks like removing duplicates, filling in missing values, fixing errors, and standardizing formats.

# check for missing values
missing_values = df.isna().sum()
print(missing_values[missing_values > 0])


##### handling missing data
 Removing the  rows and columns with excessive missing values.
 The threshold for the columns will be 50%

# Dropping columns with too many missing values(more than 50% missing values)
# define the threshold
threshold = len(df)*0.5

#Drop Columns with more than 50% missing values
df_clean= df.dropna(axis=1, thresh=threshold)

print(df_clean.columns)




df_clean.info()

Fill the rest of the missing values whose threshold is less than 50%.

# for numerical columns, fill missing with median
numerical_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df_clean[col].fillna(df_clean[col].mode()[0],inplace=True)



# For categoricalcolumns, fill missing with mode
categorical_columns = df_clean.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df_clean[col].fillna(df_clean[col].mode()[0],inplace=True)

We will run the cell below to ensure that the above steps have worked by filling the miising values and removing the columns that have more than 50% missing values

df_clean.info()

Convert the necessary columns with floats to integers.
 The necessary columns are :Number.of.Engines, 
                            Total.Fatal.Injuries, 
                            Total.Serious.Injuries, 
                            Total.Minor.Injuries. 
                            
as shown below                            

df.dtypes

# List of columns to convert
columns_to_convert = [
    'Total.Fatal.Injuries',
    'Total.Serious.Injuries',
    'Total.Minor.Injuries',
    'Total.Uninjured'
]

# Convert columns to integers
df_clean[columns_to_convert] = df_clean[columns_to_convert].astype(int)

# Verify the conversion
print(df_clean[columns_to_convert].dtypes)

We will also need to convert thw Number.of.Engines column to integer.

df_clean['Number.of.Engines'] = df_clean['Number.of.Engines'].astype(int)
print(df_clean['Number.of.Engines'].dtypes)

The columns have been converted to integer as shown below 

df_clean.dtypes

Confirm that the data has no missing values.

df_clean.isna().sum()

### 4. Data Visualization

Using Exploratory Data Analysis(EDA), we will visualize the data to enhance our understanding of the data analysis and to achieve our objective. 



# Let's get a statistical summary of the numerical features
#Summary statistics for numerical columns
print(df_clean[numerical_columns].describe())

##### Analyze how aviation accidents have trended over the years.

Analysis to see how the number of accidents have been for the past years

# Convert 'Event.Date' to datetime if it's not already
df_clean['Event.Date'] = pd.to_datetime(df_clean['Event.Date'], errors='coerce')

# Extract the 'Year' from the 'Event.Date'
df_clean['Year'] = df_clean['Event.Date'].dt.year

# Plot number of accidents per year
plt.figure(figsize=(12,6))
sns.countplot(x='Year', data=df_clean, palette='viridis')
plt.xticks(rotation=45)
plt.title('Number of Aviation Accidents per Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.show()
![My Project Logo](capture 1.png)


Interpretation:
there gradual decrease over the Years.
1982 had the highest number of accidents
The darker the colour the higher the number of accidents

#####  Accidents by Country
Identify which countries have the highest number of aviation accidents.

top_countries = df_clean['Country'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='magma')
plt.title('Top 10 Countries with Most Aviation Accidents')
plt.xlabel('Country')
plt.ylabel('Number of Accidents')
plt.show()
![My Project Logo](Top 10 Countries with Most Aviation Accidents.png)

Interpretation:
The United States has significantly high number of accidents compared to the rest of the countries

#####  Weather Conditions and Accidents
Examine the relationship between weather conditions and accident severity.

# Count of accidents by weather condition
weather_counts = df_clean['Weather.Condition'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=weather_counts.index, y=weather_counts.values, palette='coolwarm')
plt.title('Aviation Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.show()
![My Project Logo](Aviation Accidents by Weather Condition.png)

Interpretation:
vmc means  fly by visual, on clear weather
From the bar plot above most crashes occur on clear weather(vmc) meaning that weather is not a significat factor affecting accident occurence.

##### Accident counts by aircraft Make
 Compare accident counts by aircraft make

# We will be using the 'Make' column
df_clean['Make'].head(20)


# Count the number of accidents per aircraft type
df_clean['Make'].value_counts()




# Replacing repeated values in the Make column
Make_clean = df_clean['Make'].replace(['CESSNA', 'PIPER', 'BEECH', 'BELL', 'BOEING'], ['Cessna','Piper' ,'Beech', 'Bell', 'Boeing'])
Make_clean.head(10)

accident_counts= Make_clean.value_counts()
accident_counts

accident_counts.head(20)

# Convert to DataFrame for easier plotting
accident_counts_df = accident_counts.reset_index()
accident_counts_df.columns = ['Make', 'Number of Accidents']
accident_counts_df 

# Plot a bar chart
plt.figure(figsize=(16, 8))
sns.barplot(
    data=accident_counts_df.head(10),  # Top 10 aircraft types with the most accidents
    x='Make',
    y='Number of Accidents',
    palette='viridis'
)
plt.title(Number of Aviation Accidents by Make, fontsize=16)
plt.xlabel('Make', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()
![My Project Logo]('Number of Aviation Accidents by Make'.png)

Interpretation:
 The number of accidents in cessna is significatly higher than the other aeroplane makes  

The following are the conclusions

### 5. Findings

The following are the findings from the above plots
1. The number of accidents in cessna is significatly higher than the other aircraft makes.

2. There gradual decrease of accidents over the Years. 1982 had the highest number of accidents The darker the colour the higher the number of accidents.

3. The United States has significantly high number of accidents compared to the rest of the countries.

4. vmc simply means fly by visual, on clear weather. Most crashes occur on clear weather(vmc) meaning that weather is not a significat factor affecting accident occurence.



### 6. Recommendations

1.Avoid high-risk aircraft like Cessna and opt for safer alternatives such as Aviocar CASA which has significantly fewer number of accidents.

2.The company can start its business in regions with lower accident frequencies like Germany, avoiding U.S. hotspots. 

3.Adopt predictive maintenance practices and modernize the fleet with safer and newer aircraft models, to minimize risks and ensure aircraft reliability.

