#pandas, matplotlib, seaborn

# create dictionary
stu_dic = {"Name":["Aqsa","Ali","Sara","Hina","Ahmed","Fatema","Amna", "Ayesha", "Afsa","Abdul"],
           "ID": ["STD-1","STD-2","STD-3","STD-4","STD-5","STD-6","STD-7","STD-8","STD-9","STD-10"],
           "Roll_no":[1,2,3,4,5,6,7,8,9,10],
           "Semester":[7,8,6,7,5,8,6,7,5,8],}

# convert dictionary to dataframe
import pandas as pd
stu_df = pd.DataFrame(stu_dic)
print(stu_df)

# check datatype 
print("\n The data type of the syntax is: ",type(stu_df))

# use of describe function
print("\n The description of the dataframe is: \n",stu_df.describe())

# use of head function
print("\n The first 5 rows of the dataframe are: \n",stu_df.head())

# use of tail function
print("\n The last 5 rows of the dataframe are: \n",stu_df.tail())

# use of info function
print("\n The information of the dataframe is: \n",stu_df.info())

# convert dataframe to csv file
stu_df.to_csv("students.csv")

# remove index column while converting to csv file
stu_df.to_csv("students_no_index.csv", index=False)

# read csv file
stu_df_from_csv = pd.read_csv("students_no_index.csv")
print("\n The dataframe read from csv file is: \n",stu_df_from_csv)

# Use describe,head,tail and info function for CSV file
print("\n The description of the dataframe read from csv file is: \n",stu_df_from_csv.describe())
print("\n The first 5 rows of the dataframe read from csv file are: \n",stu_df_from_csv.head())
print("\n The last 5 rows of the dataframe read from csv file are: \n",stu_df_from_csv.tail())
print("\n The information of the dataframe read from csv file is: \n",stu_df_from_csv.info())       

# Access a column by its name
print("\n The Name column of the dataframe is: \n",stu_df["Name"])

# Access the 1st element of a column
print("\n The 1st element of the Name column is: \n",stu_df["Name"][0])

# Update a value in the column
stu_df.at[0,"Name"] = "Aqsa Khan"
print("\n The updated dataframe is: \n",stu_df)

# Find the columns and indexes in a data frame
print("\n The columns of the dataframe are: \n",stu_df.columns)
print("\n The indexes of the dataframe are: \n",stu_df.index)

# Create a series of 50 random numbers and check their data type and shape
import numpy as np
random_numbers = np.random.rand(50)
print("\n The random numbers are: \n",random_numbers)
print("\n The data type of the random numbers is: ",type(random_numbers))
print("\n The shape of the random numbers is: ",random_numbers.shape)

# Create a 50 x 5 data set from random values
data = np.random.rand(50,5)
print("\n The random data set is: \n",data)

# Find the minimum maximum and mean values column wise in a dataset
print("\n The minimum values column wise in the dataset are: \n",data.min(axis=0))
print("\n The maximum values column wise in the dataset are: \n",data.max(axis=0))
print("\n The mean values column wise in the dataset are: \n",data.mean(axis=0))

# Find the maximum value in 1
print("\n The maximum value in the dataset is: \n",data.max())

# Convert the dataset into numpy array and also take transpose of it
data_array = np.array(data)
print("\n The numpy array of the dataset is: \n",data_array)
transpose_data = data_array.T
print("\n The transpose of the dataset is: \n",transpose_data)

# Change names of the columns.
data_df = pd.DataFrame(data, columns=["A","B","C","D","E"])
print("\n The dataframe with column names is: \n",data_df)

# Display column B and C from the dataset
print("\n The columns B and C from the dataset are: \n",data_df[["B","C"]])

# Use head function
print("\n The first 5 rows of the dataset are: \n",data_df.head())


# Demonstrate the use of iloc function
print("\n Use ILOC function: \n",data_df.iloc[:,0:2])

# use loc function Print column A to C and fimd the value on 0,0
print("\n Use LOC function: \n",data_df.loc[:, "A":"C"])
print("\n The value at 0,0 is: \n",data_df.loc[0,"A"])

# Print 1st 12 elements of column 2 and 4
print("\n The 1st 12 elements of column 2 and 4 are: \n",data_df.iloc[0:12,2:4])

# Import matplotlib
import matplotlib.pyplot as plt

# Use plot and show functions to create a graph
x = np.array((1,2,3,4,5,6,7,8,9,10))
y = x*2

print (x)

plt.plot(x,y)
plt.show()

# Add labels and title to the graph
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Graph")
plt.plot(x,y)
plt.show()

# Plot using 3 variables on a single graph
z = x+4
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Graph with 3 variables")
plt.plot(x,y)
plt.plot(x,z)
plt.show()

# Change the color linestyle and linewidth of the graph
plt.plot(x,y, color='red', linestyle='--', linewidth=2, label='y=2x')
plt.plot(x,z, color='blue', linestyle='-.', linewidth=2, label='z=x+4')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Graph with 3 variables")
plt.show()

# Plot using subplot
plt.subplot(1,2,1)
plt.plot(x,y, color='red', linestyle='--', linewidth=2, label='y=2x')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("plotting values of x,y,z")
plt.subplot(1,2,2)
plt.plot(x,z, color='blue', linestyle='-.', linewidth=2, label='z=x+4')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("plotting values of x,y,z")
plt.show()

# Print the marks of students w.r.t their names using Dictionary
marks_dic = {"Name":["Aqsa","Ali","Sara","Hina","Ahmed","Fatema","Amna", "Ayesha", "Afsa","Abdul"],
"Marks":[85,90,78,88,92,76,81,79,95,89],}
marks_df = pd.DataFrame(marks_dic)
print("\n The marks dataframe is: \n",marks_df)
plt.bar(marks_df["Name"], marks_df["Marks"], color='purple')
plt.xlabel("Names")
plt.ylabel("Marks")
plt.title("Marks of Students")
plt.xticks(rotation=45)
plt.show()

# Plot a horizontal bar graph

plt.barh(marks_df["Name"], marks_df["Marks"], color='orange')
plt.xlabel("Marks")
plt.ylabel("Names")
plt.title("Marks of Students")
plt.xticks(rotation=45)
plt.show()

# Plot using scatter function
plt.scatter(marks_df["Name"], marks_df["Marks"], color='green')
plt.xlabel("Names")
plt.ylabel("Marks")
plt.title("Marks of Students")
plt.xticks(rotation=45)
plt.show()

# Plot a histogram
plt.hist(marks_df["Marks"], bins=5, color='cyan', edgecolor='black')
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Histogram of Marks")
plt.show()

# Demonstrate the use boxplot
l1 = [85,90,78,88,92,76,81,79,95,89]
l2 = [75,80,70,68,82,74,77,73,85,79]
l3 = [65,70,60,58,72,64,67,63,75,69]

data = list([l1,l2,l3])
plt.boxplot(data, vert=True, patch_artist=True, labels=['Class A', 'Class B', 'Class C'])
plt.xlabel("Classes")
plt.ylabel("Marks")
plt.title("Boxplot of Marks")
plt.show()

# Demonstrate the use of violin plot
plt.violinplot(data, vert=True, showmeans=True, showmedians=True)
plt.show()

# piechart
plt.pie(marks_df["Marks"], labels=marks_df["Name"], autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart of Marks")
plt.show()




