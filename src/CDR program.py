import matplotlib.pyplot as plt
# from CDRLib import *
from CDR_Analysis.CDRLib import *

# Input CSV data
print("This is a program for Call Detail Record analysis")
print("Our demonstration data is input.csv")
name = "input.csv"  # a demonstration CSV file

# Use a customized readCSV function to read CSV file
myList = readCSV(name)

# Using a helper func to recall a class "call()"  
data=[makecall(i) for i in myList]  

# Analysis options
print("There are four data analysis are available\n")
print("A: The amount and ratio of Network ERROR analysis")
print("B: The amount and ratio of Network ERROR analysis by timezone")
print("C: The amount of call time analysis by timezone")
print("D: Call time and cost analysis in different days\n")
# Decision for User
end = "y"
while end =="y":
    ans = input("Please choose one: (A/B/C/D)?")
    if ans == "A":
        networkERR(data)
    elif ans == "B":
        networkERRbyzone(data)
    elif ans == "C":
        call_by_time(data)
    elif ans == "D":
        call_by_day(data)
        
    end=input("\n keep analyzing (y/n): ")
print("Thank you~")
        
        




#END OF PROGRAM


















