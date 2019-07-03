import matplotlib.pyplot as plt
# from CDRLib import *
# import sys
# print(sys.path)
# input()
from CDRLib import *

# Input CSV data
print("This is a program for Call Detail Record analysis")
print("Our demonstration data is input.csv")
name = "input.csv"  # a demonstration CSV file

my_name = "call2.csv" # my actual call history data

# Use a customized readCSV function to read CSV file

actualList = readmyCSV(my_name)
input()

# myList = readCSV(name)
# input()

# Using a helper func to recall a class "call()"
data = [makecall(i) for i in actualList]

# Analysis options
print("There are five data analysis are available\n")
print("A: The amount and ratio of Network ERROR analysis")
print("B: The amount and ratio of Network ERROR analysis by timezone")
print("C: The amount of call time analysis by timezone")
print("D: Call time and cost analysis in different days\n")
print("E: Call favourite contact at given hour of the day \n")
# Decision for User
end = "y"
while end == "y":
    ans = input("Please choose one: (A/B/C/D/E)?")
    if ans == "A":
        networkERR(data)
    elif ans == "B":
        networkERRbyzone(data)
    elif ans == "C":
        call_by_time(data)
    elif ans == "D":
        call_by_day(data)

    elif ans == 'E':
        call_by_dynamic_contact(data)

    end = input("\n keep analyzing (y/n): ")
print("Thank you~")

# END OF PROGRAM


















