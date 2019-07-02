# CDR_RecommendationEngine
To build a recommendation engine which recommends whom to call at the given hour of the day based on users past call record history


/*
Date : 18th June, 2019
Project Title : Recommendation System for slotwise Call Data Analysis
Author : Dhwani Mehta
Designation : Artificial Intelligence engineer at Silver Touch Technologies 
Project Mentor : Ritesh Patel, Rushiraj Yadav
*/

Files : 
A) Program Files
program.py (Program written for dummy call history data from internet)
my_code.py (Program written for my own call history data obtained from sharkID history)
CDR_Analysis.py 	(Contains all the functions for data analysis as well as training, validating data)

B) Data Files
input.csv	(csv file containing CDR data from internet)
call2.csv  (call history data from my own call history)

Command to run the code : python my_code.py / python program.py
Algorithm used : Logistic Regression for classification 

->divide data into 6 slots of the day (each slot is 4 hourly) for outgoing data
->Append slot in csv data
-> X = time slot 		Y = contact number of to be called person 
-> Divide data into train_test set 
-> Apply LogReg for predictions
-> Use predict_proba to find probability of every class (here I have 65 distinct contacts so 65 classes)
-> compute slotwise highest 6 probabilities and get corresponding class and then contact number based on indexing 
-> Map recommendations with slot using zip functions
->Calculate slotwise TPR and than Accuracy


 

