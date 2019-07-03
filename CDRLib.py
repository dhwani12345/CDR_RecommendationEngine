#   This is a lib for Call Detail Records
#   including function of reading CSV file, call time analysis and a class

import csv
import matplotlib.pyplot as plt
import copy
from collections import Counter
from itertools import groupby
import collections
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


from sklearn.metrics import confusion_matrix


##################################################
# read CSV function
##################################################

def readCSV(name):
    with open(name,"r") as f:
        read = csv.reader(f)
        myList = [i for i in read]
        myList.remove(myList[0])

        # Add Slot column to input.csv file
        for item in myList:
            item.append(get_slot(item))
        # print("MYLIST ........ ::", myList)

        #Sorting data as per the date of the week as well as time consideration
        dfObj = pd.DataFrame(myList)
        # print("DFOBJ ::",dfObj)
        dfObj.sort_values(2, axis=0, ascending=True,
                         inplace=True, na_position='last')
        # print("DFOBJ ::", dfObj)

        myList.append(dfObj.values.tolist())
        myList = dfObj.values.tolist()
        # print("my List after is ::", myList)

        # # Write sorted data as per the date and time into another file
        with open("Sorted_DATA.csv", "w+") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(myList)

        # Make train-test split to the sorted data
        # ONLY THE CALLED person, call time and duration are imp field rest all fields to be dropped
        # main_df = pd.read_csv("Sorted_DATA.csv")
        main_df = pd.read_csv("Sorted_Data_2.csv")
        train_df = main_df[:856]
        test_df = main_df[857:]

        # print("Train ::", train_df)
        #
        # temp = len(train_df.iloc[:,1].unique())
        # print("TEMP ::", temp)

        X_train = np.array(train_df.iloc[:,-1])
        X_train = X_train.reshape(-1,1)
        Y_train = np.array(train_df.iloc[:,1].to_list())

        # X_train = np.array(train_df.iloc[:, -1].astype(str))
        # Y_train = np.array(train_df.iloc[:, 1].astype(str))

        # print("Type of Y_train data is ::", train_df.iloc[:,1].astype(str))
        # input()

        X_test = np.array(test_df.iloc[:, -1].to_list())
        X_test = X_test.reshape(-1, 1)
        Y_test = np.array(test_df.iloc[:, 1].to_list())

        # X_test = np.array(test_df.iloc[:, -1].astype(str))
        # Y_test = np.array(test_df.iloc[:, 1].astype(str))



        # a) Using keras model

        # model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
        # # # evaluate using 10-fold cross validation
        # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        # # results = cross_val_score(model, X_train, Y_train, cv=kfold)
        # # print("RESULTS of Keras model :", results.mean())
        # model.fit(X_train, Y_train, epochs=15)
        # print("Model rightfully fitted !!!")
        #
        # predictions = model.predict(X_test)
        # print("Predictions on keras classifier is ::", predictions)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # model = Sequential()
        # model.add(Dense(12, input_dim=1, activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        #
        # print("Model is ::", model)
        #
        # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
        #
        # # model.compile(optimizer='adam',
        # #               loss='sparse_categorical_crossentropy',
        # #               metrics=['accuracy'])
        # #
        # print("Compiled model is :", model.compile)
        #
        # model.fit(X_train, Y_train, epochs=15)
        # print("Fitted Model is :", model)
        #
        # scores = model.evaluate(X_test, Y_test)
        # print("Scores ::", scores)
        # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        #
        # predictions = model.predict(X_test)
        # print("Predictions ::", predictions)



        # # # b) Using Logistic Regression

        LogReg = LogisticRegression()
        LogReg.fit(X_train, Y_train)

        y_pred = LogReg.predict(X_test)
        print("Y_pred ::", y_pred)

        cf = confusion_matrix(Y_test, y_pred)
        tp_arr = cf.diagonal()
        print("TPR :: ", tp_arr)

        print("Confusion Metrics is ::", cf)


        mean_acc = LogReg.score(X_test, Y_test)
        print("MEAN_ACC is ::", mean_acc)

        # Using nearest neighbours algorithm

    return myList



def readmyCSV(my_name):

    with open(my_name,"r") as f:
        read = csv.reader(f)
        myList = [i for i in read]
        # print("MyList is ::", myList)
        myList.remove(myList[0])

        ###############################
        for item in myList:
            item.append(slot_for_my_data(item))
        ###############################

        dataframeObj = pd.DataFrame(myList)
        # print("Dataframe Obj is ::", dataframeObj)

        dataframeObj.sort_values(11, axis=0, ascending=True,
                         inplace=True, na_position='last')
        # print("Dataframe data sorted according to time :", dataframeObj[11])

        myList.append(dataframeObj.values.tolist())
        myList = dataframeObj.values.tolist()
        # print("my List after is ::", myList)

        with open("MY_callhistory_Sorted_DATA.csv", "w+") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(myList)

        # Make train-test split to the sorted data
        # ONLY THE CALLED person, call time and duration are imp field rest all fields to be dropped
        # main_df = pd.read_csv("Sorted_DATA.csv")
        main_df = pd.read_csv("MY_callhistory_Sorted_DATA.csv")
        train_df = main_df[26222:26849]
        # print("Train data is :: \n ", train_df)
        test_df = main_df[26849:]
        # print("Test data is :: \n ", test_df)
        #
        # temp = len(train_df.iloc[:,1].unique())
        # print("TEMP ::", temp)


        X_train = np.array(train_df.iloc[:,-1])
        # print("X_train is :: \n", X_train)
        X_train = X_train.reshape(-1,1)
        Y_train = np.array(train_df.iloc[:,8].to_list())
        # print("Y_train is :: \n", Y_train)

        # print("Unique contact numbers ::", len(list(set(Y_train))))

        X_test = np.array(test_df.iloc[:, -1].to_list())
        # print("X_test is :: \n", X_test)
        X_test2 = X_test.reshape(-1, 1)
        Y_test = np.array(test_df.iloc[:, 8].to_list())
        # print("Y_test is :: \n", Y_test)


        # # # # b) Using Logistic Regression
        #
        # LogReg = LogisticRegression()
        # LogReg.fit(X_train, Y_train)
        #
        # y_pred_prob = LogReg.predict_proba(X_test2)
        # # print("y_predict Probability is ::", y_pred_prob)
        #
        #
        # # best_n = np.argsort(y_pred_prob[0])[-6:]
        # # print("Best 6 are ::", best_n)
        #
        # y_pred = LogReg.predict(X_test2)
        # # print("y_predict is ::", y_pred)
        #
        # classes = LogReg.classes_
        # # print("Classes are :", len(classes))
        #
        # recommendations = []
        #
        # for classitem in y_pred_prob:
        #     # print("ClassItem is ::", classitem)
        #     best_6_probs = np.argsort(classitem)[-6:]
        #     # print("Best 6 are ::", best_6_probs)
        #
        #     corr_class = []
        #     for index in range(0,len(best_6_probs)):
        #         corr_class.append(classes[best_6_probs[index]])
        #
        #     # print("One of the corresponding class is :", corr_class)
        #
        #     recommendations.append(corr_class)
        #
        # # print("Recommendations List is ::", recommendations)
        #
        # recommendations_slot_mapping = list(zip(recommendations, X_test))
        # # print("MAPPING !!! ", recommendations_slot_mapping)
        #
        # calls_slot1 = []
        # calls_slot2 = []
        # calls_slot3 = []
        # calls_slot4 = []
        # calls_slot5 = []
        # calls_slot6 = []
        #
        #
        # for tuple in recommendations_slot_mapping:
        #
        #     # print("Tuple is ::", tuple)
        #
        #     if(1 in tuple):
        #         calls_slot1.append(tuple[0])
        #
        #     elif (2 in tuple):
        #         calls_slot2.append(tuple[0])
        #
        #     if (3 in tuple):
        #         calls_slot3.append(tuple[0])
        #
        #     if (4 in tuple):
        #         calls_slot4.append(tuple[0])
        #
        #     if (5 in tuple):
        #         calls_slot5.append(tuple[0])
        #
        #     if (6 in tuple):
        #         calls_slot6.append(tuple[0])
        #
        #
        # # print("call_slot1 ::", calls_slot1[0])
        # print("call_slot2 ::", calls_slot2[0])
        # print("call_slot3 ::", calls_slot3[0])
        # print("call_slot4 ::", calls_slot4[0])
        # print("call_slot5 ::", calls_slot5[0])
        # print("call_slot6 ::", calls_slot6[0])
        #
        #
        # # input()
        #
        # # print("X_test is ::", X_test)
        #
        # TP = 0
        #
        # for test_sample in range(0,len(Y_test)):
        #
        #     # print("Xtest and Ytest of sample is:", X_test[test_sample], Y_test[test_sample])
        #
        #     if(X_test[test_sample]==2 and Y_test[test_sample] in calls_slot2[0]):
        #         TP = TP + 1
        #
        #     elif(X_test[test_sample]==3 and Y_test[test_sample] in calls_slot3[0]):
        #         TP = TP + 1
        #
        #     elif(X_test[test_sample]==4 and Y_test[test_sample] in calls_slot4[0]):
        #         TP = TP + 1
        #
        #     elif(X_test[test_sample]==5 and Y_test[test_sample] in calls_slot5[0]):
        #         TP = TP + 1
        #
        #     elif(X_test[test_sample]==6 and Y_test[test_sample] in calls_slot6[0]):
        #         TP = TP + 1
        #
        # print("True Positive Rate is ::", TP)
        #
        # accuracy = TP/len(Y_test)
        # print("Accuarcy is::", accuracy)
        #
        #
        # ### number_slot_mapping = list(zip(y_pred, X_test))
        # ### print("DICT! is ::", number_slot_mapping)
        # ### input()
        #
        # # calls_slot1 = []
        # #
        # # for tuple in number_slot_mapping:
        # #
        # #     if(4 in tuple):
        # #         calls_slot1.append(tuple[0])
        # #
        # # print("Calls in slot 1 list is::", calls_slot1)
        # #
        # # unq_list = list(set(calls_slot1))
        # # print("Uniquee List is ::", unq_list)

        # # cf = confusion_matrix(Y_test, y_pred)
        # # tp_arr = cf.diagonal()
        # # print("TPR :: ", tp_arr)
        # #
        # # print("Confusion Metrics is ::", cf)
        # #
        # # mean_acc = LogReg.score(X_test, Y_test)
        # # print("MEAN_ACC is ::", mean_acc)


        ##################################################################################



        # Using KNN Classifier as Model

        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(X_train, Y_train)
        print("Fitted Model is :", knn)

        # Test the model
        y_pred_probability = knn.predict_proba(X_test2)
        print("Probabilities are ::", y_pred_probability)

        best_n = np.argsort(y_pred_probability[0])[-6:]
        print("Best 6 are ::", best_n)

        classes = knn.classes_
        print("Classes are :", len(classes))

        input()

        pred = knn.predict(X_test2)
        print("Make Predictions ::", pred)

        recommendations = []

        for classitem in y_pred_probability:
            # print("ClassItem is ::", classitem)
            best_6_probs = np.argsort(classitem)[-6:]
            # print("Best 6 are ::", best_6_probs)

            corr_class = []
            for index in range(0,len(best_6_probs)):
                corr_class.append(classes[best_6_probs[index]])

            # print("One of the corresponding class is :", corr_class)

            recommendations.append(corr_class)

        # print("Recommendations List is ::", recommendations)

        recommendations_slot_mapping = list(zip(recommendations, X_test))
        # print("MAPPING !!! ", recommendations_slot_mapping)

        calls_slot1 = []
        calls_slot2 = []
        calls_slot3 = []
        calls_slot4 = []
        calls_slot5 = []
        calls_slot6 = []


        for tuple in recommendations_slot_mapping:

            # print("Tuple is ::", tuple)

            if(1 in tuple):
                calls_slot1.append(tuple[0])

            elif (2 in tuple):
                calls_slot2.append(tuple[0])

            if (3 in tuple):
                calls_slot3.append(tuple[0])

            if (4 in tuple):
                calls_slot4.append(tuple[0])

            if (5 in tuple):
                calls_slot5.append(tuple[0])

            if (6 in tuple):
                calls_slot6.append(tuple[0])


        # print("call_slot1 ::", calls_slot1[0])
        print("call_slot2 ::", calls_slot2[0])
        print("call_slot3 ::", calls_slot3[0])
        print("call_slot4 ::", calls_slot4[0])
        print("call_slot5 ::", calls_slot5[0])
        print("call_slot6 ::", calls_slot6[0])


        # input()

        # print("X_test is ::", X_test)

        TP = 0

        for test_sample in range(0,len(Y_test)):

            # print("Xtest and Ytest of sample is:", X_test[test_sample], Y_test[test_sample])

            if(X_test[test_sample]==2 and Y_test[test_sample] in calls_slot2[0]):
                TP = TP + 1

            elif(X_test[test_sample]==3 and Y_test[test_sample] in calls_slot3[0]):
                TP = TP + 1

            elif(X_test[test_sample]==4 and Y_test[test_sample] in calls_slot4[0]):
                TP = TP + 1

            elif(X_test[test_sample]==5 and Y_test[test_sample] in calls_slot5[0]):
                TP = TP + 1

            elif(X_test[test_sample]==6 and Y_test[test_sample] in calls_slot6[0]):
                TP = TP + 1

        print("True Positive Rate is ::", TP)

        accuracy = TP/len(Y_test)
        print("Accuarcy is::", accuracy)


        input()





# # create model
	# model = Sequential()
	# model.add(Dense(12, input_dim=1, activation='relu'))
	# model.add(Dense(8, activation='relu'))
	# model.add(Dense(1, activation='sigmoid'))
	# # Compile model
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# return model


def slot_for_my_data(inputfile):
    # print("List data is ::", input_file_List_data)
    # input()

    call_slot = None

    # for item in input_file_List_data:
    # print("Item is ::", item)
    # call_time = item[2]
    call_time = inputfile[11]
    # print("Call Time is ::", call_time)
    call_hour = int(call_time[11:13])
    # print(call_hour)

    if (0 <= call_hour < 4):
        call_slot = 1
        # print("call_slot is ::", call_slot)
        return call_slot

    elif (4 <= call_hour < 8):
        call_slot = 2
        # print("call_slot is ::", call_slot)
        return call_slot

    elif (8 <= call_hour < 12):
        call_slot = 3
        # print("call_slot is ::", call_slot)
        return call_slot

    elif (12 <= call_hour < 16):
        call_slot = 4
        # print("call_slot is ::", call_slot)
        return call_slot

    elif (16 <= call_hour < 20):
        call_slot = 5
        # print("call_slot is ::", call_slot)
        return call_slot

    elif (20 <= call_hour < 24):
        call_slot = 6
        # print("call_slot is ::", call_slot)
        return call_slot

    # print("call_slot is ::", call_slot)
    return call_slot

def get_slot(input_file_List_data):

    # print("List data is ::", input_file_List_data)
    # input()

    call_slot = None

    # for item in input_file_List_data:
    # print("Item is ::", item)
    # call_time = item[2]
    call_time = input_file_List_data[2]
    # print("call time is ::", call_time)

    call_hour = int(call_time[11:13])
    # print(call_hour)

    if(0<=call_hour<4):
        call_slot = 1
        # print("call_slot is ::", call_slot)
        return call_slot

    elif(4<=call_hour<8):
        call_slot = 2
        # print("call_slot is ::", call_slot)
        return call_slot

    elif(8<=call_hour<12):
        call_slot = 3
        # print("call_slot is ::", call_slot)
        return call_slot

    elif(12<=call_hour<16):
        call_slot = 4
        # print("call_slot is ::", call_slot)
        return call_slot

    elif(16<=call_hour<20):
        call_slot = 5
        # print("call_slot is ::", call_slot)
        return call_slot

    elif(20<=call_hour<24):
        call_slot = 6
        # print("call_slot is ::", call_slot)
        return call_slot

    # print("call_slot is ::", call_slot)
    return call_slot


def CountFrequency(arr):
    return collections.Counter(arr)

##################################################
# a class to process data
##################################################
class call:

    def __init__(self,CALLER,CALLED,CALL_TIME,CONNECTION_TIME,CONNECTION_FINISH_TIME,CALL_DURATION_SEC,FINISH_REASON,COST):

        self.CALLER= CALLER
        self.CALLED = CALLED
        self.CALL_TIME = CALL_TIME
        self.CONNECTION_TIME = CONNECTION_TIME
        self.CONNECTION_FINISH_TIME = CONNECTION_FINISH_TIME
        self.CALL_DURATION_SEC = CALL_DURATION_SEC
        self.FINISH_REASON = FINISH_REASON
        self.COST = COST


    def getCALLER(self):
        return self.CALLER
    def getCALLED(self):
        return self.CALLED
    def getCALLTIME(self):
        return self.CALL_TIME    
    def getCONNECTION(self):
        return str(self.CONNECTION_TIME)[11:13]
    def getTIMEZONE(self):
        if 6<=int(str(self.CONNECTION_TIME)[11:13])<15:
            return 'Day'
        elif 15<=int(str(self.CONNECTION_TIME)[11:13])<23:
            return 'Evening'
        else:
            return 'Night'            
    def getSEC(self):
        return self.CALL_DURATION_SEC
    def getREASON(self):
        return self.FINISH_REASON
    def getCOST(self):
        if self.COST == "":
            return 0
        else :
            return float(self.COST)

        
        
#############################################

# help func
def makecall(a):
    return call(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])

# use the help
#data=[makecall(i) for i in myList]

###############################################
# Network error rate % function
###############################################

def networkERR(alist):
    errorcounter,total = 0, 0
    for i in alist:
        total +=1
        if i.getREASON()== 'NETWORK_ERROR' :
            errorcounter += 1
    print("The amount of NETWORK_ERROR calls are {}.".format(errorcounter))
    print("Total calls are {}.".format(total))
    print("NETWORK_ERROR rate is {:.1%}\n".format(errorcounter/total))
###############################################
# Network error analysis by timezone
###############################################

def networkERRbyzone(alist):
    print("This is NETWORKERROR analysis by TIMEZONE")
    DayERR,EveningERR,NightERR,total = 0,0,0,0
    for i in alist:
        total +=1
        if i.getREASON()== 'NETWORK_ERROR':
            if i.getTIMEZONE() == "Day":
                DayERR += 1
            elif i.getTIMEZONE() == "Evening":
                EveningERR += 1
            else:
                NightERR += 1

    L1 = "##NETWORK_ERROR Table##"
    L2 = "Time_zone  Counts  Rate"
    L3 = "{2:^9}  {0:^6}  {1:^4.1%}".format(DayERR,DayERR/total,"Day")
    L4 = "{2:^9}  {0:^6}  {1:^4.1%}".format(EveningERR,EveningERR/total,"Evening")
    L5 = "{2:^9}  {0:^6}  {1:^4.1%}".format(NightERR,NightERR/ total,"Night")
    table = [L1,L2,L3,L4,L5]
    ans = input("Print out on screen or output a file (s/f): ")
    if ans == "f":
        name =input("Input a file name:")
        fname = "{}.txt".format(name)
        ofile = open(fname,"w")
        for i in table:
            ofile.write(i+"\n")
        ofile.close()
    else:
        for i in table:
            print(i)


#################################################
# Call reason analysis by timezone
#################################################

def call_by_time(data):
    daycall,eveningcall,nightcall = 0,0,0
    for i in data:
        if i.getTIMEZONE() == 'Day':
            daycall += 1
        elif i.getTIMEZONE() == 'Evening':
            eveningcall += 1
        else:
            nightcall += 1
    print('There are {} day time calls'.format(daycall))
    print('There are {} evening time calls'.format(eveningcall))
    print('There are {} night time calls'.format(nightcall))
    print('********************')

    # Pie chart ploted by timezone
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Day', 'Evening', 'Night'
    sizes = [daycall,eveningcall,nightcall]
    colors = ['yellowgreen', 'gold', 'lightskyblue']
    explode = (0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()


#################################################
# Call time and cost analysis in different days
#################################################

def call_by_day(data):    
    MONtotalCALLTIME=0
    TUEtotalCALLTIME=0
    WEDtotalCALLTIME=0
    THUtotalCALLTIME=0
    FRItotalCALLTIME=0
    SATtotalCALLTIME=0
    SUNtotalCALLTIME=0

    MONtotalCOST=0
    TUEtotalCOST=0
    WEDtotalCOST=0
    THUtotalCOST=0
    FRItotalCOST=0
    SATtotalCOST=0
    SUNtotalCOST=0

    for i in data:

        if i.getCALLTIME()[0:2] == '02' :
          MONtotalCALLTIME += float(i.getSEC())
          MONtotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '03':
          TUEtotalCALLTIME += float(i.getSEC())
          TUEtotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '04' :
          WEDtotalCALLTIME += float(i.getSEC())
          WEDtotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '05':
          THUtotalCALLTIME += float(i.getSEC())
          THUtotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '06':
          FRItotalCALLTIME += float(i.getSEC())
          FRItotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '07':
          SATtotalCALLTIME += float(i.getSEC())
          SATtotalCOST += float(i.getCOST())
        elif i.getCALLTIME()[0:2] == '01' or '08':
          SUNtotalCALLTIME += float(i.getSEC())
          SUNtotalCOST += float(i.getCOST())
     
    print('MON time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(MONtotalCALLTIME,MONtotalCOST,MONtotalCOST/MONtotalCALLTIME))
    print('TUE time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(TUEtotalCALLTIME,TUEtotalCOST,TUEtotalCOST/TUEtotalCALLTIME))
    print('WED time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(WEDtotalCALLTIME,WEDtotalCOST,WEDtotalCOST/WEDtotalCALLTIME))
    print('THU time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(THUtotalCALLTIME,THUtotalCOST,THUtotalCOST/THUtotalCALLTIME))
    print('FRI time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(FRItotalCALLTIME,FRItotalCOST,FRItotalCOST/FRItotalCALLTIME))
    print('SAT time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(SATtotalCALLTIME,SATtotalCOST,SATtotalCOST/SATtotalCALLTIME))
    print('SUN time {0}; SEC cost:{1:9.2f}; per sec{2}'.format(SUNtotalCALLTIME,SUNtotalCOST,SUNtotalCOST/SUNtotalCALLTIME))
    # Pie chart ploted by days
    labels = 'Mon', 'TUE','WED','THU','FRI','SAT','SUN'
    sizes = [MONtotalCALLTIME,TUEtotalCALLTIME,WEDtotalCALLTIME,THUtotalCALLTIME,FRItotalCALLTIME,SATtotalCALLTIME,SUNtotalCALLTIME]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red','blue','green']
    explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show()


def call_by_dynamic_contact(data):

    # # TODO Make current time static i.e. run a for loop indicating every hour of the day
    # current_time = input("Please input the current time in hour/min/sec::")
    # print("Current Time is:", current_time)
    # print("Current Hour is:", current_time[0:2])
    #
    # input()

    dynamic_contact_list = []

    # for i in data:
    for current_hour in range(0, 24):

       hourly_list = []
       recommended_favourite_list = []

       # print("CallTime is :", i.getCALLTIME()[11:13])

       # for current_hour in range(0,24):
       for i in data:

           # # Note that no need to check minute and seconds, just acknowledging hour
           # # if(current_time==i.getCALLTIME()[11:13]):
           if(int(i.getCALLTIME()[11:13]) == current_hour):
               # print("Inside if !!!")
               # print(i.getCALLED())
               hourly_list.append(i.getCALLED())




       print("Hourly list is ::", hourly_list)
       input()

       freq = CountFrequency(hourly_list)
       print("FREQ is:", freq)
       for key, value in freq.items():
           print(key, " -> ", value)
       input()


       dynamic_contact_list.append(hourly_list)
       print("Dynamic contact list inside loop :", dynamic_contact_list)
       input()

    print("Dynamic Contact list outside loop ------------------------:", dynamic_contact_list)






################################################################################################
# analysis example
################################################################################################

# Get NETWORK_ERROR rate %
# networkERR(data)

# Get NETWORK_ERROR analysis by timezone
#networkERRbyzone(data)

# Get call analysis by timezone
#call_by_time(data)

# Get call analysis by day
#call_by_day(data)

# Get call analysis for favourite contact
# dynamic_contact(data)
