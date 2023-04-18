#-------------------------------------------------------------------------
# AUTHOR: Tomes Christopher
# FILENAME: bagging_random_forest.py
# SPECIFICATION: This program takes in optdigits.tra and optdigits.tes and attempts to create 3 models; a decision tree, an ensembled method via bagging, and a random forest, then compairs the accuracy
# FOR: CS 4210- Assignment #4
# TIME SPENT: 2.5 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv
dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
with open('optdigits.tra', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      dbTraining.append(row)
#reading the test data from a csv file and populate dbTest
#--> add your Python code here
with open('optdigits.tes', 'r') as csvfile:
   reader = csv.reader(csvfile)
   for row in reader:
      dbTest.append(row)
#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
for i in range(len(dbTest)):
    classVotes.append([0,0,0,0,0,0,0,0,0,0])

print("Started my base and ensemble classifier ...")

accuracy = 0
for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample


   bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

   #populate the values of X_training and y_training by using the bootstrapSample
   #--> add your Python code here
   X_training = []
   y_training = []
   for row in bootstrapSample:
      # extract the features and label for this row
      features = [int(x) for x in row[:-1]]
      label = int(row[-1])
    
      # add the features to X_training
      X_training.append(features)
    
      # add the label to y_training
      y_training.append(label)


   #fitting the decision tree to the data
   clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
   clf = clf.fit(X_training, y_training)
   correct = 0
   total = len(dbTest)
   for i, testSample in enumerate(dbTest):
      
      # make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      prediction = clf.predict([testSample[:-1]])[0] 
      classVotes[i][prediction]+=1
      
      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         #--> add your Python code here
         # calculate the accuracy of the first base classifier
         if(int(prediction) == int(testSample[-1])):
            correct+=1
            #print('prediction: ' + str(prediction) + ' true Label: ' + str(testSample[-1]))
        

   if k == 0: #for only the first base classifier, print its accuracy here
     #--> add your Python code here
     accuracy = correct / total
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")
     correct = 0

#now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
#--> add your Python code here
for i, c_votes in enumerate(classVotes):
   max_val = 0
   for j in range(len(c_votes)):
      if c_votes[j] > c_votes[max_val]:
         max_val = j
   if(int(max_val) == int(dbTest[i][-1])):
      correct+=1
accuracy = correct / len(classVotes)
#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before
X_training = []
y_training = []
for row in dbTest:
    # Extract the vals
    values = row[:-1]
    last_value = row[-1]
    # Add the vals to the first n-1 columns to
    X_training.append(values)
    # Add last column to Y
    y_training.append(last_value)
#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
correct = 0
for i, testSample in enumerate(dbTest):
   prediction = clf.predict([testSample[:-1]])[0]

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
   if(int(prediction) == int(testSample[-1])):
            correct+=1
   else:
      print("Incorrect Prediction") 
#printing Random Forest accuracy here
accuracy = correct/len(dbTest)
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
