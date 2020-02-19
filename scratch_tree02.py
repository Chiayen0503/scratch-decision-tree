#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:54:59 2019

@author: ChiaYen
"""
# credit by:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/?fbclid=IwAR0wOipSc8A5DVx0_8RtEbkNRbBB_WQGtCIjk0NVhMbqsj7ghCHZBbUx4C8
import pandas as pd
from matplotlib import pyplot as plt


plt.style.use('ggplot')


df = pd.read_csv('breast-cancer.data.txt',sep=",")
df.columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
df.head() #check first few row data
df["deg-malig"] = df["deg-malig"].astype(object)

df.shape #check shape: (285, 10)
pd.isnull(df).values.any() #check missing value: False
# check "?" value, in this case we already know it happens in stalk-root (feature 11th)

df['class'].value_counts() #check if the data is balanced 

[df[x].unique().shape[0] for x in df.columns] 

Y = df['class']
X = df[df.columns[1:]]


X = pd.get_dummies(X)
Y = Y.apply(lambda x: 0 if x=='no-recurrence-events' else 1) # 1 is recurrence-events, i.e, target

df = pd.concat([X, Y], axis=1, join_axes=[X.index])

import numpy as np
df = np.array(df)
df = df.astype(np.int)


df = list(df)

df = np.array(df).tolist()

from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate testing score

def TP(actual, predicted):
	TP = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]== 1:
			TP += 1
	return TP

def FP(actual, predicted):
	FP = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i] and actual[i] == 0:
			FP += 1
	return FP

def TN(actual, predicted):
	TN = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i] == 0:
			TN += 1
	return TN

def FN(actual, predicted):
	FN = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i] and actual[i] == 1:
			FN += 1
	return FN

def recall_metric(TP,FN):
    if TP+FN != 0:
        recall = TP / (TP+FN) * 100 
    else:
        recall = "NA"
    return recall

def precision_metric(TP,FP):
    if TP+FP != 0:
        precision = TP / (TP+FP) * 100
    else:
        precision = "NA"
    return precision

def accuracy_metric(TP,TN,FN,FP):
    accuracy = (TN+TP) / (TP+TN+FN+FP) * 100     
    return accuracy 

def F1_score(precision, recall):
    if precision + recall != 0 and precision != "NA" and recall != "NA":
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = "NA"
    return F1



# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	scores02 = list()
	scores03 = list()
	scores04 = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]

		TPA = TP(actual, predicted)
		FNA = FN(actual, predicted)   
		FPA = FP(actual, predicted)   
		TNA = TN(actual, predicted) 
        
		recall = recall_metric(TPA,FNA)  
		precision = precision_metric(TPA,FPA)  
		accuracy = accuracy_metric(TPA,TNA,FNA,FPA)
		F1 = F1_score(precision, recall)      
        
		scores.append(accuracy)   
		scores = [x for x in scores if str(x) != 'NA'] 
        
		scores02.append(precision)
		scores02 = [x for x in scores02 if str(x) != 'NA'] 
        
		scores03.append(recall)  
		scores03 = [x for x in scores03 if str(x) != 'NA'] 

		scores04.append(F1)
		scores04 = [x for x in scores04 if str(x) != 'NA'] 
        
		AVG_accuracy = (sum(scores)/float(len(scores)))
		AVG_precision = (sum(scores02)/float(len(scores02)))
		AVG_recall = (sum(scores03)/float(len(scores03)))   
		if len(scores04) != 0:
			AVG_F1 = (sum(scores04)/float(len(scores04)))
		else:
			AVG_F1 = 1000  # can't define it as NA because it can't fit the following output format. Will change the value to NA before writing a file
		AVG_accuracy = int(np.around(AVG_accuracy, decimals=0, out=None))
		AVG_precision = int(np.around(AVG_precision, decimals=0, out=None))
		AVG_recall  = int(np.around(AVG_recall, decimals=0, out=None))
		AVG_F1  = int(np.around(AVG_F1, decimals=0, out=None))    
        
	return AVG_accuracy, AVG_precision, AVG_recall, AVG_F1

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)




import time 
with open('scratch_testing10-02.txt', 'w') as f:
    for i in range(2):
        start = time.time()
        a, b, c, d = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
        a = str(a)
        b = str(b)
        c = str(c)
        if d == 1000:   # shange undefined F1 as NA
            d = "NA"
        else:
            d = str(d)      
        stop = time.time()
        e = np.around(stop-start, decimals=4, out=None)
        e = str(e)
        print(repr(a).rjust(2), repr(b).rjust(3), repr(c).rjust(4), repr(d).rjust(5), repr(e).rjust(6), file=f)
        print(e)
