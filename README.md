# Phase 5 Project


## Project Overview

For this project, I used several regression models to model the data from SyriaTel and predict if their customer will churn the plan or not.



### Business Problem

Build a classifier to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications company. Note that this is a binary classification problem.

Most naturally, your audience here would be the telecom business itself, interested in losing money on customers who don't stick around very long. Are there any predictable patterns here?


## Plan
Since the SyriaTel Customer Churn is a binary classification problem problem, I will try to use several different algorithms to fit the data and select one of the best one. The algorithms I will try include Logistic Regression, k-Nearest Neighbors, Decision Trees, Random Forest, Support Vector Machine. The target of the data we need to fit is the column 'churn'. The features of the data is the other columns in dataframe. However, when I load the data file into dataframe, i found some of the columns are linear correlated with each other. I need to drop one of them. We need to polish the data first.


##### Looking at the dataframe, I need to steply polish some features and remove some of the columns:
  1. The pairs of features inclued (total night minutes and total night charges), (total day minutes and total night   charges), (total night minutes and total night charges), (total intl charge and total intl minutes) are high correlated with each other. I need to remove one in each columns.
  2. All the phone numbers are unique and act as id. So it should not related to the target. I will remove this feature.
  3. The object columns will be catalized.

![fig1](https://raw.githubusercontent.com/sachenl/project5/main/images/ezgif.com-gif-maker.gif)

The above figures show that there are multipal columns contain some outlier data. I then collected all the columns and remove the outlier by 1.5 x   IQR


![fig2](https://raw.githubusercontent.com/sachenl/dsc-phase-3-project/main/images/fig2.png)



 
 
