---
title: "Prediction Assignment"
output: 
  html_document:
    keep_md: true
---



#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available  [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

#### Approach

Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

- exactly according to the specification (Class A) 
- throwing the elbows to the front (Class B) 
- lifting the dumbbell only halfway (Class C) 
- lowering the dumbbell only halfway (Class D) 
- throwing the hips to the front (Class E)

#### Objectives 

The objective is to predict the manner in which the participants did the exercise (outcome variable: classe).
Two models will be tested:  
  
- Random Forest
- Decision Tree. 

The model with the highest accuracy will be chosen as the final model.

#### Data

The data are available [here](http://groupware.les.inf.puc-rio.br/har).
The training data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).
The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).


Let's load the libraries and get the data.


```r
library(randomForest); library(caret); library(rpart); 
library(rpart.plot); library(rattle)
```


```r
training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""), stringsAsFactors = TRUE)
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""), stringsAsFactors = TRUE)
```

#### Data pre-processing

We first remove the unnecessary variables and the NAs.

```r
training = training[,-c(1:6)]
training <- training[,!(colSums(is.na(training))>=0.9*nrow(training))]
```

We randomly split the data into two sets without replacement as follows: 

* training data (60% of the original data set)
* testing data (40% of the original data set)


```r
set.seed(1988)
index <- createDataPartition(training$classe, p=0.6, list=FALSE)
t_train <- training[index, ]
t_test <- training[-index, ]
```

#### Model 1: Random Forest

The first model is random forest:


```r
modFit <- randomForest(classe ~ ., t_train, method = "class")
predictRF <- predict(modFit, t_test, type = "class")
confusionMatrix(predictRF, t_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    8    0    0    0
##          B    0 1510    4    0    0
##          C    0    0 1362   19    0
##          D    0    0    2 1267   10
##          E    1    0    0    0 1432
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9925, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9947   0.9956   0.9852   0.9931
## Specificity            0.9986   0.9994   0.9971   0.9982   0.9998
## Pos Pred Value         0.9964   0.9974   0.9862   0.9906   0.9993
## Neg Pred Value         0.9998   0.9987   0.9991   0.9971   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1925   0.1736   0.1615   0.1825
## Detection Prevalence   0.2854   0.1930   0.1760   0.1630   0.1826
## Balanced Accuracy      0.9991   0.9970   0.9963   0.9917   0.9965
```

#### Model 2: Decision tree

The second model is a decision tree:


```r
modFit2 <- rpart(classe ~ ., data = t_train, method = "class")
predict2 <- predict(modFit2, t_test, type = "class")
confusionMatrix(predict2, t_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2012  258   60  123   29
##          B  115  969   73   31   92
##          C   28   88 1106   49    6
##          D   59  148  101  986  106
##          E   18   55   28   97 1209
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8007          
##                  95% CI : (0.7916, 0.8095)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.747           
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9014   0.6383   0.8085   0.7667   0.8384
## Specificity            0.9163   0.9509   0.9736   0.9369   0.9691
## Pos Pred Value         0.8106   0.7570   0.8661   0.7043   0.8593
## Neg Pred Value         0.9590   0.9164   0.9601   0.9535   0.9638
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2564   0.1235   0.1410   0.1257   0.1541
## Detection Prevalence   0.3163   0.1631   0.1628   0.1784   0.1793
## Balanced Accuracy      0.9089   0.7946   0.8910   0.8518   0.9038
```

#### Comparison between models

The random forest model has an accuracy of 0.9944 while the decision tree model has 0.8007. Therefore we will use the first model to predict the actual data.

#### Predictions

We pre-process the data in the same matter as above and then predict.


```r
testing <- testing[, -c(1:6)]
testing <- testing[, !(colSums(is.na(testing))>=0.9*nrow(testing))]
pred_test <- predict(modFit, testing, type = "class")
pred_test
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
