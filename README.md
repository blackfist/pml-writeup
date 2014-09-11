---
title: "Practical Machine Learning Assignment"
author: "Kevin Thompson"
date: "September 9, 2014"
output: html_document
---

# Getting and Cleaning Data for analysis
There are several columns that should just be removed because they provide almost no value to the machine learning algorithm. For example, the time and date
are not relevant to the prediction and there are columns that are almost all the same value. These columns should be removed entirely. We are provided with a
training data set and a testing data set, however the testing data set is a paltry 20 rows. So we're going to divide the training data up a bit further and
use a smaller part for training and and a larger sample to validate the model.

```r
library(caret)
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

# Remmove the times and user_name
training <- training[,-c(1,2,3,4,5)]
testing <- testing[,-c(1,2,3,4,5)]

# remove the columsn with near zero variance. I remove the same columns from the testing set as were taken from the training set to
# ensure that the two data sets are consistent
nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing <- testing[,-nzv]

# There are a lot of columns that are NA for every value. Let's just remove them since most
# machine learning algorithms don't like NA values anyway.
mostly_na <- apply(training, 2, function(x) { sum(is.na(x)) } )
training <- training[,which(mostly_na==0)]
testing <- testing[,which(mostly_na==0)]

# Since I'm slashing columns out left and right, why not take out any 
# highly correlated variables
muchCor <- findCorrelation(cor(training[, 1:dim(training)[2]-1]), cutoff=0.8)
training <- training[,-muchCor]
testing <- testing[,-muchCor]

# Finally, we will only keep the complete cases from the two data sets
# instead of removing these cases we might have to do some of that impugning data.
training <- training[complete.cases(training),]
testing <- testing[complete.cases(testing),]

# Now divide the training set up into modelTraining and modelTesting
randomSelection <- createDataPartition(training$classe, p = 0.7, list = FALSE)
modelTraining <- training[randomSelection, ]
modelTesting <- training[-randomSelection, ]

# Final dimensions of modelTraining data set
dim(modelTraining)
```

```
## [1] 13737    42
```

```r
# Final dimension fo testing data set
dim(testing)
```

```
## [1] 20 42
```

# Let's train some models
We're going to train several models and see if any of them are better at predicting outcomes on the testing data. We will train random forest, linear discriminant analysis, and boosted trees. These take a while to make and I didn't want to
rebuild them every time I ran this so I save them off to the file system. This section of code only ends up running if the saved models are not found.

```r
if (!file.exists("rfModel.save")) {
  rfModel <- train(classe ~ ., data=modelTraining, method="rf", verbose=FALSE)
  save(rfModel, file="rfModel.save")
} else {
  load("rfModel.save")
}

if (!file.exists("gbmModel.save")) {
  gbmModel <- train(classe ~ ., data=modelTraining, method="gbm", verbose=FALSE)
  save(gbmModel, file="gbmModel.save")
} else {
  load("gbmModel.save")
}

if (!file.exists("ldaModel.save")) {
  ldaModel <- train(classe ~ ., data=modelTraining, method="lda", verbose=FALSE)
  save(ldaModel, file="ldaModel.save")
} else {
  load("ldaModel.save")
}

if (!file.exists("rpartModel.save")) {
  rpartModel <- train(classe ~ ., data=modelTraining, method="rpart")
  save(rpartModel, file="rpartModel.save")
} else {
  load("rpartModel.save")
}
```

# Let's predict something
With the models trained, we can do some predicting and see how we did.

```r
rfPredictions <- predict(rfModel, modelTesting)
gbmPredictions <- predict(rfModel, modelTesting)
ldaPredictions <- predict(ldaModel, modelTesting)
rpartPredictions <- predict(rpartModel, modelTesting)
```
### Performance of Random Forest

```r
confusionMatrix(rfPredictions, modelTesting$classe)$overall[1]
```

```
## Accuracy 
##   0.9973
```
### Performance of Linear Discriminant Analysis

```r
confusionMatrix(ldaPredictions, modelTesting$classe)$overall[1]
```

```
## Accuracy 
##   0.6639
```
### Performance of Boosted Trees

```r
confusionMatrix(gbmPredictions, modelTesting$classe)$overall[1]
```

```
## Accuracy 
##   0.9973
```
### Performance of Tree

```r
confusionMatrix(rpartPredictions, modelTesting$classe)$overall[1]
```

```
## Accuracy 
##   0.5319
```


