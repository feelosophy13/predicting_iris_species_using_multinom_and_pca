###### Predicting Iris Species with Multinomial Regression and principal Component Analysis



###### data preparation

#### initial setup
## remove all global variables
rm(list = ls())

## set working directory
setwd('/Users/hawooksong/Desktop/predicting_iris_species_using_multinom_and_pca')

## import library paths
library(nnet)  # for multinom()
library(caTools)  # for sample.split()
library(psych)  # for describeBy()

## import data
data(iris)

## lower column names
colnames(iris) <- tolower(colnames(iris))


#### split data into train and test
set.seed(123)
split <- sample.split(iris$species, SplitRatio=2/3)
train <- iris[split==TRUE, ]
test <- iris[split==FALSE, ]

dim(train)
dim(test)
table(train$species)
table(test$species)


#### exploratory analysis
## sepal and petal measurements means by species
setosa.mean <- apply(train[train$species=="setosa", -5], 2, mean)
versicolor.mean <- apply(train[train$species=="versicolor", -5], 2, mean)
virginica.mean <- apply(train[train$species=="virginica", -5], 2, mean)
means <- rbind(setosa.mean, versicolor.mean, virginica.mean)
means

## other comprehensive summary statistics by species
train$species <- as.factor(train$species)
describeBy(train, group=train$species)

## pairs for all metrics
pairs(train[, 1:4], col=train$species, upper.panel=NULL)

## correlations (should be used with caution since there are only a few dozen data points)
species <- as.character(unique(train$species))
for (s in species) {
  subDF <- subset(train, species==s)
  corMatrix <- round(cor(subDF[ , 1:4]), 2)
  corMatrix[upper.tri(corMatrix)] <- NA
  cat(s)
  print(corMatrix)
  cat('------------------------\n')
}



###### model building part 1: using the original features
#### build a multinomial regression model using the original features
model <- multinom(species ~ ., data=train)
summary(model)


#### predict on test data using the model that uses original features
preds <- predict(model, newdata=test)
conf_matrix <- table(preds, test$species)
conf_matrix



###### creating principal components

#### create prcomp object
features_train <- as.matrix(train[ , -5])  # original features
prcomp <- prcomp(features_train, retx=TRUE, center=TRUE, scale=TRUE)  # prcomp object


#### summary
class(prcomp)
summary(prcomp)
plot(prcomp, type='l')

#### original features vs. principal components
pc_train <- prcomp$x

head(features_train)
head(pc_train)


#### create a new training df with principal components (instead of with the original features)
train_pca <- as.data.frame(pc_train)
train_pca$species <- train[ , 5]

head(train)
head(train_pca)


#### build a new test df with principal components (instead of with the original features)
test_pca <- predict(prcomp, newdata=test[ , -5])  # matrix
test_pca <- as.data.frame(test_pca)  # data frame
test_pca$species <- test$species

head(test)
head(test_pca)



###### model building part 2: using the principal components
#### build a multinomial regression model
model_pca <- multinom(species ~ PC1 + PC2, data=train_pca)
summary(model_pca)



#### predict using a model that uses principal components
preds_pca <- predict(model_pca, newdata=test_pca)
conf_matrix_pca <- table(preds_pca, test$species)
conf_matrix_pca



###### good reads
# http://stackoverflow.com/questions/10876040/principal-component-analysis-in-r
# http://stats.stackexchange.com/questions/72839/how-to-use-r-prcomp-results-for-prediction
# http://stackoverflow.com/questions/1805149/how-to-fit-a-linear-regression-model-with-two-principal-components-in-r





