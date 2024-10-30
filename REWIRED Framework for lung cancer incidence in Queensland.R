#This code is based on "O:\Population-based data\Projects\National Cancer Atlas\Atlas 2.0\Urban Health Indicators project\Programs\Area-level index\Random forest\test random forest 1001.r"
#used to identify important predictors for breast cancer incidence
#Author: Kou Kou

install.packages("randomForest")
install.packages("caret")
install.packages("readstata13")
install.packages("MLmetrics")
install.packages("smotefamily")
install.packages("glmnet")

setwd("O:\\Population-based data\\Projects\\National Cancer Atlas\\Atlas 2.0\\Urban Health Indicators project\\Data\\R data\\Area-level index")
library(randomForest)
library(datasets)
library(caret)
library(dplyr)
library(readstata13)
library(MLmetrics)
library(glmnet)

#load data
data23_1619=read.csv("data23_1619.csv")
data23_1619 = subset(data23_1619, select=-X)
area1619=read.csv("area1619.csv")
area1619 = subset(area1619, select=-X)
sir23_1619=read.csv("sir_23_smoothed.csv")

####################################
# Step 1 - Missing data imputation #
####################################
# # append obs with missing values to the Balanced Dataset for Modeling 
# data_complete <- rbind(data1215_rose, data_missing)
# summary(data_complete$bc_category)

# Original data (before imputation)
original_data <- data23_1619

# Create a logical matrix indicating the locations of missing values
missing_indices <- is.na(original_data)

# Identify rows with at least one missing value
rows_with_missing <- apply(data23_1619, 1, function(x) any(is.na(x)))
# Extract the rows with at least one missing value
data_with_missing <- data23_1619[rows_with_missing, ]  #106 out of 519 rows had at least one missing values, accounts for 20.4% of the whole obs

# Identify features with at least one missing value
cols_with_missing <- apply(data23_1619, 2, function(x) any(is.na(x)))
# Extract features with at least one missing value
cols_with_missing_names <- names(data23_1619)[cols_with_missing] #183-1 out of 201-1 columns had at least one missing values, accounts for 91% of the whole features

#generate 21 boots samples for data imputation (addressing 20.4% of missing obs)
# Define the control parameters for train
train_control <- trainControl(method = "boot", number = 21) # 21 bootstrap samples
# Impute the missing values using bagging
set.seed(7777)
bagMissing <- preProcess(data23_1619, method = "bagImpute") #number of trees (boot samples) has been defined as 21
data23_1619_imputed <- predict(bagMissing, newdata = data23_1619)

#check if the imputed values are within the range of the original values
# Function to check if imputed values are within the range of original non-missing values
compare_imputed_to_original <- function(original, imputed, missing_indices) {
  for (col in colnames(original)) {
    if (any(missing_indices[, col])) {
      non_missing_range <- range(original[, col], na.rm = TRUE)
      imputed_values <- imputed[missing_indices[, col], col]
      
      # Check if imputed values are within the non-missing range
      within_range <- imputed_values >= non_missing_range[1] & imputed_values <= non_missing_range[2]
      cat("\nVariable:", col, "\n")
      cat("Non-missing range:", non_missing_range, "\n")
      cat("Imputed values within range:", all(within_range), "\n")
      if (!all(within_range)) {
        cat("Imputed values out of range:", imputed_values[!within_range], "\n")
      }
    }
  }
}
# Compare imputed values to original non-missing values
compare_imputed_to_original(original_data, data23_1619_imputed, missing_indices) #predicted missing values are within the range

# use the bagging imputation method for filling in missing data for data23_1619
data23_1619 <- predict(bagMissing, newdata = data23_1619)

#Predict missing data for the full sa2 list
#This step is away from the RF model, just need to get those values for future index generation
sir23_1619$category="Average"
sir23_1619$category[sir23_1619$sir>1 & sir23_1619$PPD>=0.6]="Exceed"
sir23_1619$category[sir23_1619$sir<1 & sir23_1619$PPD>=0.6]="Below"
table(sir23_1619$category)

data23_1619full=merge(sir23_1619, area1619,by="SA2_5DIG16")
data23_1619full <- predict(bagMissing, newdata=data23_1619full)   #this is the final data including all values for all variables.
missing_obs <- which(!complete.cases(data23_1619full)) #no missing data

# #clean the datasets
# List all objects in the environment
all_objects <- ls()
# Identify the object(s) to keep
keep_objects <- c("data23_1619", "data23_1619full")
# Remove all objects except those to keep
rm(list = setdiff(all_objects, keep_objects))
# Verify the remaining objects
ls()
# remove(bagMissing)
# remove(sir23_1619)
# remove(area1619)

##################################
# Step 2 - normalize predictors #
##################################
# convert to a z scale for all the predictors
data23_1619X <- subset(data23_1619, select=-category)
set.seed(7777)
zModel <- preProcess(data23_1619X, Method = c("center", "scale"))
data23_1619X <- predict(zModel, newdata = data23_1619X)
#add response variable to the dataset
data23_1619<- cbind(data23_1619$category, data23_1619X)
#make sure the variable names are correct
names(data23_1619)[names(data23_1619)=="data23_1619$category"] = "category"
data23_1619$category=as.factor(data23_1619$category)
remove(data23_1619X)
remove(zModel)

##############################
# Step 3 - feature selection #
##############################
#option 1 - Recursive feature elimination (RFE) - select important features using backwards feature selection#
subsets <- seq(10, 196, by=10) 
set.seed(7777)
#Controlling the Feature Selection Algorithms
rfeCtrl <- rfeControl(functions = rfFuncs,
                      method = "boot",
                      rerank = TRUE,
                      verbose = FALSE)
rfProfile <- rfe(x=data23_1619[,2:197],
                 y=data23_1619[,1],
                 sizes = subsets,
                 rfeControl = rfeCtrl,
                 metric="Kappa")
rfProfile

#kappa peaks at 30. select more precisely again
subsets <- seq(20, 40, by=5)
set.seed(7777)
#Controlling the Feature Selection Algorithms
rfeCtrl <- rfeControl(functions = rfFuncs,
                      method = "boot",
                      rerank = TRUE,
                      verbose = FALSE)
rfProfile <- rfe(x=data23_1619[,2:197],
                 y=data23_1619[,1],
                 sizes = subsets,
                 rfeControl = rfeCtrl,
                 metric="Kappa")
rfProfile

#kappa peaks at 20. select more precisely again
subsets <- seq(10, 30, by=2)
set.seed(7777)
#Controlling the Feature Selection Algorithms
rfeCtrl <- rfeControl(functions = rfFuncs,
                      method = "boot",
                      rerank = TRUE,
                      verbose = FALSE)
rfProfile <- rfe(x=data23_1619[,2:197],
                 y=data23_1619[,1],
                 sizes = subsets,
                 rfeControl = rfeCtrl,
                 metric="Kappa")
rfProfile


# 22 features has the best model fit
saveRDS(rfProfile, file = "features_RFE.rds")
# Extract the important features selected by RFE
features_rfe <- rfProfile$optVariables
# Create a subset of trainingSet with only the selected features
data23_1619_rfe <- data23_1619[, c("category", features_rfe)]

# option 1.1 force the number of features less than 15 #
subsets <- seq(2, 15, by=1)
set.seed(7777)
#Controlling the Feature Selection Algorithms
rfeCtrl <- rfeControl(functions = rfFuncs,
                      method = "boot",
                      rerank = TRUE,
                      verbose = FALSE)
rfProfile <- rfe(x=data23_1619[,2:197],
                 y=data23_1619[,1],
                 sizes = subsets,
                 rfeControl = rfeCtrl,
                 metric="Kappa")
rfProfile

saveRDS(rfProfile, file = "features_RFE_simp.rds")

# Extract the important features selected by RFE
features_rfe_simp <- rfProfile$optVariables
# Create a subset of trainingSet with only the selected features
data23_1619_rfe_simp <- data23_1619[, c("category", features_rfe_simp)]

# Option 2 - Feature Selection with Least Absolute Shrinkage and Selection Operator (LASSO) #
set.seed(7777)
# Prepare the data for glmnet
x <- as.matrix(data23_1619[, 2:197])
y <-data23_1619[, 1]
# LASSO model with cross-validation
lasso_model <- cv.glmnet(x, y, alpha = 1, family = "multinomial", maxit = 200000)
# Extract the coefficients for the model at the optimal lambda
lasso_coefs <- coef(lasso_model, s = "lambda.min")
# Combine the coefficients into a single matrix, excluding the intercept (first row)
lasso_coefs_matrix <- do.call(cbind, lapply(lasso_coefs, function(x) x[-1,]))
# Identify non-zero coefficients across any outcome level
nonzero_indices <- which(apply(lasso_coefs_matrix, 1, function(x) any(abs(x) > 0)))
# Select the feature names corresponding to non-zero coefficients
features_lasso <- colnames(data23_1619)[nonzero_indices + 1]
# Create a subset of the dataset with only the selected features from LASSO
data23_1619_lasso <- data23_1619[, c("category", features_lasso)]

# Option 2.1 - Less features using LASSO by 
#1. remove features only have non-zero coefficients for the "average" class
# Identify features that have non-zero coefficients only in the "average" column
exclude_indices <- which((lasso_coefs_matrix[, 1] != 0) & 
                           (lasso_coefs_matrix[, 2] == 0) & 
                           (lasso_coefs_matrix[, 3] == 0))
# Identify features to keep (i.e., features not in the exclude_indices)
include_indices <- setdiff(seq_len(nrow(lasso_coefs_matrix)), exclude_indices)
#remove features only have non-zero coefficients for the "average" class
nonaverage_coefs <- lasso_coefs_matrix[include_indices, , drop = FALSE]
# Select the feature names after exclude features only have non-zero coefficients for the "average" class
features_nonaverage <- colnames(data23_1619)[include_indices + 1]
# Identify non-zero coefficients across "below" and "exceed" outcome level
nonzero_indices_elastic_net <- which(apply(nonaverage_coefs, 1, function(x) any(abs(x) > 0)))
# Select the feature names corresponding to coefficients > 0 for below and exceed
features_lasso_simp <- features_nonaverage[nonzero_indices_elastic_net]
# Create a subset of the dataset with only the selected features from Elastic Net
data23_1619_lasso_simp <- data23_1619[, c("category", features_lasso_simp)]

#Option 2.2 - Less features selected by setting the shred hold of selected coefficient-Lasso
#set a threshold of 0.1
threshold <- 0.1
select_indices_lasso <- which(apply(nonaverage_coefs, 1, function(x) any(abs(x) > threshold)))
# Select the feature names corresponding to coefficients > threshold(0.1) for below and exceed
features_lasso_select <- features_nonaverage[select_indices_lasso]
# Create a subset of the dataset with only the selected features from lasso
data23_1619_lasso_select <- data23_1619[, c("category", features_lasso_select)]

# option 3 - Feature Selection with Elastic Net #
set.seed(7777)
# Elastic Net model with cross-validation
# Note: alpha = 0.5 is a common choice for Elastic Net, blending LASSO and Ridge penalties
#Whe alpha=1, Elastic Net becomes LASSO.
#When alpha=0, Elastic Net becomes Ridge. Ridge regression tends to keep correlated features together by shrinking their coefficients at similar rates.
#For 0<alpha<1, Elastic Net is a compromise between the two.
elastic_net_model <- cv.glmnet(x, y, alpha = 0.5, family = "multinomial", maxit = 200000)
# Extract the coefficients for the model at the optimal lambda
elastic_net_coefs <- coef(elastic_net_model, s = "lambda.min")
# Combine the coefficients into a single matrix, excluding the intercept (first row)
elastic_net_coefs_matrix <- do.call(cbind, lapply(elastic_net_coefs, function(x) x[-1,]))
# Identify non-zero coefficients across any outcome level
nonzero_indices_elastic_net <- which(apply(elastic_net_coefs_matrix, 1, function(x) any(abs(x) > 0)))
# Select the feature names corresponding to non-zero coefficients
features_en <- colnames(data23_1619)[nonzero_indices_elastic_net + 1]
# Create a subset of the dataset with only the selected features from Elastic Net
data23_1619_en <- data23_1619[, c("category", features_en)]

#Option 3.1 - Less features using EN by 
#1. remove features only have non-zero coefficients for the "average" class
# Identify features that have non-zero coefficients only in the "average" column
exclude_indices <- which((elastic_net_coefs_matrix[, 1] != 0) & 
                           (elastic_net_coefs_matrix[, 2] == 0) & 
                           (elastic_net_coefs_matrix[, 3] == 0))
# Identify features to keep (i.e., features not in the exclude_indices)
include_indices <- setdiff(seq_len(nrow(elastic_net_coefs_matrix)), exclude_indices)
#remove features only have non-zero coefficients for the "average" class
nonaverage_coefs <- elastic_net_coefs_matrix[include_indices, , drop = FALSE]
# Select the feature names after exclude features only have non-zero coefficients for the "average" class
features_nonaverage <- colnames(data23_1619)[include_indices + 1]
# Identify non-zero coefficients across "below" and "exceed" outcome level
nonzero_indices_elastic_net <- which(apply(nonaverage_coefs, 1, function(x) any(abs(x) > 0)))
# Select the feature names corresponding to coefficients > 0 for below and exceed
features_en_simp <- features_nonaverage[nonzero_indices_elastic_net]
# Create a subset of the dataset with only the selected features from Elastic Net
data23_1619_en_simp <- data23_1619[, c("category", features_en_simp)]

#Option 3.2 - Less features selected by setting the shred hold of selected coefficient-EN
#set a threshold of 0.1
threshold <- 0.1
select_indices_elastic_net <- which(apply(nonaverage_coefs, 1, function(x) any(abs(x) > threshold)))
# Select the feature names corresponding to coefficients > threshold(0.1) for below and exceed
features_en_select <- features_nonaverage[select_indices_elastic_net]
# Create a subset of the dataset with only the selected features from Elastic Net
data23_1619_en_select <- data23_1619[, c("category", features_en_select)]

#compare the selected features
print(features_rfe)
print(features_rfe_simp)
print(features_lasso)
print(features_lasso_simp)
print(features_lasso_select)
print(features_en)
print(features_en_simp)
print(features_en_select)

# #clean the datasets
# List all objects in the environment
all_objects <- ls()
# Identify the object(s) to keep
keep_objects <- c("data23_1619", "data23_1619_rfe", "data23_1619_rfe_simp", "data23_1619_lasso", "data23_1619_lasso_simp","data23_1619_lasso_select", "data23_1619_en", "data23_1619_en_select", "data23_1619_en_simp", "data23_1619full")
# Remove all objects except those to keep
rm(list = setdiff(all_objects, keep_objects))
# Verify the remaining objects
ls()

#information of the three feature selection methods:
# LASSO: Removes correlated features by shrinking coefficients to zero, which can help reduce multicollinearity.
# Elastic Net: Combines LASSO and Ridge penalties to handle multicollinearity by selecting groups of correlated features together.
# RFE: Selects features based on their importance but does not specifically address multicollinearity.

#################################################
# Step 4 - split into training and testing sets #
#################################################
set.seed(6789)
trainIndex <- createDataPartition(data23_1619$category, p=0.8, list = FALSE)
trainingSet_rfe <- data23_1619_rfe[trainIndex,]
trainingSet_rfe_simp <- data23_1619_rfe_simp[trainIndex,] #this one contains less features by forcing the number of features less than 20
trainingSet_lasso <- data23_1619_lasso[trainIndex,]
trainingSet_lasso_simp<- data23_1619_lasso_simp[trainIndex,]
trainingSet_lasso_select <- data23_1619_lasso_select[trainIndex,]
trainingSet_en <- data23_1619_en[trainIndex,]
trainingSet_en_simp <- data23_1619_en_simp[trainIndex,]
trainingSet_en_select <- data23_1619_en_select[trainIndex,]

testSet <- data23_1619[-trainIndex,]

# # Shuffle the training set
# set.seed(1234)
# trainingSet <- trainingSet[sample(nrow(trainingSet)), ]
# # Shuffle the test set
# testSet <- testSet[sample(nrow(testSet)), ]

#############################################
# # Oversampling minority group (SMOTE)
# library(smotefamily)
# # SMOTE: Synthetic Minority Over-sampling Technique
# # generate synthetic samples for the minority class in imbalanced datasets.
# #split trainingSet into two datasets
# 
# #For the Below group
# A <- trainingSet[trainingSet$bc_category %in% c("Average", "Below"), ]
# A$bc_category <- factor(A$bc_category, levels = c("Average", "Below"))
# summary(A$bc_category) 
# 
# smoteA = SMOTE(A[-1], 
#               A$bc_category,
#               K=5)
# #extract the synthetic data for the below group
# syn_below = smoteA$syn_data
# below=smoteA$orig_P #this is the origional minority group
# 
# #For the Exceed group
# B <- trainingSet[trainingSet$bc_category %in% c("Average", "Exceed"), ]
# B$bc_category <- factor(B$bc_category, levels = c("Average", "Exceed"))
# summary(B$bc_category) 
# 
# smoteB = SMOTE(B[-1], 
#                B$bc_category,
#                K=5)
# #extract the synthetic data for the below group
# syn_exceed = smoteB$syn_data
# exceed=smoteB$orig_P
# 
# #compare the synthetic data with origional data
# # syn_below$syn=1
# # below$syn=0  #not done yet, check it for important features later
# 
# #combine synthetic data to real trainset
# syn_below <- syn_below %>%
#   select(bc_category = class, everything())
# 
# syn_exceed <- syn_exceed %>%
#   select(bc_category = class, everything())
# 
# trainingSet <- rbind(trainingSet, syn_below)
# trainingSet <- rbind(trainingSet, syn_exceed)
# 
# # Shuffle the training set
# set.seed(1234)
# trainingSet <- trainingSet[sample(nrow(trainingSet)), ]

#############################################
#############################################
# Step 5 - training the random forest model #
#############################################
# Option 1 - Define the train control for cross-validation
ctrl_cv <- trainControl(
  method = "cv",  # k-fold cross-validation
  number = 10,    # 10-fold CV
  savePredictions = "final",
  classProbs = TRUE,
  sampling = "up",  # remove this if you have used SMOTE to oversample the minority groups
  summaryFunction = multiClassSummary  # multi-class summary function for comprehensive evaluation
)

# Option 2 - Set up the training control options for bagging (bootstrap aggregating)#
ctrl_boot <- trainControl(
  method = "boot",
  number = 100,
  savePredictions = "final",
  classProbs = T,
  sampling = "up"  #remove this if have used smote to oversampled the minority groups
)


# Model 1.0 - Train the random forest model with cross-validation and features selected using rfe 
set.seed(7777)
#tuneGrid <- expand.grid(mtry = seq(2, 22, by=2))
# rf_cv_rfe <- train(category ~ ., 
#                    data = trainingSet_rfe, 
#                    method = "rf", 
#                    trControl = ctrl_cv, 
#                    tuneGrid = tuneGrid,
#                    metric="Kappa")

# Define the parameter grid, including nodesize
tuneGrid <- expand.grid(mtry = seq(2, 23, by=2), nodesize = c(1, 5, 10, 15, 20))

# Custom random forest function to include nodesize
customRF <- list(type = "Classification", library = "randomForest", 
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "nodesize"), 
                                  class = c("numeric", "numeric"), 
                                  label = c("mtry", "nodesize"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {
  expand.grid(mtry = seq(2, 23, by=2), nodesize = c(1, 5, 10, 15, 20))
}

customRF$fit <- function(x, y, wts, param, lev, last, classProbs, ...) { 
  randomForest(x, y, mtry = param$mtry, nodesize = param$nodesize, ...)
}

customRF$predict <- function(modelFit, newdata, submodels = NULL) {
  predict(modelFit, newdata)
}

customRF$prob <- function(modelFit, newdata, submodels = NULL) {
  predict(modelFit, newdata, type = "prob")
}

customRF$sort <- function(x) x[order(x[,1]),]

customRF$levels <- function(x) x$classes

rf_cv_rfe <- train(category ~ ., 
                  data = trainingSet_rfe, 
                  method = customRF, 
                  trControl = ctrl_cv, 
                  tuneGrid = tuneGrid,
                  metric="Kappa")
#save the model
saveRDS(rf_cv_rfe, file="rf23_cv_rfe.rds")
print(rf_cv_rfe)
plot(rf_cv_rfe)
#variable importance
print(varImp(rf_cv_rfe))
plot(varImp(rf_cv_rfe))

# Model 1.1 - Train the random forest model with cross-validation and features selected using rfe-simple
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 13, by=1))
rf_cv_rfe_simp <- train(category ~ ., 
                   data = trainingSet_rfe_simp, 
                   method = "rf", 
                   trControl = ctrl_cv, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_cv_rfe_simp, file="rf23_cv_rfe_simp.rds")
print(rf_cv_rfe_simp)
plot(rf_cv_rfe_simp)
#variable importance
print(varImp(rf_cv_rfe_simp))
plot(varImp(rf_cv_rfe_simp))

# Model 2 - Train the random forest model with cross-validation and features selected using lasso 
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 29, by=2))
rf_cv_lasso <- train(category ~ ., 
                   data = trainingSet_lasso, 
                   method = "rf", 
                   trControl = ctrl_cv, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_cv_lasso, file="rf23_cv_lasso.rds")
print(rf_cv_lasso)
plot(rf_cv_lasso)
#variable importance
print(varImp(rf_cv_lasso))
plot(varImp(rf_cv_lasso))

# Model 2.1 - Train the random forest model with cross-validation and features selected using lasso-simple
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 23, by=2))
rf_cv_lasso_simp <- train(category ~ ., 
                     data = trainingSet_lasso_simp, 
                     method = "rf", 
                     trControl = ctrl_cv, 
                     tuneGrid = tuneGrid,
                     metric="Kappa")
#save the model
saveRDS(rf_cv_lasso_simp, file="rf23_cv_lasso_simp.rds")
print(rf_cv_lasso_simp)
plot(rf_cv_lasso_simp)
#variable importance
print(varImp(rf_cv_lasso_simp))
plot(varImp(rf_cv_lasso_simp))

# Model 2.2 - Train the random forest model with cross-validation and features selected using lasso-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 12, by=1))
rf_cv_lasso_select <- train(category ~ ., 
                          data = trainingSet_lasso_select, 
                          method = "rf", 
                          trControl = ctrl_cv, 
                          tuneGrid = tuneGrid,
                          metric="Kappa")
#save the model
saveRDS(rf_cv_lasso_select, file="rf23_cv_lasso_select.rds")
print(rf_cv_lasso_select)
plot(rf_cv_lasso_select)
#variable importance
print(varImp(rf_cv_lasso_select))
plot(varImp(rf_cv_lasso_select))

# Model 3 - Train the random forest model with cross-validation and features selected using Elastic Net
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 67, by=2))
rf_cv_en <- train(category ~ ., 
                   data = trainingSet_en, 
                   method = "rf", 
                   trControl = ctrl_cv, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_cv_en, file="rf23_cv_en.rds")
print(rf_cv_en)
plot(rf_cv_en)
#variable importance
print(varImp(rf_cv_en))
plot(varImp(rf_cv_en))

# Model 3.1 - Train the random forest model with cross-validation and features selected using Elastic Net-simple
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 46, by=2))
rf_cv_en_simp <- train(category ~ ., 
                  data = trainingSet_en_simp, 
                  method = "rf", 
                  trControl = ctrl_cv, 
                  tuneGrid = tuneGrid,
                  metric="Kappa")
#save the model
saveRDS(rf_cv_en_simp, file="rf23_cv_en_simp.rds")
print(rf_cv_en_simp)
plot(rf_cv_en_simp)
#variable importance
print(varImp(rf_cv_en_simp))
plot(varImp(rf_cv_en_simp))

# Model 3.2 - Train the random forest model with cross-validation and features selected using Elastic Net-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 21, by=2))
rf_cv_en_select <- train(category ~ ., 
                  data = trainingSet_en_select, 
                  method = "rf", 
                  trControl = ctrl_cv, 
                  tuneGrid = tuneGrid,
                  metric="Kappa")
#save the model
saveRDS(rf_cv_en_select, file="rf23_cv_en_select.rds")
print(rf_cv_en_select)
plot(rf_cv_en_select)
#variable importance
print(varImp(rf_cv_en_select))
plot(varImp(rf_cv_en_select))

# Model 4 - Train the random forest model with boot and features selected using rfe 
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 22, by=2))
rf_boot_rfe <- train(category ~ ., 
                   data = trainingSet_rfe, 
                   method = "rf", 
                   trControl = ctrl_boot, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_boot_rfe, file="rf23_boot_rfe.rds")
print(rf_boot_rfe)
plot(rf_boot_rfe)
#variable importance
print(varImp(rf_boot_rfe))
plot(varImp(rf_boot_rfe))

# Model 4.1 - Train the random forest model with boot and features selected using rfe-simple 
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(1, 13, by=1))
rf_boot_rfe_simp <- train(category ~ ., 
                     data = trainingSet_rfe_simp, 
                     method = "rf", 
                     trControl = ctrl_boot, 
                     tuneGrid = tuneGrid,
                     metric="Kappa")
#save the model
saveRDS(rf_boot_rfe_simp, file="rf23_boot_rfe_simp.rds")
print(rf_boot_rfe_simp)
plot(rf_boot_rfe_simp)
#variable importance
print(varImp(rf_boot_rfe_simp))
plot(varImp(rf_boot_rfe_simp))

# Model 5 - Train the random forest model with boot and features selected using lasso 
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 29, by=2))
rf_boot_lasso <- train(category ~ ., 
                   data = trainingSet_lasso, 
                   method = "rf", 
                   trControl = ctrl_boot, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_boot_lasso, file="rf23_boot_lasso.rds")
print(rf_boot_lasso)
plot(rf_boot_lasso)
#variable importance
print(varImp(rf_boot_lasso))
plot(varImp(rf_boot_lasso))

# Model 5.1 - Train the random forest model with boot and features selected using lasso-simple
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 23, by=2))
rf_boot_lasso_simp <- train(category ~ ., 
                          data = trainingSet_lasso_simp, 
                          method = "rf", 
                          trControl = ctrl_boot, 
                          tuneGrid = tuneGrid,
                          metric="Kappa")
#save the model
saveRDS(rf_boot_lasso_simp, file="rf23_boot_lasso_simp.rds")
print(rf_boot_lasso_simp)
plot(rf_boot_lasso_simp)
#variable importance
print(varImp(rf_boot_lasso_simp))
plot(varImp(rf_boot_lasso_simp))

# Model 5.2 - Train the random forest model with boot and features selected using lasso-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 12, by=1))
rf_boot_lasso_select <- train(category ~ ., 
                            data = trainingSet_lasso_select, 
                            method = "rf", 
                            trControl = ctrl_boot, 
                            tuneGrid = tuneGrid,
                            metric="Kappa")
#save the model
saveRDS(rf_boot_lasso_select, file="rf23_boot_lasso_select.rds")
print(rf_boot_lasso_select)
plot(rf_boot_lasso_select)
#variable importance
print(varImp(rf_boot_lasso_select))
plot(varImp(rf_boot_lasso_select))

# Model 6 - Train the random forest model with boot and features selected using Elastic Net
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 67, by=2))
rf_boot_en <- train(category ~ ., 
                   data = trainingSet_en, 
                   method = "rf", 
                   trControl = ctrl_boot, 
                   tuneGrid = tuneGrid,
                   metric="Kappa")
#save the model
saveRDS(rf_boot_en, file="rf23_boot_en.rds")
print(rf_boot_en)
plot(rf_boot_en)
#variable importance
print(varImp(rf_boot_en))
plot(varImp(rf_boot_en))

# Model 6.1 - Train the random forest model with boot and features selected using Elastic Net-simple
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 46, by=2))
rf_boot_en_simp <- train(category ~ ., 
                       data = trainingSet_en_simp, 
                       method = "rf", 
                       trControl = ctrl_boot, 
                       tuneGrid = tuneGrid,
                       metric="Kappa")
#save the model
saveRDS(rf_boot_en_simp, file="rf23_boot_en_simp.rds")
print(rf_boot_en_simp)
plot(rf_boot_en_simp)
#variable importance
print(varImp(rf_boot_en_simp))
plot(varImp(rf_boot_en_simp))

# Model 6.2 - Train the random forest model with boot and features selected using Elastic Net-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 21, by=2))
rf_boot_en_select <- train(category ~ ., 
                         data = trainingSet_en_select, 
                         method = "rf", 
                         trControl = ctrl_boot, 
                         tuneGrid = tuneGrid,
                         metric="Kappa")
#save the model
saveRDS(rf_boot_en_select, file="rf23_boot_en_select.rds")
print(rf_boot_en_select)
plot(rf_boot_en_select)
#variable importance
print(varImp(rf_boot_en_select))
plot(varImp(rf_boot_en_select))

# Compare models
results_cv <- resamples(list(RFE_CV = rf_cv_rfe, RFE_CV_simp = rf_cv_rfe_simp, LASSO_CV = rf_cv_lasso, LASSO_CV_simp = rf_cv_lasso_simp, LASSO_CV_select = rf_cv_lasso_select, ElasticNet_CV = rf_cv_en, ElasticNet_CV_simp = rf_cv_en_simp, ElasticNet_CV_select = rf_cv_en_select))
summary(results_cv)
results_boot <- resamples(list(RFE_boot = rf_boot_rfe, RFE_boot_simp = rf_boot_rfe_simp, LASSO_boot = rf_boot_lasso, LASSO_boot_simp = rf_boot_lasso_simp, LASSO_boot_select = rf_boot_lasso_select, ElasticNet_boot = rf_boot_en, ElasticNet_boot_simp = rf_boot_en_simp, ElasticNet_boot_select = rf_boot_en_select))
summary(results_boot)
#model rf_cv_en has the best performance for the training set

##################################################
# Optional-Manually removed some of the freatures #
##################################################
#manually select features based on domain knowledge, and the importance score generate by randomforest, and the correlation between features
#based on features_RFE (22 features)
#extract the features selected using RFE
rfProfile <- readRDS("features_RFE.rds")
features_rfe <- rfProfile$optVariables
# Extract the relevant columns
selected_data <- data23_1619[, features_rfe]
# Calculate the correlation matrix
correlation_matrix <- cor(selected_data, use = "complete.obs")
# Convert the correlation matrix into a data frame
cor_df <- as.data.frame(as.table(correlation_matrix))
# Round the correlation coefficients to 2 decimal places
cor_df$Freq <- round(cor_df$Freq, 2)
# Filter the pairs with correlation > 0.70 or < -0.70, excluding self-correlations (diagonal elements)
filtered_cor <- cor_df %>%
  filter(Freq > 0.70 | Freq < -0.70) %>%
  filter(Var1 != Var2)
# Print the filtered pairs
print(filtered_cor)
# Plot the heatmap
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#select features(correlation>0.70)
print(varImp(rf_cv_rfe))
plot(varImp(rf_cv_rfe))
print(features_rfe)
features_rfe_select <- features_rfe[-c(7,19,22)] #remove variables with corr>0.7 and less importance score based on re_cv_rfe
selected_data <- data23_1619[, features_rfe_select]
correlation_matrix <- cor(selected_data, use = "complete.obs")
# Convert the correlation matrix into a data frame
cor_df <- as.data.frame(as.table(correlation_matrix))
# Round the correlation coefficients to 2 decimal places
cor_df$Freq <- round(cor_df$Freq, 2)
# Filter the pairs with correlation > 0.70 or < -0.70, excluding self-correlations (diagonal elements)
filtered_cor <- cor_df %>%
  filter(Freq > 0.70 | Freq < -0.70) %>%
  filter(Var1 != Var2)
# Print the filtered pairs
print(filtered_cor) #no correlation >0.70
data23_1619_rfe_select <- data23_1619[, c("category", features_rfe_select)]

#based on features_RFE_simp (13 features)
#extract the features selected using RFE
rfProfile <- readRDS("features_RFE_simp.rds")
features_rfe_simp <- rfProfile$optVariables
# Extract the relevant columns
selected_data <- data23_1619[, features_rfe_simp]
# Calculate the correlation matrix
correlation_matrix <- cor(selected_data, use = "complete.obs")
# Convert the correlation matrix into a data frame
cor_df <- as.data.frame(as.table(correlation_matrix))
# Round the correlation coefficients to 2 decimal places
cor_df$Freq <- round(cor_df$Freq, 2)
# Filter the pairs with correlation > 0.70 or < -0.70, excluding self-correlations (diagonal elements)
filtered_cor <- cor_df %>%
  filter(Freq > 0.70 | Freq < -0.70) %>%
  filter(Var1 != Var2)
# Print the filtered pairs
print(filtered_cor)
# Plot the heatmap
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#select features(correlation>0.70)
print(varImp(rf_cv_rfe_simp))
plot(varImp(rf_cv_rfe_simp))
print(features_rfe_simp)
features_rfe_simp_select <- features_rfe_simp[-c(10, 3, 9)] #remove variables with corr>0.7 and less importance score based on re_cv_rfe
selected_data <- data23_1619[, features_rfe_simp_select]
correlation_matrix <- cor(selected_data, use = "complete.obs")
# Convert the correlation matrix into a data frame
cor_df <- as.data.frame(as.table(correlation_matrix))
# Round the correlation coefficients to 2 decimal places
cor_df$Freq <- round(cor_df$Freq, 2)
# Filter the pairs with correlation > 0.70 or < -0.70, excluding self-correlations (diagonal elements)
filtered_cor <- cor_df %>%
  filter(Freq > 0.70 | Freq < -0.70) %>%
  filter(Var1 != Var2)
# Print the filtered pairs
print(filtered_cor) #no correlation >0.70
data23_1619_rfe_simp_select <- data23_1619[, c("category", features_rfe_simp_select)]

# Model 1.2 - Train the random forest model with cross-validation and features selected using rfe-select
trainingSet_rfe_select <- data23_1619_rfe_select[trainIndex,]
trainingSet_rfe_simp_select <- data23_1619_rfe_simp_select[trainIndex,] #this one contains less features by forcing the number of features less than 20
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 19, by=2))
rf_cv_rfe_select <- train(category ~ ., 
                          data = trainingSet_rfe_select, 
                          method = "rf", 
                          trControl = ctrl_cv, 
                          tuneGrid = tuneGrid,
                          metric="Kappa")
#save the model
saveRDS(rf_cv_rfe_select, file="rf23_cv_rfe_select.rds")
print(rf_cv_rfe_select)
plot(rf_cv_rfe_select)
#variable importance
print(varImp(rf_cv_rfe_select))
plot(varImp(rf_cv_rfe_select))

# Model 1.3 - Train the random forest model with cross-validation and features selected using rfe-simple-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 10, by=1))
rf_cv_rfe_simp_select<- train(category ~ ., 
                              data = trainingSet_rfe_simp_select, 
                              method = "rf", 
                              trControl = ctrl_cv, 
                              tuneGrid = tuneGrid,
                              metric="Kappa")
#save the model
saveRDS(rf_cv_rfe_simp_select, file="rf23_cv_rfe_simp_select.rds")
print(rf_cv_rfe_simp_select)
plot(rf_cv_rfe_simp_select)
#variable importance
print(varImp(rf_cv_rfe_simp_select))
plot(varImp(rf_cv_rfe_simp_select))

# Model 4.2 - Train the random forest model with boot and features selected using rfe-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 19, by=2))
rf_boot_rfe_select <- train(category ~ ., 
                            data = trainingSet_rfe_select, 
                            method = "rf", 
                            trControl = ctrl_boot, 
                            tuneGrid = tuneGrid,
                            metric="Kappa")
#save the model
saveRDS(rf_boot_rfe_select, file="rf23_boot_rfe_select.rds")
print(rf_boot_rfe_select)
plot(rf_boot_rfe_select)
#variable importance
print(varImp(rf_boot_rfe_select))
plot(varImp(rf_boot_rfe_select))

# Model 4.3 - Train the random forest model with boot and features selected using rfe-simple-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 10, by=1))
rf_boot_rfe_simp_select<- train(category ~ ., 
                                data = trainingSet_rfe_simp_select, 
                                method = "rf", 
                                trControl = ctrl_boot, 
                                tuneGrid = tuneGrid,
                                metric="Kappa")
#save the model
saveRDS(rf_boot_rfe_simp_select, file="rf23_boot_rfe_simp_select.rds")
print(rf_boot_rfe_simp_select)
plot(rf_boot_rfe_simp_select)
#variable importance
print(varImp(rf_boot_rfe_simp_select))
plot(varImp(rf_boot_rfe_simp_select))

##################################################
# Step 6 - check model performance using TestSet #
##################################################
fitted_cv_rfe <- predict(rf_cv_rfe, testSet)
fitted_cv_rfe_select <- predict(rf_cv_rfe_select, testSet)
fitted_cv_rfe_simp <- predict(rf_cv_rfe_simp, testSet)
fitted_cv_rfe_simp_select <- predict(rf_cv_rfe_simp_select, testSet)
fitted_cv_lasso <- predict(rf_cv_lasso, testSet)
fitted_cv_lasso_simp <- predict(rf_cv_lasso_simp, testSet)
fitted_cv_lasso_select <- predict(rf_cv_lasso_select, testSet)
fitted_cv_en <- predict(rf_cv_en, testSet)
fitted_cv_en_simp <- predict(rf_cv_en_simp, testSet)
fitted_cv_en_select <- predict(rf_cv_en_select, testSet)

fitted_boot_rfe <- predict(rf_boot_rfe, testSet)
fitted_boot_rfe_select <- predict(rf_boot_rfe_select, testSet)
fitted_boot_rfe_simp <- predict(rf_boot_rfe_simp, testSet)
fitted_boot_rfe_simp_select <- predict(rf_boot_rfe_simp_select, testSet)
fitted_boot_lasso <- predict(rf_boot_lasso, testSet)
fitted_boot_lasso_simp <- predict(rf_boot_lasso_simp, testSet)
fitted_boot_lasso_select <- predict(rf_boot_lasso_select, testSet)
fitted_boot_en <- predict(rf_boot_en, testSet)
fitted_boot_en_simp <- predict(rf_boot_en_simp, testSet)
fitted_boot_en_select <- predict(rf_boot_en_select, testSet)

#confusion matrix comparing fitted values with actual values
cm_cv_rfe <- confusionMatrix(reference=testSet$category, data=fitted_cv_rfe, mode="everything", positive = "Exceed")
cm_cv_rfe_select <- confusionMatrix(reference=testSet$category, data=fitted_cv_rfe_select, mode="everything", positive = "Exceed")
cm_cv_rfe_simp <- confusionMatrix(reference=testSet$category, data=fitted_cv_rfe_simp, mode="everything", positive = "Exceed")
cm_cv_rfe_simp_select <- confusionMatrix(reference=testSet$category, data=fitted_cv_rfe_simp_select, mode="everything", positive = "Exceed")
cm_cv_lasso <- confusionMatrix(reference=testSet$category, data=fitted_cv_lasso, mode="everything", positive = "Exceed")
cm_cv_lasso_simp <- confusionMatrix(reference=testSet$category, data=fitted_cv_lasso_simp, mode="everything", positive = "Exceed")
cm_cv_lasso_select <- confusionMatrix(reference=testSet$category, data=fitted_cv_lasso_select, mode="everything", positive = "Exceed")
cm_cv_en <- confusionMatrix(reference=testSet$category, data=fitted_cv_en, mode="everything", positive = "Exceed")
cm_cv_en_simp <- confusionMatrix(reference=testSet$category, data=fitted_cv_en_simp, mode="everything", positive = "Exceed")
cm_cv_en_select <- confusionMatrix(reference=testSet$category, data=fitted_cv_en_select, mode="everything", positive = "Exceed")

cm_boot_rfe <- confusionMatrix(reference=testSet$category, data=fitted_boot_rfe, mode="everything", positive = "Exceed")
cm_boot_rfe_select <- confusionMatrix(reference=testSet$category, data=fitted_boot_rfe_select, mode="everything", positive = "Exceed")
cm_boot_rfe_simp <- confusionMatrix(reference=testSet$category, data=fitted_boot_rfe_simp, mode="everything", positive = "Exceed")
cm_boot_rfe_simp_select <- confusionMatrix(reference=testSet$category, data=fitted_boot_rfe_simp_select, mode="everything", positive = "Exceed")
cm_boot_lasso <- confusionMatrix(reference=testSet$category, data=fitted_boot_lasso, mode="everything", positive = "Exceed")
cm_boot_lasso_simp <- confusionMatrix(reference=testSet$category, data=fitted_boot_lasso_simp, mode="everything", positive = "Exceed")
cm_boot_lasso_select <- confusionMatrix(reference=testSet$category, data=fitted_boot_lasso_select, mode="everything", positive = "Exceed")
cm_boot_en <- confusionMatrix(reference=testSet$category, data=fitted_boot_en, mode="everything", positive = "Exceed")
cm_boot_en_simp <- confusionMatrix(reference=testSet$category, data=fitted_boot_en_simp, mode="everything", positive = "Exceed")
cm_boot_en_select <- confusionMatrix(reference=testSet$category, data=fitted_boot_en_select, mode="everything", positive = "Exceed")

#compare the confusion matrix
# Function to extract key metrics from a confusion matrix
extract_metrics <- function(cm) {
  metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Sensitivity_Average = cm$byClass["Class: Average", "Sensitivity"],
    Specificity_Average = cm$byClass["Class: Average", "Specificity"],
    F1_Average = cm$byClass["Class: Average", "F1"],
    Sensitivity_Below = cm$byClass["Class: Below", "Sensitivity"],
    Specificity_Below = cm$byClass["Class: Below", "Specificity"],
    F1_Below = cm$byClass["Class: Below", "F1"],
    Sensitivity_Exceed = cm$byClass["Class: Exceed", "Sensitivity"],
    Specificity_Exceed = cm$byClass["Class: Exceed", "Specificity"],
    F1_Exceed = cm$byClass["Class: Exceed", "F1"]
  )
  return(metrics)
}

# Extract metrics from each confusion matrix
metrics_cv_rfe <- extract_metrics(cm_cv_rfe)
metrics_cv_rfe_select <- extract_metrics(cm_cv_rfe_select)
metrics_cv_rfe_simp <- extract_metrics(cm_cv_rfe_simp)
metrics_cv_rfe_simp_select <- extract_metrics(cm_cv_rfe_simp_select)
metrics_cv_lasso <- extract_metrics(cm_cv_lasso)
metrics_cv_lasso_simp <- extract_metrics(cm_cv_lasso_simp)
metrics_cv_lasso_select <- extract_metrics(cm_cv_lasso_select)
metrics_cv_en <- extract_metrics(cm_cv_en)
metrics_cv_en_simp <- extract_metrics(cm_cv_en_simp)
metrics_cv_en_select <- extract_metrics(cm_cv_en_select)
metrics_boot_rfe <- extract_metrics(cm_boot_rfe)
metrics_boot_rfe_select <- extract_metrics(cm_boot_rfe_select)
metrics_boot_rfe_simp <- extract_metrics(cm_boot_rfe_simp)
metrics_boot_rfe_simp_select <- extract_metrics(cm_boot_rfe_simp_select)
metrics_boot_lasso <- extract_metrics(cm_boot_lasso)
metrics_boot_lasso_simp <- extract_metrics(cm_boot_lasso_simp)
metrics_boot_lasso_select <- extract_metrics(cm_boot_lasso_select)
metrics_boot_en <- extract_metrics(cm_boot_en)
metrics_boot_en_simp <- extract_metrics(cm_boot_en_simp)
metrics_boot_en_select <- extract_metrics(cm_boot_en_select)

# Combine the metrics into a single dataframe for comparison
comparison_df <- rbind(
  cv_rfe = metrics_cv_rfe,
  cv_rfe_select = metrics_cv_rfe_select,
  cv_rfe_simp = metrics_cv_rfe_simp,
  cv_rfe_simp_select = metrics_cv_rfe_simp_select,
  cv_lasso = metrics_cv_lasso,
  cv_lasso_simp = metrics_cv_lasso_simp,
  cv_lasso_select = metrics_cv_lasso_select,
  cv_en = metrics_cv_en,
  cv_en_simp = metrics_cv_en_simp,
  cv_en_select = metrics_cv_en_select,
  boot_rfe = metrics_boot_rfe,
  boot_rfe_select = metrics_boot_rfe_select,
  boot_rfe_simp = metrics_boot_rfe_simp,
  boot_rfe_simp_select = metrics_boot_rfe_simp_select,
  boot_lasso = metrics_boot_lasso,
  boot_lasso_simp = metrics_boot_lasso_simp,
  boot_lasso_select = metrics_boot_lasso_select,
  boot_en = metrics_boot_en,
  boot_en_simp = metrics_boot_en_simp,
  boot_en_select = metrics_boot_en_select
)

# Print the comparison dataframe
print(comparison_df)

###########################################################################
# Additional: remove features with no association with the continuous SIR #
###########################################################################

#the R2 for each of the important features.
dm <- lm(log(sir) ~ dm_p_asr, data = data23_1619full)
summary(dm)

insur <- lm(log(sir) ~ private_insurance_rate_per_1000, data = data23_1619full)
summary(insur)

fruit <- lm(log(sir) ~ fruit_p_asr, data = data23_1619full)
summary(fruit)

bach <- lm(log(sir) ~ Bachelor_degree, data = data23_1619full)
summary(bach)

employ<- lm(log(sir) ~ Unemployeed_rate, data = data23_1619full)
summary(employ)

seperate <- lm(log(sir) ~ seperated_percentage, data = data23_1619full)
summary(seperate)

super <- lm(log(sir) ~ Median_Super_income_per_year, data = data23_1619full)
summary(super)

yr11 <- lm(log(sir) ~ Completed_year_11, data = data23_1619full)
summary(yr11)

buss <- lm(log(sir) ~ Median_own_unincorporated_business_income_per_year, data = data23_1619full)
summary(buss)

smk <- lm(log(sir) ~ smk_p_asr, data = data23_1619full)
summary(smk)

irsd <- lm(log(sir) ~ irsd_state_rank, data = data23_1619full)
summary(irsd)

irsad <- lm(log(sir) ~ irsad_state_rank, data = data23_1619full)
summary(irsad)

males1014 <- lm(log2(sir) ~ males_10_14, data = data23_1619full)
summary(males1014)

cor(data23_1619full$smk_p_asr, data23_1619full$dm_p_asr,
    method = "pearson")

#further select features those are not related to the continuous sir
#FOR RFE_SIMP_SELECT
print(features_rfe_simp_select)
final_select <- features_rfe_simp_select[-c(6, 9)] #remove variables not associated with continuous SIR
data23_1619_final_select <- data23_1619[, c("category", final_select)]

# Model 1.4 - Train the random forest model with cross-validation and features selected using rfe-select
trainingSet_final_select <- data23_1619_final_select[trainIndex,]
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 9, by=1))
rf_cv_final_select <- train(category ~ ., 
                            data = trainingSet_final_select, 
                            method = "rf", 
                            trControl = ctrl_cv, 
                            tuneGrid = tuneGrid,
                            metric="Kappa")
#save the model
saveRDS(rf_cv_final_select, file="rf23_cv_final_select.rds")
print(rf_cv_final_select)
plot(rf_cv_final_select)
#variable importance
print(varImp(rf_cv_final_select))
plot(varImp(rf_cv_final_select))

# Model 4.4 - Train the random forest model with boot and features selected using rfe-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 9, by=1))
rf_boot_final_select <- train(category ~ ., 
                              data = trainingSet_final_select, 
                              method = "rf", 
                              trControl = ctrl_boot, 
                              tuneGrid = tuneGrid,
                              metric="Kappa")
#save the model
saveRDS(rf_boot_final_select, file="rf23_boot_final_select.rds")
print(rf_boot_final_select)
plot(rf_boot_final_select)
#variable importance
print(varImp(rf_boot_final_select))
plot(varImp(rf_boot_final_select))

#FOR RFE_SELECT
print(features_rfe_select)
final_select2 <- features_rfe_select[-c(4,10,12)] #remove variables not associated with continuous SIR
data23_1619_final_select2 <- data23_1619[, c("category", final_select2)]

# Model 1.5 - Train the random forest model with cross-validation and features selected using rfe-select
trainingSet_final_select2 <- data23_1619_final_select2[trainIndex,]
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 17, by=2))
rf_cv_final_select2 <- train(category ~ ., 
                             data = trainingSet_final_select2, 
                             method = "rf", 
                             trControl = ctrl_cv, 
                             tuneGrid = tuneGrid,
                             metric="Kappa")
#save the model
saveRDS(rf_cv_final_select2, file="rf23_cv_final_select2.rds")
print(rf_cv_final_select2)
plot(rf_cv_final_select2)
#variable importance
print(varImp(rf_cv_final_select2))
plot(varImp(rf_cv_final_select2))

# Model 4.4 - Train the random forest model with boot and features selected using rfe-select
set.seed(7777)
tuneGrid <- expand.grid(mtry = seq(2, 17, by=2))
rf_boot_final_select2 <- train(category ~ ., 
                               data = trainingSet_final_select2, 
                               method = "rf", 
                               trControl = ctrl_boot, 
                               tuneGrid = tuneGrid,
                               metric="Kappa")
#save the model
saveRDS(rf_boot_final_select2, file="rf23_boot_final_select2.rds")
print(rf_boot_final_select2)
plot(rf_boot_final_select2)
#variable importance
print(varImp(rf_boot_final_select2))
plot(varImp(rf_boot_final_select2))

#check model performance using testset
fitted_cv_final_select <- predict(rf_cv_final_select, testSet)
confusionMatrix(reference=testSet$category, data=fitted_cv_final_select, mode="everything", positive = "Exceed")

fitted_boot_final_select <- predict(rf_boot_final_select, testSet)
confusionMatrix(reference=testSet$category, data=fitted_boot_final_select, mode="everything", positive = "Exceed")

fitted_cv_final_select2 <- predict(rf_cv_final_select2, testSet)
confusionMatrix(reference=testSet$category, data=fitted_cv_final_select2, mode="everything", positive = "Exceed")

fitted_boot_final_select2 <- predict(rf_boot_final_select2, testSet)
confusionMatrix(reference=testSet$category, data=fitted_boot_final_select2, mode="everything", positive = "Exceed")

############################################################
# Step 7 - check model performance using the whole dataset #
############################################################
#############################################################
fitted_cv_rfe <- predict(rf_cv_rfe, data23_1619)
fitted_cv_rfe_select <- predict(rf_cv_rfe_select, data23_1619)
fitted_cv_rfe_simp <- predict(rf_cv_rfe_simp, data23_1619)
fitted_cv_rfe_simp_select <- predict(rf_cv_rfe_simp_select, data23_1619)
fitted_cv_lasso <- predict(rf_cv_lasso, data23_1619)
fitted_cv_en <- predict(rf_cv_en, data23_1619)

fitted_boot_rfe <- predict(rf_boot_rfe, data23_1619)
fitted_boot_rfe_select <- predict(rf_boot_rfe_select, data23_1619)
fitted_boot_rfe_simp <- predict(rf_boot_rfe_simp, data23_1619)
fitted_boot_rfe_simp_select <- predict(rf_boot_rfe_simp_select, data23_1619)
fitted_boot_lasso <- predict(rf_boot_lasso, data23_1619)
fitted_boot_en <- predict(rf_boot_en, data23_1619)

#confusion matrix comparing fitted values with actual values
cm_cv_rfe <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_rfe, mode="everything", positive = "Exceed")
cm_cv_rfe_select <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_rfe_select, mode="everything", positive = "Exceed")
cm_cv_rfe_simp <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_rfe_simp, mode="everything", positive = "Exceed")
cm_cv_rfe_simp_select <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_rfe_simp_select, mode="everything", positive = "Exceed")
cm_cv_lasso <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_lasso, mode="everything", positive = "Exceed")
cm_cv_en <- confusionMatrix(reference=data23_1619$category, data=fitted_cv_en, mode="everything", positive = "Exceed")

cm_boot_rfe <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_rfe, mode="everything", positive = "Exceed")
cm_boot_rfe_select <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_rfe_select, mode="everything", positive = "Exceed")
cm_boot_rfe_simp <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_rfe_simp, mode="everything", positive = "Exceed")
cm_boot_rfe_simp_select <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_rfe_simp_select, mode="everything", positive = "Exceed")
cm_boot_lasso <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_lasso, mode="everything", positive = "Exceed")
cm_boot_en <- confusionMatrix(reference=data23_1619$category, data=fitted_boot_en, mode="everything", positive = "Exceed")

#compare the confusion matrix
# Function to extract key metrics from a confusion matrix
extract_metrics <- function(cm) {
  metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Sensitivity_Average = cm$byClass["Class: Average", "Sensitivity"],
    Specificity_Average = cm$byClass["Class: Average", "Specificity"],
    F1_Average = cm$byClass["Class: Average", "F1"],
    Sensitivity_Below = cm$byClass["Class: Below", "Sensitivity"],
    Specificity_Below = cm$byClass["Class: Below", "Specificity"],
    F1_Below = cm$byClass["Class: Below", "F1"],
    Sensitivity_Exceed = cm$byClass["Class: Exceed", "Sensitivity"],
    Specificity_Exceed = cm$byClass["Class: Exceed", "Specificity"],
    F1_Exceed = cm$byClass["Class: Exceed", "F1"]
  )
  return(metrics)
}

# Extract metrics from each confusion matrix
metrics_cv_rfe <- extract_metrics(cm_cv_rfe)
metrics_cv_rfe_select <- extract_metrics(cm_cv_rfe_select)
metrics_cv_rfe_simp <- extract_metrics(cm_cv_rfe_simp)
metrics_cv_rfe_simp_select <- extract_metrics(cm_cv_rfe_simp_select)
metrics_cv_lasso <- extract_metrics(cm_cv_lasso)
metrics_cv_en <- extract_metrics(cm_cv_en)
metrics_boot_rfe <- extract_metrics(cm_boot_rfe)
metrics_boot_rfe_select <- extract_metrics(cm_boot_rfe_select)
metrics_boot_rfe_simp <- extract_metrics(cm_boot_rfe_simp)
metrics_boot_rfe_simp_select <- extract_metrics(cm_boot_rfe_simp_select)
metrics_boot_lasso <- extract_metrics(cm_boot_lasso)
metrics_boot_en <- extract_metrics(cm_boot_en)

# Combine the metrics into a single dataframe for comparison
comparison_df <- rbind(
  cv_rfe = metrics_cv_rfe,
  cv_rfe_select = metrics_cv_rfe_select,
  cv_rfe_simp = metrics_cv_rfe_simp,
  cv_rfe_simp_select = metrics_cv_rfe_simp_select,
  cv_lasso = metrics_cv_lasso,
  cv_en = metrics_cv_en,
  boot_rfe = metrics_boot_rfe,
  boot_rfe_select = metrics_boot_rfe_select,
  boot_rfe_simp = metrics_boot_rfe_simp,
  boot_rfe_simp_select = metrics_boot_rfe_simp_select,
  boot_lasso = metrics_boot_lasso,
  boot_en = metrics_boot_en
)

# Print the comparison dataframe
print(comparison_df)

###########################################################
# Step 8 - check model performance using previous cohort #
###########################################################
#prepare for the 2012-2015 data
data23_1215=read.csv("data23_1215.csv")
data23_1215 = subset(data23_1215, select=-X)
# use the bagging imputation method for filling in missing data
set.seed(6666)
bagMissing <- preProcess(data23_1215, method="bagImpute")
data23_1215 <- predict(bagMissing, newdata = data23_1215)
remove(bagMissing)

# convert to a z scale for all the predictors
data23_1215X <- subset(data23_1215, select=-category)
zModel <- preProcess(data23_1215X, Method = c("center", "scale"))
data23_1215X <- predict(zModel, newdata = data23_1215X)
#add response variable to the dataset
data23_1215<- cbind(data23_1215$category, data23_1215X)
#make sure the variable names are correct
names(data23_1215)[names(data23_1215)=="data23_1215$category"] = "category"
data23_1215$category=as.factor(data23_1215$category)

#use the 2012-2015 data to check the model performance
#fitted_cv_rfe <- predict(rf_cv_rfe, data23_1215)  #mhp_f_asr not exist in this data period#
#fitted_cv_rfe_select <- predict(rf_cv_rfe_select, data23_1215)  #mhp_f_asr not exist in this data period#
fitted_cv_rfe_simp <- predict(rf_cv_rfe_simp, data23_1215)
fitted_cv_rfe_simp_select <- predict(rf_cv_rfe_simp_select, data23_1215)
#fitted_cv_lasso <- predict(rf_cv_lasso, data23_1215)
#fitted_cv_lasso_simp <- predict(rf_cv_lasso_simp, data23_1215) #ieo_state_rank_f_asr not exist in this data period#
#fitted_cv_lasso_select <- predict(rf_cv_lasso_select, data23_1215)  #ieo_state_rank_f_asr not exist in this data period#
#fitted_cv_en <- predict(rf_cv_en, data23_1215)  #mhp_f_asr not exist in this data period#
#fitted_cv_en_simp <- predict(rf_cv_en_simp, data23_1215)  #mhp_f_asr not exist in this data period#
#fitted_cv_en_select <- predict(rf_cv_en_select, data23_1215) #mhp_f_asr not exist in this data period#

#fitted_boot_rfe <- predict(rf_boot_rfe, data23_1215)
#fitted_boot_rfe_select <- predict(rf_boot_rfe_select, data23_1215)
fitted_boot_rfe_simp <- predict(rf_boot_rfe_simp, data23_1215)
fitted_boot_rfe_simp_select <- predict(rf_boot_rfe_simp_select, data23_1215)
# fitted_boot_lasso <- predict(rf_boot_lasso, data23_1215)
# fitted_boot_lasso_simp <- predict(rf_boot_lasso_simp, data23_1215)
# fitted_boot_lasso_select <- predict(rf_boot_lasso_select, data23_1215)
# fitted_boot_en <- predict(rf_boot_en, data23_1215)
# fitted_boot_en_simp <- predict(rf_boot_en_simp, data23_1215)
# fitted_boot_en_select <- predict(rf_boot_en_select, data23_1215)

#confusion matrix comparing fitted values with actual values
# cm_cv_rfe <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_rfe, mode="everything", positive = "Exceed")
# cm_cv_rfe_select <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_rfe_select, mode="everything", positive = "Exceed")
cm_cv_rfe_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_rfe_simp, mode="everything", positive = "Exceed")
cm_cv_rfe_simp_select <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_rfe_simp_select, mode="everything", positive = "Exceed")
# cm_cv_lasso <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_lasso, mode="everything", positive = "Exceed")
# cm_cv_lasso_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_lasso_simp, mode="everything", positive = "Exceed")
# cm_cv_lasso_select <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_lasso_select, mode="everything", positive = "Exceed")
# cm_cv_en <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_en, mode="everything", positive = "Exceed")
# cm_cv_en_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_en_simp, mode="everything", positive = "Exceed")
# cm_cv_en_select <- confusionMatrix(reference=data23_1215$category, data=fitted_cv_en_select, mode="everything", positive = "Exceed")

# cm_boot_rfe <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_rfe, mode="everything", positive = "Exceed")
# cm_boot_rfe_select <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_rfe_select, mode="everything", positive = "Exceed")
cm_boot_rfe_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_rfe_simp, mode="everything", positive = "Exceed")
cm_boot_rfe_simp_select <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_rfe_simp_select, mode="everything", positive = "Exceed")
# cm_boot_lasso <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_lasso, mode="everything", positive = "Exceed")
# cm_boot_lasso_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_lasso_simp, mode="everything", positive = "Exceed")
# cm_boot_lasso_select <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_lasso_select, mode="everything", positive = "Exceed")
# cm_boot_en <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_en, mode="everything", positive = "Exceed")
# cm_boot_en_simp <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_en_simp, mode="everything", positive = "Exceed")
# cm_boot_en_select <- confusionMatrix(reference=data23_1215$category, data=fitted_boot_en_select, mode="everything", positive = "Exceed")

# Extract metrics from each confusion matrix
# metrics_cv_rfe <- extract_metrics(cm_cv_rfe)
# metrics_cv_rfe_select <- extract_metrics(cm_cv_rfe_select)
metrics_cv_rfe_simp <- extract_metrics(cm_cv_rfe_simp)
metrics_cv_rfe_simp_select <- extract_metrics(cm_cv_rfe_simp_select)
# metrics_cv_lasso <- extract_metrics(cm_cv_lasso)
# metrics_cv_lasso_simp <- extract_metrics(cm_cv_lasso_simp)
# metrics_cv_lasso_select <- extract_metrics(cm_cv_lasso_select)
# metrics_cv_en <- extract_metrics(cm_cv_en)
# metrics_cv_en_simp <- extract_metrics(cm_cv_en_simp)
# metrics_cv_en_select <- extract_metrics(cm_cv_en_select)
# metrics_boot_rfe <- extract_metrics(cm_boot_rfe)
# metrics_boot_rfe_select <- extract_metrics(cm_boot_rfe_select)
metrics_boot_rfe_simp <- extract_metrics(cm_boot_rfe_simp)
metrics_boot_rfe_simp_select <- extract_metrics(cm_boot_rfe_simp_select)
# metrics_boot_lasso <- extract_metrics(cm_boot_lasso)
# metrics_boot_lasso_simp <- extract_metrics(cm_boot_lasso_simp)
# metrics_boot_lasso_select <- extract_metrics(cm_boot_lasso_select)
# metrics_boot_en <- extract_metrics(cm_boot_en)
# metrics_boot_en_simp <- extract_metrics(cm_boot_en_simp)
# metrics_boot_en_select <- extract_metrics(cm_boot_en_select)

# Combine the metrics into a single dataframe for comparison
comparison_df <- rbind(
  # cv_rfe = metrics_cv_rfe,
  # cv_rfe_select = metrics_cv_rfe_select,
  cv_rfe_simp = metrics_cv_rfe_simp,
  cv_rfe_simp_select = metrics_cv_rfe_simp_select,
  # cv_lasso = metrics_cv_lasso,
  # cv_lasso_simp = metrics_cv_lasso_simp,
  # cv_lasso_select = metrics_cv_lasso_select,
  # cv_en = metrics_cv_en,
  # cv_en_simp = metrics_cv_en_simp,
  # cv_en_select = metrics_cv_en_select,
  # boot_rfe = metrics_boot_rfe,
  # boot_rfe_select = metrics_boot_rfe_select,
  boot_rfe_simp = metrics_boot_rfe_simp,
  boot_rfe_simp_select = metrics_boot_rfe_simp_select
  # boot_lasso = metrics_boot_lasso,
  # boot_lasso_simp = metrics_boot_lasso_simp,
  # boot_lasso_select = metrics_boot_lasso_select,
  # boot_en = metrics_boot_en,
  # boot_en_simp = metrics_boot_en_simp,
  # boot_en_select = metrics_boot_en_select
)

# Print the comparison dataframe
print(comparison_df)
#######################################
# Step 9 - Extract feature importance #
#######################################
rf_model=rf_boot_final_select$finalModel
print(varImp(rf_model))
plot(varImp(rf_model))

var=varImp(rf_model)
data <- data.frame(matrix(nrow=8))
data$features <- rf_model$xNames
data$impscore <- rf_model$importance
data<-subset(data, select=c(features, impscore))

#calculate the avarage values of features by category
features <- rf_model$xNames
data23_1619_sub <- data23_1619[, c("category", features)]
avg_values <- aggregate(. ~ category, data = data23_1619_sub, median)
datanew <- as.data.frame(t(avg_values))
names(datanew)[names(datanew)=="V1"] = "Average"
names(datanew)[names(datanew)=="V2"] = "Below"
names(datanew)[names(datanew)=="V3"] = "Exceed"
datanew <- datanew[-1,]
datanew$features <- rownames(datanew)
rownames(datanew) <- seq_len(nrow(datanew))

#merge average values to feature importance scores
weight<- merge(data, datanew, by="features")
weight <- weight[,c("features", "impscore", "Below", "Average", "Exceed")]
weight <- arrange(weight, desc(impscore))
rownames(weight) <- seq_len(nrow(weight))
write.csv(weight,"Important features for lung cancer_23.csv",row.names = F)

#save the dataset with missing value imputated 
#this complete dataset is generated in step 2 - extra
featureorder <- weight$features
data_impfeature <- data23_1619full[, c("SA2_5DIG16", "category", "sir", featureorder)]
write.csv(data_impfeature,"Imputed data with selected features_23.csv",row.names = F)

######################################################################
# Step 10 - manually assign direction of association to the features #
######################################################################
weight$positive<-c(1,0,0,0,1,0,1,1) #1 is positively associated with outcome, 0 is negatively, 2 is highest for average

data23_1619$category <- as.factor(data23_1619$category)
featurePlot(x=data23_1619[,c(featureorder)],
            y=data23_1619$category,
            plot="box")

#Step 11 - calculate the adjusted percentile rank for all the 8 features for the 519 sa2
rank <- data.frame(matrix(nrow=519))
rank$sa2 <- data23_1619full$SA2_5DIG16
rank$category <- data23_1619full$category
rank$sir <- data23_1619full$sir
rank <- rank[,-1]

#write a loop for all the 14 features
for (v in featureorder) {
  fmax=max(data_impfeature[[v]])
  fmin=min(data_impfeature[[v]])
  mm <- fmax-fmin
  n <- 519
  gap <- mm/(n-1)
  
    for (i in 1:519) {
     m = floor((data_impfeature[[v]][i]-fmin)/gap)+1
     rank[[v]][i] = m/n*100
  }
}

summary(rank)

#change the rank for the features which has a negative association with the outcome
negfeature <- weight$features[weight$positive ==0]
length(negfeature)

featurePlot(x=data23_1619[, c(negfeature)],
            y=data23_1619$category,
            plot="box")

#for features with a negative association with the outcome variable,
#gap=(V.max-V.min)/(N-1) 
#gap-adjusted rank m=floor[(V.max-v)/gap]+1
#percentile rank (range from 0 to 100) p=(m-1)/N*100

#write a loop for the negative features
for (v in negfeature) {
  fmax=max(data_impfeature[[v]])
  fmin=min(data_impfeature[[v]])
  mm <- fmax-fmin
  n <- 519
  gap <- mm/(n-1)
  
  for (i in 1:519) {
    m = floor((fmax-data_impfeature[[v]][i])/gap)+1
    rank[[v]][i] = m/n*100
  }
} 

rank$category <- factor(rank$category)
str(rank)
summary(rank) #Question: since the rank is gap adjusted. if there are extrem values in a certain feature, the rank for other features will be pushed together.
####################################################################################
# #now the rank of those negative features are positively associated with outcome
# featurePlot(x=rank[, c(negfeature)],
#             y=rank$category,
#             plot="box")   
# 
# #insurance is an important feature, however, the rank has been pushed to nearly 100 because of one outlier
# featurePlot(x=data23_1619[, c("insurance_rate")],
#             y=data23_1619$category,
#             plot="box")
# 
# featurePlot(x=rank[, c("insurance_rate")],
#             y=rank$category,
#             plot="box")
# 
# #check features with highest values in "Average" group (non-linear association)
# nonlfeature <- weight$features[weight$positive ==2]
# length(nonlfeature)
# #check in the origional value
# featurePlot(x=data23_1619[, c(nonlfeature)],
#             y=data23_1619$category,
#             plot="box")
# #check the rank - keep it for now, assuming it is positively associated with the outcome
# featurePlot(x=rank[, c(nonlfeature)],
#             y=rank$category,
#             plot="box")   
####################################################################################

#check for all the ranked features
featurePlot(x=rank[, c(featureorder)],
            y=rank$category,
            plot="box")   
################################
# Step 11 - generate the index #
################################
#Option 1: index for a SA2=sum(percentile rank*feature importance score)#
#Note: Gap adjusted percentile rank can be pushed to one end if have extrame values
index_gapadj <- data.frame(matrix(nrow=519))
index_gapadj$sa2 <- data23_1619full$SA2_5DIG16
index_gapadj$category <- data23_1619full$category
index_gapadj$sir <- data23_1619full$sir
index_gapadj <- index_gapadj[,-1]

for (i in 1:519) {
  for (v in featureorder) {
    feature.score=weight$impscore[weight$features == v]
    feature.rank=rank[i,v]
    index_gapadj[i,v] <- feature.score*feature.rank
  }
}

#index is the row sum of all the features
index_gapadj$index <- rowSums(index_gapadj[, 4:11])
index_gapadj$category <- factor(index_gapadj$category)
str(index_gapadj)

# Trim extreme index values
cut.offs <- c(quantile(index_gapadj$index, 0.05), quantile(index_gapadj$index, 0.95))
index_gapadj$index.clip <- index_gapadj$index
index_gapadj$index.clip[which(index_gapadj$index.clip < cut.offs[1], arr.ind = TRUE)] <- cut.offs[1]
index_gapadj$index.clip[which(index_gapadj$index.clip > cut.offs[2], arr.ind = TRUE)] <- cut.offs[2]
summary(index_gapadj$index.clip)

#transfer the index
index_gapadj$index.trans <- scale(index_gapadj$index.clip, center = min(index_gapadj$index.clip), scale = max(index_gapadj$index.clip) - min(index_gapadj$index.clip)) * 10
summary(index_gapadj$index.trans)

#include irsd in the dataset for sensitivity analysi
if(!all(data23_1619full$sir == index_gapadj$sir)) {
  stop("SIR values do not correspond between the two dataframes.")
}
index_gapadj$irsd <- data23_1619full$irsd_state_rank
index_gapadj$irsd_reversed <- 11 - index_gapadj$irsd #generate a reversed irsd, because irsd is negatively associated with SIR
write.csv(index_gapadj, "CVI for lung cancer.csv")

# #Option 2: index for a SA2 = sum(true value of each variable * feature importance score)#
# #Note: this true value is normalized using Z scale
# index_rawval <- data.frame(matrix(nrow=519))
# index_rawval$sa2 <- data23_1619full$SA2_5DIG16
# index_rawval$category <- data23_1619full$category
# index_rawval$sir <- data23_1619full$sir
# index_rawval <- index_rawval[,-1]
# 
# for (i in 1:519) {
#   for (v in featureorder) {
#     feature.score=weight$impscore[weight$features == v]
#     feature.rank=data23_1619_sub[i,v]  #value is normalized using Z scale
#     index_rawval[i,v] <- feature.score*feature.rank
#   }
# }
# #index is the row sum of all the features
# index_rawval$index <- rowSums(index_rawval[, 4:13])
# index_rawval$category <- factor(index_rawval$category)
# str(index_rawval)
# 
# #transfer the index
# index_rawval$index <- scale(index_rawval$index)
# summary(index_rawval$index)

#Step 12 - check the association between index and outcome
#check the index generated using option 1
featurePlot(x=index_gapadj$index.trans,
            y=index_gapadj$category,
            plot="box")   

# Create a boxplot using ggplot2 with color aesthetics
ggdata <- data.frame(index = index_gapadj$index.trans,
                   category = factor(index_gapadj$category, levels = c("Below", "Average", "Exceed")))
plot <- ggplot(ggdata, aes( x = category, y = index, fill = category)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Below" = "#2C7BB6", "Average" = "#FFFFBF", "Exceed" = "#D7191C")) +
  scale_x_discrete(labels = c("Below" = "Below", "Average" = "Average", "Exceed" = "Above")) +  # Change Exceed to Above
  labs(x = "Burden of lung cancer incidence (SIR) compared to QLD average", y = "Lung cancer vulnerability index (LcVI)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),  # axis line color
    text = element_text(size = 16),  # Text size for all elements
    legend.position = "none"  # Remove the legend
  ) 
ggsave("boxplot of index by category.png", plot, width = 8, height = 6, units = "in", dpi = 300)

#generate a violin plots
#Warning: since the median/ 25%/ 75% value of the violin plot relying on the density estimate, it is not consistent with the boxplot and the true values.
plot <- ggplot(ggdata, aes( x = category, y = index, fill = category)) +
  geom_violin(draw_quantiles = c(0.25, 0.75), linetype = "dotted") +
  geom_violin(draw_quantiles = c(0.5), fill = NA) +
  scale_fill_manual(values = c("Below" = "#2C7BB6", "Average" = "#FFFFBF", "Exceed" = "#D7191C")) +
  scale_x_discrete(labels = c("Below" = "Below", "Average" = "Average", "Exceed" = "Above")) +  # Change Exceed to Above
  labs(x = "Burden of lung cancer incidence (SIR) compared to QLD average", y = "Lung cancer vulnerability index (LcVI)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),  # axis line color
    text = element_text(size = 16),  # Text size for all elements
    legend.position = "none"  # Remove the legend
  )
ggsave("violinplot of index by category_estimatedline.png", plot, width = 8, height = 6, units = "in", dpi = 300)

plot <-
  ggplot(ggdata, aes( x = category, y = index, fill = category)) +
  geom_violin(color="white") +
  geom_boxplot(width = 0.5, fill = NA) +
  scale_fill_manual(values = c("Below" = "#2C7BB6", "Average" = "#FFFFBF", "Exceed" = "#D7191C")) +
  scale_x_discrete(labels = c("Below" = "Below", "Average" = "Average", "Exceed" = "Above")) +  # Change Exceed to Above
  labs(x = "Burden of lung cancer incidence (SIR) compared to QLD average", y = "Lung cancer vulnerability index (LcVI)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),  # axis line color
    text = element_text(size = 16),  # Text size for all elements
    legend.position = "none"  # Remove the legend
  )
ggsave("violinplot and boxplot of index by category.png", plot, width = 8, height = 6, units = "in", dpi = 300)

#since the median/ 25%/ 75% value of the violin plot relying on the density estimate, we need to manually draw the median values on top of the violin plot. 
# Calculate the 25th, 50th (median), and 75th percentiles for each category
quartiles <- ggdata %>%
  group_by(category) %>%
  summarise(
    Q1 = quantile(index, 0.25),
    Median = quantile(index, 0.5),
    Q3 = quantile(index, 0.75)
  )

plot <- ggplot(ggdata, aes(x = category, y = index, fill = category)) +
  geom_violin(fill = NA) +  # Keep the violin outline
  geom_violin() +  # Add the fill color for violins
  # Add lines for the 25th and 75th percentiles (dashed), covering the whole violin
  geom_segment(data = quartiles, aes(x = as.numeric(category) - 0.3, xend = as.numeric(category) + 0.3, 
                                     y = Q1, yend = Q1), color = "black", size = 0.5, linetype = "dashed") +  # 25th percentile line
  geom_segment(data = quartiles, aes(x = as.numeric(category) - 0.3, xend = as.numeric(category) + 0.3, 
                                     y = Q3, yend = Q3), color = "black", size = 0.5, linetype = "dashed") +  # 75th percentile line
  # Add the median line (solid), covering the whole violin
  geom_segment(data = quartiles, aes(x = as.numeric(category) - 0.33, xend = as.numeric(category) + 0.33, 
                                     y = Median, yend = Median), color = "black", size = 1, linetype = "solid") +  # Median line
  
  scale_fill_manual(values = c("Below" = "#2C7BB6", "Average" = "#FFFFBF", "Exceed" = "#D7191C")) +
  scale_x_discrete(labels = c("Below" = "Below", "Average" = "Average", "Exceed" = "Above")) +  # Change Exceed to Above
  labs(x = "Burden of lung cancer incidence (SIR) compared to QLD average", y = "Lung cancer vulnerability index (LcVI)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),  # axis line color
    text = element_text(size = 16),  # Text size for all elements
    legend.position = "none"  # Remove the legend
  )
ggsave("violinplot of index by category_trueline.png", plot, width = 8, height = 6, units = "in", dpi = 300)


#boxplot for IRSD
ggdata <- data.frame(index = index_gapadj$irsd_reversed,
                     category = factor(index_gapadj$category, levels = c("Below", "Average", "Exceed")))
plot <- ggplot(ggdata, aes( x = category, y = index, fill = category)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Below" = "#2C7BB6", "Average" = "#FFFFBF", "Exceed" = "#D7191C")) +
  scale_x_discrete(labels = c("Below" = "Below", "Average" = "Average", "Exceed" = "Above")) +  # Change Exceed to Above
  labs(x = "Burden of lung cancer incidence (SIR) compared to QLD average", y = "Relative Socio-economic Disadvantage (IRSD)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),  # axis line color
    text = element_text(size = 16),  # Text size for all elements
    legend.position = "none"  # Remove the legend
  ) 
ggsave("boxplot of reversed irsd by category.png", plot, width = 8, height = 6, units = "in", dpi = 300)

# Calculate quartiles and other statistics of index by category
CVIstatistics <- ggdata %>%
  group_by(category) %>%
  summarise(
    Median = median(index),
    Mean = mean(index),
    SD = sd(index),
    Min = min(index),
    Q1 = quantile(index, 0.25),  # 1st quartile
    Q3 = quantile(index, 0.75),  # 3rd quartile
    Max = max(index)
  )
print(CVIstatistics)

#ANOVA test che check the index and category
model.aov <- aov(index.trans ~ category, data = index_gapadj)
summary(model.aov)
# Conduct post-hoc test
posthoc <- TukeyHSD(model.aov)
print(posthoc)
plot(posthoc)

#ANOVA test che check the irsd and category
model.aov <- aov(irsd ~ category, data = index_gapadj)
summary(model.aov)
# Conduct post-hoc test
posthoc <- TukeyHSD(model.aov)
print(posthoc)
plot(posthoc)

# Fit the linear model
model_gapadj <- lm(log(sir) ~ index.trans, data = index_gapadj)
# Summarize the model to get the R-squared value
summary(model_gapadj)

# Extract the R-squared value
r_squared1 <- summary_model1$r.squared

#stratify data by category
# Subset data by category levels
data_cat1 <- subset(index_gapadj, category == levels(index_gapadj$category)[1])
data_cat2 <- subset(index_gapadj, category == levels(index_gapadj$category)[2])
data_cat3 <- subset(index_gapadj, category == levels(index_gapadj$category)[3])

# Fit linear models for each category
model_cat1 <- lm(log2(sir) ~ index.trans, data = data_cat1)
model_cat2 <- lm(log2(sir) ~ index.trans, data = data_cat2)
model_cat3 <- lm(log2(sir) ~ index.trans, data = data_cat3)

# Summarize models
summary(model_cat1)
summary(model_cat2)
summary(model_cat3)

# #check the index generated using option 2
# featurePlot(x=index_rawval$index,
#             y=index_rawval$category,
#             plot="box")   
# # Fit the linear model
# model_rawval <- lm(sir ~ index, data = index_rawval)
# # Summarize the model to get the R-squared value
# summary_model2 <- summary(model_rawval)
# print(summary_model2)
# # Extract the R-squared value
# r_squared2 <- summary_model1$r.squared
# 
# #Check the indipendent association between dm and lung incidence
# model_rawval <- lm(sir ~ dm_p_asr, data = data23_1619full)
# 
# model_rawval <- lm(sir ~ Study_field_education, data = data23_1619full)
# summary(model_rawval)

#calculate the avarage RAW values of features by category
rf_model=rf_boot_final_select$finalModel
print(varImp(rf_model))
plot(varImp(rf_model))

var=varImp(rf_model)
data <- data.frame(matrix(nrow=8))
data$features <- rf_model$xNames
data$impscore <- rf_model$importance
data<-subset(data, select=c(features, impscore))

features <- rf_model$xNames
data23_1619_rawsub <- data23_1619full[, c("category", features)]
avg_rawvalues <- aggregate(. ~ category, data = data23_1619_rawsub, median)
dataraw <- as.data.frame(t(avg_rawvalues))
names(dataraw)[names(dataraw)=="V1"] = "Average"
names(dataraw)[names(dataraw)=="V2"] = "Below"
names(dataraw)[names(dataraw)=="V3"] = "Exceed"
dataraw <- dataraw[-1,]
dataraw$features <- rownames(dataraw)
rownames(dataraw) <- seq_len(nrow(dataraw))

#merge average values to feature importance scores
rawcat<- merge(data, dataraw, by="features")
rawcat <- rawcat[,c("features", "impscore", "Below", "Average", "Exceed")]
rawcat <- arrange(rawcat, desc(impscore))
rownames(rawcat) <- seq_len(nrow(rawcat))
write.csv(rawcat,"Median values by category lung cancer_23.csv",row.names = F)

#plot the correlation for the features in the final model
selected_data <- data23_1619[, final_select]
#reorder the features based on importance
selected_data <- selected_data %>%
  select(
    dm_p_asr,
    fruit_p_asr,  
    private_insurance_rate_per_1000,
    Bachelor_degree,
    Unemployeed_rate,
    Median_Super_income_per_year,
    Completed_year_11,
    seperated_percentage
  ) %>%
  rename(
    `1 Diabetes` = dm_p_asr,
    `2 Adequate fruit intake` = fruit_p_asr,
    `3 Private health insurance` = private_insurance_rate_per_1000,
    `4 Bachelor Degree` = Bachelor_degree,
    `5 Unemployed` = Unemployeed_rate,
    `6 Superannuation/annuity income` = Median_Super_income_per_year,
    `7 Highest education as Year 11 or equivalent` = Completed_year_11,
    `8 Separated` = seperated_percentage
  )
correlation_matrix <- cor(selected_data, use = "complete.obs")
# Plot the heatmap
plot.new()
dev.off()
corrplot(correlation_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 90)

#plot the SIR and index for each SA2 
# #install.packages("ggnewscale")
# library(ggnewscale)
#plot the SIR and index for each SA2 
 
# # Create separate data for SIR and Index as having different colour scales
# sir_data <- index_gapadj %>% select(obs, sir)
# index_data <- index_gapadj %>% 
#   mutate(index_scaled = index.trans / 10 * (sir_range[2] - sir_range[1]) + sir_range[1]) %>% 
#   select(obs, index_scaled, sir)
# 
# #install.packages("ggnewscale")
# #library(ggnewscale)
# p <- ggplot() +
#   geom_point(data = sir_data, aes(x = obs, y = sir, color = sir), shape = 19, size = 1) +  #SIR with color gradient
#   geom_point(data = index_data, aes(x = obs, y = index_scaled, color = sir), shape = 8, size = 1) +  # Index with transformed scale and color gradient
#   scale_y_continuous(
#     name = "Standard incidence ratio (SIR)",
#     trans = "log2",
#     limits = sir_range,
#     sec.axis = sec_axis(~ (. - sir_range[1]) / (sir_range[2] - sir_range[1]) * 10, name = "Cancer vulnerability index (CVI)", breaks = seq(0, 10, by = 1))
#   ) +
#   scale_color_gradientn(colors = c("#2C7BB6", "#2C7BB6","#2C7BB6","#2C7BB6","#2C7BB6","#ABD9E9","#ABD9E9", "#FFFFBF","#FDAE61", "#FDAE61", "#D7191C","#D7191C","#D7191C","#D7191C","#D7191C"), values = scales::rescale(c(0.5, 2)), name = "SIR") +
#   labs(x = "519 SA2 in Queensland") +
#   theme_minimal() +
#   theme(
#     axis.line = element_line(color = "black"),
#     text = element_text(size = 12)
#   ) 
# # Display the plot
# print(p)
# ggsave("Plot of sir and cvi.png", p, width = 11, height = 5, units = "in", dpi = 1200)

# Order the data by sir
index_gapadj <- index_gapadj %>%
  arrange(sir) %>%
  mutate(obs = row_number())

# Define the range for the SIR and Index axes
sir_range <- c(0.4, 2)
index_range <- c(0, 10)

# Create separate data for SIR and Index as having different colour scales
index_data <- index_gapadj %>% 
  mutate(sir_scaled = (log2(sir) - log2(sir_range[1])) / (log2(sir_range[2]) - log2(sir_range[1])) * (index_range[2] - index_range[1]) + index_range[1]) %>% 
  select(obs, index.trans, sir_scaled, sir)

#install.packages("ggnewscale")
#library(ggnewscale)
p <- ggplot(index_data) +
  geom_point(aes(x = obs, y = sir_scaled, color = sir), shape = 19, size = 2) +  # SIR with color gradient
  geom_point(aes(x = obs, y = index.trans, color = sir), shape = 2, size = 2, show.legend = TRUE) +  # Index with normal scale and color gradient, same color as SIR
  scale_y_continuous(
    name = "Lung cancer vulnerability index (LcVI)",
    limits = index_range,
    breaks = seq(0, 10, by = 1),
    sec.axis = sec_axis(
      trans = ~ 2^(log2(sir_range[1]) + (. - index_range[1]) * (log2(sir_range[2]) - log2(sir_range[1])) / (index_range[2] - index_range[1])),
      name = "Standard incidence ratio (SIR)",
      labels = function(x) round(x, 2),
      breaks = c(0.5, 1, 2)
    )
  ) +
  scale_color_gradientn(
    colors = c("#2C7BB6", "#2C7BB6", "#2C7BB6", "#2C7BB6", "#ABD9E9", "#ABD9E9", "#FFFFBF", "#FDAE61", "#FDAE61", "#D7191C", "#D7191C", "#D7191C", "#D7191C"), 
    values = scales::rescale(c(0, 2)), 
    name = "SIR"
  ) +
  labs(x = "519 SA2 in Queensland") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),
    text = element_text(size = 12)
  )
# Display the plot
print(p)
ggsave("Plot of sir and cvi.png", p, width = 11, height = 5, units = "in", dpi = 1200)

#Map the association between IRSD and sir
model_irsd <- lm(log(sir) ~ irsd_state_rank, data = data23_1619full) 
summary(model_irsd)
#plot the SIR and irsd for each SA2 
data23_1619full$irsd_state_rank_reversed <- 11 - data23_1619full$irsd_state_rank #reversed irsd, because irsd is negatively related to sir
# Order the data by sir
data23_1619full <- data23_1619full %>%
  arrange(sir) %>%
  mutate(obs = row_number())

# Define the range for the SIR and Index axes
sir_range <- c(0.4, 2)
index_range <- c(1, 10)

# Create separate data for SIR and Index as having different color scales
index_data2 <- data23_1619full %>%
  mutate(sir_scaled = (log2(sir) - log2(sir_range[1])) / (log2(sir_range[2]) - log2(sir_range[1])) * (index_range[2] - index_range[1]) + index_range[1]) %>% 
  select(obs, irsd_state_rank, sir_scaled, sir)  # Ensure the 'sir' column is present

# Create the plot
q <- ggplot(index_data2) +
  geom_point(aes(x = obs, y = sir_scaled, color = sir), shape = 19, size = 2) +  # SIR with color gradient
  geom_point(aes(x = obs, y = irsd_state_rank, color = sir), shape = 4, size = 2, show.legend = TRUE) +  # Index with normal scale and color gradient, same color as SIR
  scale_y_continuous(
    name = "Relative Socio-economic Disadvantage (IRSD)",
    limits = index_range,
    breaks = seq(1, 10, by = 1),
    sec.axis = sec_axis(
      trans = ~ 2^(log2(sir_range[1]) + (. - index_range[1]) * (log2(sir_range[2]) - log2(sir_range[1])) / (index_range[2] - index_range[1])),
      name = "Standard incidence ratio (SIR)",
      labels = function(x) round(x, 2),
      breaks = c(0.5, 1, 2)
    )
  ) +
  scale_color_gradientn(
    colors = c("#2C7BB6", "#2C7BB6", "#2C7BB6", "#2C7BB6", "#ABD9E9", "#ABD9E9", "#FFFFBF", "#FDAE61", "#FDAE61", "#D7191C", "#D7191C", "#D7191C", "#D7191C"), 
    values = scales::rescale(c(0, 2)), 
    name = "SIR"
  ) +
  labs(x = "519 SA2 in Queensland") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),
    text = element_text(size = 12)
  )

# Display the plot
print(q)
ggsave("Plot of sir and irsd.png", q, width = 11, height = 5, units = "in", dpi = 1200)

#plot the association with reversed irsd
index_data3 <- data23_1619full %>%
  mutate(sir_scaled = (log2(sir) - log2(sir_range[1])) / (log2(sir_range[2]) - log2(sir_range[1])) * (index_range[2] - index_range[1]) + index_range[1]) %>% 
  select(obs, irsd_state_rank_reversed, sir_scaled, sir)  # Ensure the 'sir' column is present

# Create the plot
q <- ggplot(index_data3) +
  geom_point(aes(x = obs, y = sir_scaled, color = sir), shape = 19, size = 2) +  # SIR with color gradient
  geom_point(aes(x = obs, y = irsd_state_rank_reversed, color = sir), shape = 4, size = 2, show.legend = TRUE) +  # Index with normal scale and color gradient, same color as SIR
  scale_y_continuous(
    name = "Reversed Relative Socio-economic Disadvantage (IRSD)",
    limits = index_range,
    breaks = seq(1, 10, by = 1),
    sec.axis = sec_axis(
      trans = ~ 2^(log2(sir_range[1]) + (. - index_range[1]) * (log2(sir_range[2]) - log2(sir_range[1])) / (index_range[2] - index_range[1])),
      name = "Standard incidence ratio (SIR)",
      labels = function(x) round(x, 2),
      breaks = c(0.5, 1, 2)
    )
  ) +
  scale_color_gradientn(
    colors = c("#2C7BB6", "#2C7BB6", "#2C7BB6", "#2C7BB6", "#ABD9E9", "#ABD9E9", "#FFFFBF", "#FDAE61", "#FDAE61", "#D7191C", "#D7191C", "#D7191C", "#D7191C"), 
    values = scales::rescale(c(0, 2)), 
    name = "SIR"
  ) +
  labs(x = "519 SA2 in Queensland") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),
    text = element_text(size = 12)
  )

# Display the plot
print(q)
ggsave("Plot of sir and reversed irsd.png", q, width = 11, height = 5, units = "in", dpi = 1200)
