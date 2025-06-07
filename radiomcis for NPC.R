## For T stage
# The first affiliated hospital of Xiamen University for training
library(readr)
CT_features <- read_csv("xm_ct.csv")
PET_features <- read_csv("xm_pet.csv")
label <- read_csv("xm.csv")
library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'
colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'
data <- merge(CT_features,PET_features,by="Patient")
data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
train_data_label <- data_label
numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num
variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]
library(caret)
# 
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)
X_scaled <- X_scaled_clinical[,-c(1:12)]
y=data_label[,c(4:9)]          
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 


data_Tstage <- cbind.data.frame(data[,c(1,2)],data[,-c(1:6)])                      
data_Tstage1 <- data_Tstage[,-c(2)]                               
train <- data_Tstage1

significant_features <- list()
alpha <- 0.05


for (column in 2:ncol(train)) {
   
  feature_name <- colnames(train)[column]
  feature <- unlist(train[, column])
  
   
  label_0 <- feature[train[, 1] == 0]
  label_1 <- feature[train[, 1] == 1]
  
   
  shapiro_test <- shapiro.test(feature)
  if (shapiro_test$p.value < alpha) {
     
    mannwhitney_test <- wilcox.test(label_0, label_1)
    p_value <- mannwhitney_test$p.value
  } else {
     
    t_test <- t.test(label_0, label_1)
    p_value <- t_test$p.value
  }
  
   
  if (p_value < alpha) {
    significant_features[[feature_name]] <- p_value
  }
}

 
significant_features_df <- data.frame(
  Feature = names(significant_features),
  P_Value = unlist(significant_features)
)

significant_features_df <- significant_features_df[order(significant_features_df$P_Value), ]

X_train <- as.matrix(train[, significant_features_df$Feature])
y_train <- train[1]
y_train <- y_train[,1]
train_data <- cbind.data.frame(train[1],X_train)


set.seed(42)
library(glmnet)

X_train <- as.matrix(X_train) 
y_train <- as.factor(y_train)                                                                                 

cv.lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", nfolds = 10, maxit = 50000, type.measure = "class")
lasso.coef1 <- coef(cv.lasso, s = "lambda.min")


selected_features <- rownames(lasso.coef1)[lasso.coef1[, 1] != 0]
selected_features <- selected_features[-1]  

selected_feature_names <- selected_features

X_train_lasso_selected <- X_train[, selected_feature_names]
y_train

train_data <- cbind.data.frame(y_train,X_train_lasso_selected)
















# Hunan Cancer hospital for testing
library(readr)
CT_features <- read_csv("hn_ct.csv")
PET_features <- read_csv("hn_pet.csv")
label <- read_csv("hn.csv")

library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'
colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'
data <- merge(CT_features,PET_features,by="Patient")
data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
test_data_label <- data_label
numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num
variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]
library(caret)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)
X_scaled <- X_scaled_clinical[,-c(1:11)]
y=data_label[,c(5:10)]
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 

data_Tstage <- cbind.data.frame(data[,c(1,2)],data[,-c(1:6)])                      
data_Tstage1 <- data_Tstage[,-c(2)]                               
test <- data_Tstage1

X_test_lasso_selected <- test[, selected_feature_names]
y_test <- test[,1]
test_data <- cbind.data.frame(y_test,X_test_lasso_selected)











library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(pROC)
library(ggplot2)


#Logistic Regression
set.seed(42)
logreg_model <- glm(y_train ~ ., data = train_data, family = binomial,control = list(maxit=50000))
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_pred <- ifelse(logreg_pred > 0.5, 1, 0)
logreg_accuracy <- mean(logreg_pred == y_test)
logreg_conf_matrix <- table(Predicted = logreg_pred, Actual = y_test)


logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))

logreg_precision <- logreg_conf_matrix$byClass['Pos Pred Value']  
logreg_recall <- logreg_conf_matrix$byClass['Sensitivity']  
logreg_f1_score <- 2 * ((logreg_precision * logreg_recall) / (logreg_precision + logreg_recall))  


logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))


tn <- logreg_conf_matrix$table[1, 1]
fn <- logreg_conf_matrix$table[1, 2]
fp <- logreg_conf_matrix$table[2, 1]
tp <- logreg_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int

#ROC and AUC
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_roc <- roc(y_test, logreg_pred)
auc <- auc(logreg_roc)
auc_ci <- ci.auc(logreg_roc)
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)



NIR <- logreg_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- logreg_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- logreg_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)







# predicting on the training set
logreg_pred_train <- predict(logreg_model, as.data.frame(X_train_lasso_selected), type = "response")
logreg_pred_train <- ifelse(logreg_pred_train > 0.5, 1, 0)
logreg_conf_matrix_train <- confusionMatrix(as.factor(logreg_pred_train), as.factor(y_train))
tn_train <- logreg_conf_matrix_train$table[1, 1]
fn_train <- logreg_conf_matrix_train$table[1, 2]
fp_train <- logreg_conf_matrix_train$table[2, 1]
tp_train <- logreg_conf_matrix_train$table[2, 2]
# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int
sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int
specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int
logreg_pred_train <- predict(logreg_model, as.data.frame(X_train_lasso_selected), type = "response")
logreg_roc_train <- roc(y_train, logreg_pred_train)
auc_train <- auc(logreg_roc_train)
auc_ci_train <- ci.auc(logreg_roc_train)
# Printing the result of Training set
cat("Logistic Regression Model (Training Set) Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)







#Random Forest
set.seed(42)
y_train <- as.factor(y_train)
#rf_model <- randomForest(x = X_train_lasso_selected, y = as.factor(y_train), ntree = 100)
rf_model <- randomForest(x = train_data[, -1] , y = train_data[, 1] , 
                         ntree = 150,   
                         importance = TRUE,     
                         do.trace = FALSE,       
                         keep.forest = TRUE,
                         nodesize = 9  , 
                         oob.prox = TRUE)       

rf_pred <- predict(rf_model, X_test_lasso_selected)
rf_accuracy <- mean(rf_pred == y_test)
rf_conf_matrix <- table(Predicted = rf_pred, Actual = y_test)
rf_probs <- predict(rf_model, X_test_lasso_selected, type = "prob")
rf_roc <- roc(y_test, rf_probs[, 2])
auc <- auc(rf_roc)
auc_ci <- ci.auc(rf_roc)

rf_conf_matrix <- confusionMatrix(as.factor(rf_pred), as.factor(y_test))


tn <- rf_conf_matrix$table[1, 1]
fn <- rf_conf_matrix$table[1, 2]
fp <- rf_conf_matrix$table[2, 1]
tp <- rf_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)


NIR <- rf_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- rf_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- rf_conf_matrix[["overall"]][["Kappa"]]
print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
rf_pred_train <- predict(rf_model, X_train_lasso_selected)
rf_accuracy_train <- mean(rf_pred_train == y_train)
rf_conf_matrix_train <- table(Predicted = rf_pred_train, Actual = y_train)

rf_probs_train <- predict(rf_model, X_train_lasso_selected, type = "prob")
rf_roc_train <- roc(y_train, rf_probs_train[, 2])
auc_train <- auc(rf_roc_train)
auc_ci_train <- ci.auc(rf_roc_train)
rf_conf_matrix_train <- confusionMatrix(as.factor(rf_pred_train), as.factor(y_train))
tn_train <- rf_conf_matrix_train$table[1, 1]
fn_train <- rf_conf_matrix_train$table[1, 2]
fp_train <- rf_conf_matrix_train$table[2, 1]
tp_train <- rf_conf_matrix_train$table[2, 2]
# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






#XGBoost-
set.seed(42)
y_train <- as.numeric(as.character(y_train))
xgb_model <- xgboost(
  data = X_train_lasso_selected, 
  label = y_train,
  nrounds = 100,           
  objective = "binary:logistic", 
  eval_metric = "logloss", 
  eta = 0.05,           
  max_depth = 6, 
  min_child_weight = 1,  
  subsample = 0.8, 
  colsample_bytree = 0.7, 
  gamma = 1          
)




xgb_pred <- predict(xgb_model, as.matrix(X_test_lasso_selected))
xgb_roc <- roc(y_test, xgb_pred)
auc <- auc(xgb_roc)
auc_ci <- ci.auc(xgb_roc)

xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)
xgb_accuracy <- mean(xgb_pred == y_test)
xgb_conf_matrix <- table(Predicted = xgb_pred, Actual = y_test)
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred), as.factor(y_test))


tn <- xgb_conf_matrix$table[1, 1]
fn <- xgb_conf_matrix$table[1, 2]
fp <- xgb_conf_matrix$table[2, 1]
tp <- xgb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)





NIR <- xgb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- xgb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- xgb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
xgb_pred_train <- predict(xgb_model, as.matrix(X_train_lasso_selected))
xgb_roc_train <- roc(y_train, xgb_pred_train)
auc_train <- auc(xgb_roc_train)
auc_ci_train <- ci.auc(xgb_roc_train)
xgb_pred_train <- ifelse(xgb_pred_train > 0.5, 1, 0)
xgb_accuracy_train <- mean(xgb_pred_train == y_train)
xgb_conf_matrix_train <- table(Predicted = xgb_pred_train, Actual = y_train)
xgb_conf_matrix_train <- confusionMatrix(as.factor(xgb_pred_train), as.factor(y_train))

tn_train <- xgb_conf_matrix_train$table[1, 1]
fn_train <- xgb_conf_matrix_train$table[1, 2]
fp_train <- xgb_conf_matrix_train$table[2, 1]
tp_train <- xgb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)















## Naive Bayes
set.seed(42)
nb_model <- naiveBayes(X_train_lasso_selected, as.factor(y_train))
nb_pred <- predict(nb_model, X_test_lasso_selected)
nb_accuracy <- mean(nb_pred == y_test)
nb_conf_matrix <- table(Predicted = nb_pred, Actual = y_test)
nb_probs <- predict(nb_model, X_test_lasso_selected, type = "raw")
nb_roc <- roc(y_test, nb_probs[, 2])
auc <- auc(nb_roc)
auc_ci <- ci.auc(nb_roc)
nb_conf_matrix <- confusionMatrix(as.factor(nb_pred), as.factor(y_test))


tn <- nb_conf_matrix$table[1, 1]
fn <- nb_conf_matrix$table[1, 2]
fp <- nb_conf_matrix$table[2, 1]
tp <- nb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(nb_roc)
print(auc_ci)


NIR <- nb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- nb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- nb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)






nb_pred_train <- predict(nb_model, X_train_lasso_selected)
nb_accuracy_train <- mean(nb_pred_train == y_train)
nb_conf_matrix_train <- table(Predicted = nb_pred_train, Actual = y_train)

nb_probs_train <- predict(nb_model, X_train_lasso_selected, type = "raw")
nb_roc_train <- roc(y_train, nb_probs_train[, 2])
auc_train <- auc(nb_roc_train)
auc_ci_train <- ci.auc(nb_roc_train)


nb_conf_matrix_train <- confusionMatrix(as.factor(nb_pred_train), as.factor(y_train))

tn_train <- nb_conf_matrix_train$table[1, 1]
fn_train <- nb_conf_matrix_train$table[1, 2]
fp_train <- nb_conf_matrix_train$table[2, 1]
tp_train <- nb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(nb_roc_train)
print(auc_ci_train)







## SVM 
set.seed(42)
# Tune the parameters and specify the linear kernel function
tune_result <- tune(
  svm, 
  train.x = X_train_lasso_selected, 
  train.y = as.factor(y_train),
  ranges = list(cost = 10^(-3:2)), 
  kernel = "linear",       
  probability = TRUE
)

# Acquiring the optimal model
svm_model <- tune_result$best.model

svm_pred <- predict(svm_model, X_test_lasso_selected, probability = TRUE)
svm_pred_labels <- as.numeric(svm_pred) - 1  
svm_accuracy <- mean(svm_pred_labels == y_test)
svm_conf_matrix <- table(Predicted = svm_pred_labels, Actual = y_test)
svm_probs <- attr(svm_pred, "probabilities")[,2]
svm_roc <- roc(y_test, svm_probs)
auc <- auc(svm_roc)
auc_ci <- ci.auc(svm_roc)


svm_conf_matrix <- confusionMatrix(as.factor(svm_pred), as.factor(y_test))


tn <- svm_conf_matrix$table[1, 1]
fn <- svm_conf_matrix$table[1, 2]
fp <- svm_conf_matrix$table[2, 1]
tp <- svm_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)



NIR <- svm_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- svm_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- svm_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)








# prediction on training set
svm_pred_train <- predict(svm_model, X_train_lasso_selected, probability = TRUE)
svm_pred_labels_train <- as.numeric(svm_pred_train) - 1  
# Accuracy on training set
svm_accuracy_train <- mean(svm_pred_labels_train == y_train)

svm_conf_matrix_train <- table(Predicted = svm_pred_labels_train, Actual = y_train)
svm_probs_train <- attr(svm_pred_train, "probabilities")[, 2]
svm_roc_train <- roc(y_train, svm_probs_train)
auc_train <- auc(svm_roc_train)
auc_ci_train <- ci.auc(svm_roc_train)

tn_train <- svm_conf_matrix_train[1, 1]
fn_train <- svm_conf_matrix_train[1, 2]
fp_train <- svm_conf_matrix_train[2, 1]
tp_train <- svm_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






## KNN
set.seed(42)
knn_model <- train(train_data[, -1], as.factor(train_data[, 1]), method = "knn", trControl = trainControl(method = "cv", number = 5))
knn_pred <- predict(knn_model, as.data.frame(X_test_lasso_selected))
knn_accuracy <- mean(knn_pred == y_test)
knn_conf_matrix <- table(Predicted = knn_pred, Actual = y_test)
knn_probs <- predict(knn_model, X_test_lasso_selected, type = "prob")
knn_roc <- roc(y_test, knn_probs[, 2])
auc <- auc(knn_roc)
auc_ci <- ci.auc(knn_roc)

knn_conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(y_test))


tn <- knn_conf_matrix$table[1, 1]
fn <- knn_conf_matrix$table[1, 2]
fp <- knn_conf_matrix$table[2, 1]
tp <- knn_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print( auc)
print(auc_ci)








NIR <- knn_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- knn_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- knn_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)









# prediction on training set
knn_pred_train <- predict(knn_model, as.data.frame(X_train_lasso_selected))
# Accuracy on training set
knn_accuracy_train <- mean(knn_pred_train == y_train)
knn_conf_matrix_train <- table(Predicted = knn_pred_train, Actual = y_train)
knn_probs_train <- predict(knn_model, X_train_lasso_selected, type = "prob")
knn_roc_train <- roc(y_train, knn_probs_train[, 2])
auc_train <- auc(knn_roc_train)
auc_ci_train <- ci.auc(knn_roc_train)

tn_train <- knn_conf_matrix_train[1, 1]
fn_train <- knn_conf_matrix_train[1, 2]
fp_train <- knn_conf_matrix_train[2, 1]
tp_train <- knn_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("KNN Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)











## For N stage
# The first affiliated hospital of Xiamen University for training
library(readr)
CT_features <- read_csv("xm_ct.csv")
PET_features <- read_csv("xm_pet.csv")
label <- read_csv("xm.csv")
library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'

colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'

data <- merge(CT_features,PET_features,by="Patient")

data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
train_data_label <- data_label



numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num


variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]


library(caret)
# 
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)

X_scaled <- X_scaled_clinical[,-c(1:12)]




y=data_label[,c(4:9)]          
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 








data_Nstage <- cbind.data.frame(data[,c(3,4)],data[,-c(1:6)])                      
data_Nstage1 <- data_Nstage[,-c(2)]                               
train <- data_Nstage1









         







significant_features <- list()
alpha <- 0.05


for (column in 2:ncol(train)) {
   
  feature_name <- colnames(train)[column]
  feature <- unlist(train[, column])
  
   
  label_0 <- feature[train[, 1] == 0]
  label_1 <- feature[train[, 1] == 1]
  
   
  shapiro_test <- shapiro.test(feature)
  if (shapiro_test$p.value < alpha) {
     
    mannwhitney_test <- wilcox.test(label_0, label_1)
    p_value <- mannwhitney_test$p.value
  } else {
     
    t_test <- t.test(label_0, label_1)
    p_value <- t_test$p.value
  }
  
   
  if (p_value < alpha) {
    significant_features[[feature_name]] <- p_value
  }
}

 
significant_features_df <- data.frame(
  Feature = names(significant_features),
  P_Value = unlist(significant_features)
)

significant_features_df <- significant_features_df[order(significant_features_df$P_Value), ]






X_train <- as.matrix(train[, significant_features_df$Feature])
y_train <- train[1]
y_train <- y_train[,1]
train_data <- cbind.data.frame(train[1],X_train)
















set.seed(42)
library(glmnet)

X_train <- as.matrix(X_train) 
y_train <- as.factor(y_train)                                                                                 

cv.lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", nfolds = 10, maxit = 50000, type.measure = "class")
lasso.coef1 <- coef(cv.lasso, s = "lambda.min")




selected_features <- rownames(lasso.coef1)[lasso.coef1[, 1] != 0]
selected_features <- selected_features[-1]  

selected_feature_names <- selected_features

X_train_lasso_selected <- X_train[, selected_feature_names]
y_train

train_data <- cbind.data.frame(y_train,X_train_lasso_selected)








# Hunan Cancer hospital for testing
library(readr)
CT_features <- read_csv("hn_ct.csv")
PET_features <- read_csv("hn_pet.csv")
label <- read_csv("hn.csv")


library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'
colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'
data <- merge(CT_features,PET_features,by="Patient")
data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
test_data_label <- data_label
numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num
variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]
library(caret)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)
X_scaled <- X_scaled_clinical[,-c(1:11)]
y=data_label[,c(5:10)]
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 


data_Nstage <- cbind.data.frame(data[,c(3,4)],data[,-c(1:6)])                      
data_Nstage1 <- data_Nstage[,-c(2)]                               
test <- data_Nstage1



X_test_lasso_selected <- test[, selected_feature_names]
y_test <- test$Nstage1

test_data <- cbind.data.frame(y_test,X_test_lasso_selected)














library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(pROC)
library(ggplot2)

train_data <- train_data[,!colnames(train_data) %in% c("original_shape_Elongation_ct")]
X_train_lasso_selected <- X_train_lasso_selected[,!colnames(X_train_lasso_selected) %in% c("original_shape_Elongation_ct")]
test_data <- test_data[,!colnames(test_data) %in% c("original_shape_Elongation_ct")]
X_test_lasso_selected <- X_test_lasso_selected[,!colnames(X_test_lasso_selected) %in% c("original_shape_Elongation_ct")]
##Logistic Regression
set.seed(42)
logreg_model <- glm(y_train ~ ., data = train_data, family = binomial,control = list(maxit=50000))
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_pred <- ifelse(logreg_pred > 0.5, 1, 0)
logreg_accuracy <- mean(logreg_pred == y_test)
logreg_conf_matrix <- table(Predicted = logreg_pred, Actual = y_test)
logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))
logreg_precision <- logreg_conf_matrix$byClass['Pos Pred Value']  
logreg_recall <- logreg_conf_matrix$byClass['Sensitivity']  
logreg_f1_score <- 2 * ((logreg_precision * logreg_recall) / (logreg_precision + logreg_recall))  
logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))

tn <- logreg_conf_matrix$table[1, 1]
fn <- logreg_conf_matrix$table[1, 2]
fp <- logreg_conf_matrix$table[2, 1]
tp <- logreg_conf_matrix$table[2, 2]
# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int
sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int
specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_roc <- roc(y_test, logreg_pred)
auc_ci <- ci.auc(logreg_roc)
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(logreg_roc)
print(auc_ci)





#Random Forest
set.seed(42)
y_train <- as.factor(y_train)
#rf_model <- randomForest(x = X_train_lasso_selected, y = as.factor(y_train), ntree = 100)
rf_model <- randomForest(x = train_data[, -1] , y = train_data[, 1] , 
                         ntree = 150,   
                         importance = TRUE,     
                         do.trace = FALSE,       
                         keep.forest = TRUE,
                         nodesize = 9  , 
                         oob.prox = TRUE)       

rf_pred <- predict(rf_model, X_test_lasso_selected)
rf_accuracy <- mean(rf_pred == y_test)
rf_conf_matrix <- table(Predicted = rf_pred, Actual = y_test)
rf_probs <- predict(rf_model, X_test_lasso_selected, type = "prob")
rf_roc <- roc(y_test, rf_probs[, 2])
auc <- auc(rf_roc)
auc_ci <- ci.auc(rf_roc)

rf_conf_matrix <- confusionMatrix(as.factor(rf_pred), as.factor(y_test))


tn <- rf_conf_matrix$table[1, 1]
fn <- rf_conf_matrix$table[1, 2]
fp <- rf_conf_matrix$table[2, 1]
tp <- rf_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)


NIR <- rf_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- rf_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- rf_conf_matrix[["overall"]][["Kappa"]]
print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
rf_pred_train <- predict(rf_model, X_train_lasso_selected)
rf_accuracy_train <- mean(rf_pred_train == y_train)
rf_conf_matrix_train <- table(Predicted = rf_pred_train, Actual = y_train)

rf_probs_train <- predict(rf_model, X_train_lasso_selected, type = "prob")
rf_roc_train <- roc(y_train, rf_probs_train[, 2])
auc_train <- auc(rf_roc_train)
auc_ci_train <- ci.auc(rf_roc_train)
rf_conf_matrix_train <- confusionMatrix(as.factor(rf_pred_train), as.factor(y_train))
tn_train <- rf_conf_matrix_train$table[1, 1]
fn_train <- rf_conf_matrix_train$table[1, 2]
fp_train <- rf_conf_matrix_train$table[2, 1]
tp_train <- rf_conf_matrix_train$table[2, 2]
# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






#XGBoost-
set.seed(42)
y_train <- as.numeric(as.character(y_train))
xgb_model <- xgboost(
  data = X_train_lasso_selected, 
  label = y_train,
  nrounds = 100,           
  objective = "binary:logistic", 
  eval_metric = "logloss", 
  eta = 0.05,           
  max_depth = 6, 
  min_child_weight = 1,  
  subsample = 0.8, 
  colsample_bytree = 0.7, 
  gamma = 1          
)




xgb_pred <- predict(xgb_model, as.matrix(X_test_lasso_selected))
xgb_roc <- roc(y_test, xgb_pred)
auc <- auc(xgb_roc)
auc_ci <- ci.auc(xgb_roc)

xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)
xgb_accuracy <- mean(xgb_pred == y_test)
xgb_conf_matrix <- table(Predicted = xgb_pred, Actual = y_test)
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred), as.factor(y_test))


tn <- xgb_conf_matrix$table[1, 1]
fn <- xgb_conf_matrix$table[1, 2]
fp <- xgb_conf_matrix$table[2, 1]
tp <- xgb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)





NIR <- xgb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- xgb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- xgb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
xgb_pred_train <- predict(xgb_model, as.matrix(X_train_lasso_selected))
xgb_roc_train <- roc(y_train, xgb_pred_train)
auc_train <- auc(xgb_roc_train)
auc_ci_train <- ci.auc(xgb_roc_train)
xgb_pred_train <- ifelse(xgb_pred_train > 0.5, 1, 0)
xgb_accuracy_train <- mean(xgb_pred_train == y_train)
xgb_conf_matrix_train <- table(Predicted = xgb_pred_train, Actual = y_train)
xgb_conf_matrix_train <- confusionMatrix(as.factor(xgb_pred_train), as.factor(y_train))

tn_train <- xgb_conf_matrix_train$table[1, 1]
fn_train <- xgb_conf_matrix_train$table[1, 2]
fp_train <- xgb_conf_matrix_train$table[2, 1]
tp_train <- xgb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)















## Naive Bayes
set.seed(42)
nb_model <- naiveBayes(X_train_lasso_selected, as.factor(y_train))
nb_pred <- predict(nb_model, X_test_lasso_selected)
nb_accuracy <- mean(nb_pred == y_test)
nb_conf_matrix <- table(Predicted = nb_pred, Actual = y_test)
nb_probs <- predict(nb_model, X_test_lasso_selected, type = "raw")
nb_roc <- roc(y_test, nb_probs[, 2])
auc <- auc(nb_roc)
auc_ci <- ci.auc(nb_roc)
nb_conf_matrix <- confusionMatrix(as.factor(nb_pred), as.factor(y_test))


tn <- nb_conf_matrix$table[1, 1]
fn <- nb_conf_matrix$table[1, 2]
fp <- nb_conf_matrix$table[2, 1]
tp <- nb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(nb_roc)
print(auc_ci)


NIR <- nb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- nb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- nb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)






nb_pred_train <- predict(nb_model, X_train_lasso_selected)
nb_accuracy_train <- mean(nb_pred_train == y_train)
nb_conf_matrix_train <- table(Predicted = nb_pred_train, Actual = y_train)

nb_probs_train <- predict(nb_model, X_train_lasso_selected, type = "raw")
nb_roc_train <- roc(y_train, nb_probs_train[, 2])
auc_train <- auc(nb_roc_train)
auc_ci_train <- ci.auc(nb_roc_train)


nb_conf_matrix_train <- confusionMatrix(as.factor(nb_pred_train), as.factor(y_train))

tn_train <- nb_conf_matrix_train$table[1, 1]
fn_train <- nb_conf_matrix_train$table[1, 2]
fp_train <- nb_conf_matrix_train$table[2, 1]
tp_train <- nb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(nb_roc_train)
print(auc_ci_train)







## SVM 
set.seed(42)
# Tune the parameters and specify the linear kernel function
tune_result <- tune(
  svm, 
  train.x = X_train_lasso_selected, 
  train.y = as.factor(y_train),
  ranges = list(cost = 10^(-3:2)), 
  kernel = "linear",       
  probability = TRUE
)

# Acquiring the optimal model
svm_model <- tune_result$best.model

svm_pred <- predict(svm_model, X_test_lasso_selected, probability = TRUE)
svm_pred_labels <- as.numeric(svm_pred) - 1  
svm_accuracy <- mean(svm_pred_labels == y_test)
svm_conf_matrix <- table(Predicted = svm_pred_labels, Actual = y_test)
svm_probs <- attr(svm_pred, "probabilities")[,2]
svm_roc <- roc(y_test, svm_probs)
auc <- auc(svm_roc)
auc_ci <- ci.auc(svm_roc)


svm_conf_matrix <- confusionMatrix(as.factor(svm_pred), as.factor(y_test))


tn <- svm_conf_matrix$table[1, 1]
fn <- svm_conf_matrix$table[1, 2]
fp <- svm_conf_matrix$table[2, 1]
tp <- svm_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)



NIR <- svm_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- svm_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- svm_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)








# prediction on training set
svm_pred_train <- predict(svm_model, X_train_lasso_selected, probability = TRUE)
svm_pred_labels_train <- as.numeric(svm_pred_train) - 1  
# Accuracy on training set
svm_accuracy_train <- mean(svm_pred_labels_train == y_train)

svm_conf_matrix_train <- table(Predicted = svm_pred_labels_train, Actual = y_train)
svm_probs_train <- attr(svm_pred_train, "probabilities")[, 2]
svm_roc_train <- roc(y_train, svm_probs_train)
auc_train <- auc(svm_roc_train)
auc_ci_train <- ci.auc(svm_roc_train)

tn_train <- svm_conf_matrix_train[1, 1]
fn_train <- svm_conf_matrix_train[1, 2]
fp_train <- svm_conf_matrix_train[2, 1]
tp_train <- svm_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






## KNN
set.seed(42)
knn_model <- train(train_data[, -1], as.factor(train_data[, 1]), method = "knn", trControl = trainControl(method = "cv", number = 5))
knn_pred <- predict(knn_model, as.data.frame(X_test_lasso_selected))
knn_accuracy <- mean(knn_pred == y_test)
knn_conf_matrix <- table(Predicted = knn_pred, Actual = y_test)
knn_probs <- predict(knn_model, X_test_lasso_selected, type = "prob")
knn_roc <- roc(y_test, knn_probs[, 2])
auc <- auc(knn_roc)
auc_ci <- ci.auc(knn_roc)

knn_conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(y_test))


tn <- knn_conf_matrix$table[1, 1]
fn <- knn_conf_matrix$table[1, 2]
fp <- knn_conf_matrix$table[2, 1]
tp <- knn_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print( auc)
print(auc_ci)








NIR <- knn_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- knn_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- knn_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)









# prediction on training set
knn_pred_train <- predict(knn_model, as.data.frame(X_train_lasso_selected))
# Accuracy on training set
knn_accuracy_train <- mean(knn_pred_train == y_train)
knn_conf_matrix_train <- table(Predicted = knn_pred_train, Actual = y_train)
knn_probs_train <- predict(knn_model, X_train_lasso_selected, type = "prob")
knn_roc_train <- roc(y_train, knn_probs_train[, 2])
auc_train <- auc(knn_roc_train)
auc_ci_train <- ci.auc(knn_roc_train)

tn_train <- knn_conf_matrix_train[1, 1]
fn_train <- knn_conf_matrix_train[1, 2]
fp_train <- knn_conf_matrix_train[2, 1]
tp_train <- knn_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("KNN Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






# For M stage

# The first affiliated hospital of Xiamen University for training
library(readr)
CT_features <- read_csv("xm_ct.csv")
PET_features <- read_csv("xm_pet.csv")
label <- read_csv("xm.csv")

library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'

colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'

data <- merge(CT_features,PET_features,by="Patient")

data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
train_data_label <- data_label



numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num


variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]


library(caret)
# 
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)

X_scaled <- X_scaled_clinical[,-c(1:12)]




y=data_label[,c(4:9)]          
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 


data_Mstage <- cbind.data.frame(data[,c(5,6)],data[,-c(1:6)])                      
data_Mstage1 <- data_Mstage[,-c(2)]                               
train <- data_Mstage1


significant_features <- list()
alpha <- 0.05


for (column in 2:ncol(train)) {
   
  feature_name <- colnames(train)[column]
  feature <- unlist(train[, column])
  
   
  label_0 <- feature[train[, 1] == 0]
  label_1 <- feature[train[, 1] == 1]
  
   
  shapiro_test <- shapiro.test(feature)
  if (shapiro_test$p.value < alpha) {
     
    mannwhitney_test <- wilcox.test(label_0, label_1)
    p_value <- mannwhitney_test$p.value
  } else {
     
    t_test <- t.test(label_0, label_1)
    p_value <- t_test$p.value
  }
  
   
  if (p_value < alpha) {
    significant_features[[feature_name]] <- p_value
  }
}

 
significant_features_df <- data.frame(
  Feature = names(significant_features),
  P_Value = unlist(significant_features)
)

significant_features_df <- significant_features_df[order(significant_features_df$P_Value), ]


X_train <- as.matrix(train[, significant_features_df$Feature])
y_train <- train[1]
y_train <- y_train[,1]
train_data <- cbind.data.frame(train[1],X_train)


set.seed(42)
library(glmnet)

X_train <- as.matrix(X_train) 
y_train <- as.factor(y_train)                                                                                 

cv.lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", nfolds = 10, maxit = 50000, type.measure = "class")
lasso.coef1 <- coef(cv.lasso, s = "lambda.min")




selected_features <- rownames(lasso.coef1)[lasso.coef1[, 1] != 0]
selected_features <- selected_features[-1]  

selected_feature_names <- selected_features

X_train_lasso_selected <- X_train[, selected_feature_names]
y_train

train_data <- cbind.data.frame(y_train,X_train_lasso_selected)






# Hunan Cancer hospital for testing
library(readr)
CT_features <- read_csv("hn_ct.csv")
PET_features <- read_csv("hn_pet.csv")
label <- read_csv("hn.csv")

library(dplyr)
library(stringr)
colnames(CT_features) <- str_c(colnames(CT_features), "_ct")
colnames(CT_features)[1] <- 'Patient'
colnames(PET_features) <- str_c(colnames(PET_features), "_pet")
colnames(PET_features)[1] <- 'Patient'
data <- merge(CT_features,PET_features,by="Patient")
data_label <- merge(data,label,by='Patient')
data_label <- data_label[, c(names(label), setdiff(names(data_label), names(label)))]
rownames(data_label) <-  data_label$Patient
test_data_label <- data_label
numeric_columns <- sapply(data_label, is.numeric)
data_df_num <- data_label[, numeric_columns]
X <- data_df_num
variance <- apply(X, 2, var)
non_zero_variance_indices <- which(variance > 1e-8)
X <- X[, non_zero_variance_indices]
library(caret)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled_clinical <- predict(scaler, X)
X_scaled <- X_scaled_clinical[,-c(1:11)]
y=data_label[,c(5:10)]
data_normalized <- cbind(y,X_scaled)
data_normalized2 <- na.omit(data_normalized)
data = data_normalized2 


data_Mstage <- cbind.data.frame(data[,c(5,6)],data[,-c(1:6)])                      
data_Mstage1 <- data_Mstage[,-c(2)]                               
test <- data_Mstage1


X_test_lasso_selected <- test[, selected_feature_names]
y_test <- test[,1]

test_data <- cbind.data.frame(y_test,X_test_lasso_selected)



library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(pROC)
library(ggplot2)


##Logistic Regression
set.seed(42)
logreg_model <- glm(y_train ~ ., data = train_data, family = binomial,control = list(maxit=50000))
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_pred <- ifelse(logreg_pred > 0.5, 1, 0)
logreg_accuracy <- mean(logreg_pred == y_test)
logreg_conf_matrix <- table(Predicted = logreg_pred, Actual = y_test)
logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))
logreg_precision <- logreg_conf_matrix$byClass['Pos Pred Value']  
logreg_recall <- logreg_conf_matrix$byClass['Sensitivity']  
logreg_f1_score <- 2 * ((logreg_precision * logreg_recall) / (logreg_precision + logreg_recall))  
logreg_conf_matrix <- confusionMatrix(as.factor(logreg_pred), as.factor(y_test))

tn <- logreg_conf_matrix$table[1, 1]
fn <- logreg_conf_matrix$table[1, 2]
fp <- logreg_conf_matrix$table[2, 1]
tp <- logreg_conf_matrix$table[2, 2]
# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int
sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int
specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
logreg_pred <- predict(logreg_model, as.data.frame(X_test_lasso_selected), type = "response")
logreg_roc <- roc(y_test, logreg_pred)
auc_ci <- ci.auc(logreg_roc)
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(logreg_roc)
print(auc_ci)












# predicting on the training set
logreg_pred_train <- predict(logreg_model, as.data.frame(X_train_lasso_selected), type = "response")
logreg_pred_train <- ifelse(logreg_pred_train > 0.5, 1, 0)
logreg_conf_matrix_train <- confusionMatrix(as.factor(logreg_pred_train), as.factor(y_train))
logreg_precision_train <- logreg_conf_matrix_train$byClass['Pos Pred Value']  
logreg_recall_train <- logreg_conf_matrix_train$byClass['Sensitivity']  
logreg_f1_score_train <- 2 * ((logreg_precision_train * logreg_recall_train) / (logreg_precision_train + logreg_recall_train))  
tn_train <- logreg_conf_matrix_train$table[1, 1]
fn_train <- logreg_conf_matrix_train$table[1, 2]
fp_train <- logreg_conf_matrix_train$table[2, 1]
tp_train <- logreg_conf_matrix_train$table[2, 2]
# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int
sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int
specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int
logreg_roc_train <- roc(y_train, logreg_pred_train)
auc_ci_train <- ci.auc(logreg_roc_train)
# Printing the result of Training set
cat("Logistic Regression Model (Training Set) Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(logreg_roc_train)
print(auc_ci_train)








#Random Forest
set.seed(42)
y_train <- as.factor(y_train)
#rf_model <- randomForest(x = X_train_lasso_selected, y = as.factor(y_train), ntree = 100)
rf_model <- randomForest(x = train_data[, -1] , y = train_data[, 1] , 
                         ntree = 150,   
                         importance = TRUE,     
                         do.trace = FALSE,       
                         keep.forest = TRUE,
                         nodesize = 9  , 
                         oob.prox = TRUE)       

rf_pred <- predict(rf_model, X_test_lasso_selected)
rf_accuracy <- mean(rf_pred == y_test)
rf_conf_matrix <- table(Predicted = rf_pred, Actual = y_test)
rf_probs <- predict(rf_model, X_test_lasso_selected, type = "prob")
rf_roc <- roc(y_test, rf_probs[, 2])
auc <- auc(rf_roc)
auc_ci <- ci.auc(rf_roc)

rf_conf_matrix <- confusionMatrix(as.factor(rf_pred), as.factor(y_test))


tn <- rf_conf_matrix$table[1, 1]
fn <- rf_conf_matrix$table[1, 2]
fp <- rf_conf_matrix$table[2, 1]
tp <- rf_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)


NIR <- rf_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- rf_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- rf_conf_matrix[["overall"]][["Kappa"]]
print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
rf_pred_train <- predict(rf_model, X_train_lasso_selected)
rf_accuracy_train <- mean(rf_pred_train == y_train)
rf_conf_matrix_train <- table(Predicted = rf_pred_train, Actual = y_train)

rf_probs_train <- predict(rf_model, X_train_lasso_selected, type = "prob")
rf_roc_train <- roc(y_train, rf_probs_train[, 2])
auc_train <- auc(rf_roc_train)
auc_ci_train <- ci.auc(rf_roc_train)
rf_conf_matrix_train <- confusionMatrix(as.factor(rf_pred_train), as.factor(y_train))
tn_train <- rf_conf_matrix_train$table[1, 1]
fn_train <- rf_conf_matrix_train$table[1, 2]
fp_train <- rf_conf_matrix_train$table[2, 1]
tp_train <- rf_conf_matrix_train$table[2, 2]
# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






#XGBoost-
set.seed(42)
y_train <- as.numeric(as.character(y_train))
xgb_model <- xgboost(
  data = X_train_lasso_selected, 
  label = y_train,
  nrounds = 100,           
  objective = "binary:logistic", 
  eval_metric = "logloss", 
  eta = 0.05,           
  max_depth = 6, 
  min_child_weight = 1,  
  subsample = 0.8, 
  colsample_bytree = 0.7, 
  gamma = 1          
)




xgb_pred <- predict(xgb_model, as.matrix(X_test_lasso_selected))
xgb_roc <- roc(y_test, xgb_pred)
auc <- auc(xgb_roc)
auc_ci <- ci.auc(xgb_roc)

xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)
xgb_accuracy <- mean(xgb_pred == y_test)
xgb_conf_matrix <- table(Predicted = xgb_pred, Actual = y_test)
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred), as.factor(y_test))


tn <- xgb_conf_matrix$table[1, 1]
fn <- xgb_conf_matrix$table[1, 2]
fp <- xgb_conf_matrix$table[2, 1]
tp <- xgb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)





NIR <- xgb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- xgb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- xgb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)





# predicting on the training set
xgb_pred_train <- predict(xgb_model, as.matrix(X_train_lasso_selected))
xgb_roc_train <- roc(y_train, xgb_pred_train)
auc_train <- auc(xgb_roc_train)
auc_ci_train <- ci.auc(xgb_roc_train)
xgb_pred_train <- ifelse(xgb_pred_train > 0.5, 1, 0)
xgb_accuracy_train <- mean(xgb_pred_train == y_train)
xgb_conf_matrix_train <- table(Predicted = xgb_pred_train, Actual = y_train)
xgb_conf_matrix_train <- confusionMatrix(as.factor(xgb_pred_train), as.factor(y_train))

tn_train <- xgb_conf_matrix_train$table[1, 1]
fn_train <- xgb_conf_matrix_train$table[1, 2]
fp_train <- xgb_conf_matrix_train$table[2, 1]
tp_train <- xgb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)















## Naive Bayes
set.seed(42)
nb_model <- naiveBayes(X_train_lasso_selected, as.factor(y_train))
nb_pred <- predict(nb_model, X_test_lasso_selected)
nb_accuracy <- mean(nb_pred == y_test)
nb_conf_matrix <- table(Predicted = nb_pred, Actual = y_test)
nb_probs <- predict(nb_model, X_test_lasso_selected, type = "raw")
nb_roc <- roc(y_test, nb_probs[, 2])
auc <- auc(nb_roc)
auc_ci <- ci.auc(nb_roc)
nb_conf_matrix <- confusionMatrix(as.factor(nb_pred), as.factor(y_test))


tn <- nb_conf_matrix$table[1, 1]
fn <- nb_conf_matrix$table[1, 2]
fp <- nb_conf_matrix$table[2, 1]
tp <- nb_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(nb_roc)
print(auc_ci)


NIR <- nb_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- nb_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- nb_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)






nb_pred_train <- predict(nb_model, X_train_lasso_selected)
nb_accuracy_train <- mean(nb_pred_train == y_train)
nb_conf_matrix_train <- table(Predicted = nb_pred_train, Actual = y_train)

nb_probs_train <- predict(nb_model, X_train_lasso_selected, type = "raw")
nb_roc_train <- roc(y_train, nb_probs_train[, 2])
auc_train <- auc(nb_roc_train)
auc_ci_train <- ci.auc(nb_roc_train)


nb_conf_matrix_train <- confusionMatrix(as.factor(nb_pred_train), as.factor(y_train))

tn_train <- nb_conf_matrix_train$table[1, 1]
fn_train <- nb_conf_matrix_train$table[1, 2]
fp_train <- nb_conf_matrix_train$table[2, 1]
tp_train <- nb_conf_matrix_train$table[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(nb_roc_train)
print(auc_ci_train)







## SVM 
set.seed(42)
# Tune the parameters and specify the linear kernel function
tune_result <- tune(
  svm, 
  train.x = X_train_lasso_selected, 
  train.y = as.factor(y_train),
  ranges = list(cost = 10^(-3:2)), 
  kernel = "linear",       
  probability = TRUE
)

# Acquiring the optimal model
svm_model <- tune_result$best.model

svm_pred <- predict(svm_model, X_test_lasso_selected, probability = TRUE)
svm_pred_labels <- as.numeric(svm_pred) - 1  
svm_accuracy <- mean(svm_pred_labels == y_test)
svm_conf_matrix <- table(Predicted = svm_pred_labels, Actual = y_test)
svm_probs <- attr(svm_pred, "probabilities")[,2]
svm_roc <- roc(y_test, svm_probs)
auc <- auc(svm_roc)
auc_ci <- ci.auc(svm_roc)


svm_conf_matrix <- confusionMatrix(as.factor(svm_pred), as.factor(y_test))


tn <- svm_conf_matrix$table[1, 1]
fn <- svm_conf_matrix$table[1, 2]
fp <- svm_conf_matrix$table[2, 1]
tp <- svm_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print(auc)
print(auc_ci)



NIR <- svm_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- svm_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- svm_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)








# prediction on training set
svm_pred_train <- predict(svm_model, X_train_lasso_selected, probability = TRUE)
svm_pred_labels_train <- as.numeric(svm_pred_train) - 1  
# Accuracy on training set
svm_accuracy_train <- mean(svm_pred_labels_train == y_train)

svm_conf_matrix_train <- table(Predicted = svm_pred_labels_train, Actual = y_train)
svm_probs_train <- attr(svm_pred_train, "probabilities")[, 2]
svm_roc_train <- roc(y_train, svm_probs_train)
auc_train <- auc(svm_roc_train)
auc_ci_train <- ci.auc(svm_roc_train)

tn_train <- svm_conf_matrix_train[1, 1]
fn_train <- svm_conf_matrix_train[1, 2]
fp_train <- svm_conf_matrix_train[2, 1]
tp_train <- svm_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)






## KNN
set.seed(42)
knn_model <- train(train_data[, -1], as.factor(train_data[, 1]), method = "knn", trControl = trainControl(method = "cv", number = 5))
knn_pred <- predict(knn_model, as.data.frame(X_test_lasso_selected))
knn_accuracy <- mean(knn_pred == y_test)
knn_conf_matrix <- table(Predicted = knn_pred, Actual = y_test)
knn_probs <- predict(knn_model, X_test_lasso_selected, type = "prob")
knn_roc <- roc(y_test, knn_probs[, 2])
auc <- auc(knn_roc)
auc_ci <- ci.auc(knn_roc)

knn_conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(y_test))


tn <- knn_conf_matrix$table[1, 1]
fn <- knn_conf_matrix$table[1, 2]
fp <- knn_conf_matrix$table[2, 1]
tp <- knn_conf_matrix$table[2, 2]

# calculate the index
accuracy <- (tp + tn) / (tp + tn + fp + fn)
accuracy_ci <- binom.test(tp + tn, tp + tn + fp + fn)$conf.int

sensitivity <- tp / (tp + fn)
sensitivity_ci <- binom.test(tp, tp + fn)$conf.int

specificity <- tn / (tn + fp)
specificity_ci <- binom.test(tn, tn + fp)$conf.int
# print result
cat("Model Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("95% CI for Accuracy:", accuracy_ci[1], "-", accuracy_ci[2], "\n")
cat("Sensitivity:", sensitivity, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci[1], "-", sensitivity_ci[2], "\n")
cat("Specificity:", specificity, "\n")
cat("95% CI for Specificity:", specificity_ci[1], "-", specificity_ci[2], "\n")
print( auc)
print(auc_ci)








NIR <- knn_conf_matrix[["overall"]][["AccuracyNull"]]
P_Value_NIR <- knn_conf_matrix[["overall"]][["AccuracyPValue"]]
Kappa <- knn_conf_matrix[["overall"]][["Kappa"]]

print(NIR)
print(P_Value_NIR)
print(Kappa)









# prediction on training set
knn_pred_train <- predict(knn_model, as.data.frame(X_train_lasso_selected))
# Accuracy on training set
knn_accuracy_train <- mean(knn_pred_train == y_train)
knn_conf_matrix_train <- table(Predicted = knn_pred_train, Actual = y_train)
knn_probs_train <- predict(knn_model, X_train_lasso_selected, type = "prob")
knn_roc_train <- roc(y_train, knn_probs_train[, 2])
auc_train <- auc(knn_roc_train)
auc_ci_train <- ci.auc(knn_roc_train)

tn_train <- knn_conf_matrix_train[1, 1]
fn_train <- knn_conf_matrix_train[1, 2]
fp_train <- knn_conf_matrix_train[2, 1]
tp_train <- knn_conf_matrix_train[2, 2]

# Evaluation index on Training set
accuracy_train <- (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
accuracy_ci_train <- binom.test(tp_train + tn_train, tp_train + tn_train + fp_train + fn_train)$conf.int

sensitivity_train <- tp_train / (tp_train + fn_train)
sensitivity_ci_train <- binom.test(tp_train, tp_train + fn_train)$conf.int

specificity_train <- tn_train / (tn_train + fp_train)
specificity_ci_train <- binom.test(tn_train, tn_train + fp_train)$conf.int

# Printing the result of Training set
cat("KNN Model Training Set Evaluation Metrics:\n")
cat("Accuracy:", accuracy_train, "\n")
cat("95% CI for Accuracy:", accuracy_ci_train[1], "-", accuracy_ci_train[2], "\n")
cat("Sensitivity:", sensitivity_train, "\n")
cat("95% CI for Sensitivity:", sensitivity_ci_train[1], "-", sensitivity_ci_train[2], "\n")
cat("Specificity:", specificity_train, "\n")
cat("95% CI for Specificity:", specificity_ci_train[1], "-", specificity_ci_train[2], "\n")
print(auc_train)
print(auc_ci_train)




