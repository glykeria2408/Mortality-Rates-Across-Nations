###--------------------- LIBRARIES ---------------------###
# Load necessary libraries
library(dplyr)
library(fastDummies)
library(ggplot2)
library(caret)
library(caTools)
library(randomForest)
library(smotefamily)
library(factoextra)  # For PCA visualization

###--------------------- DATA LOADING ---------------------###
# Load the dataset
data <- read.csv("C:/Users/Glykeria/Downloads/archive (1)/Global Health Statistics.csv")

###--------------------- DATA EXPLORATION ---------------------###
# Step 1: Inspect the Dataset
str(data)
summary(data)

# Step 2: Check for Missing Values
missing_values <- colSums(is.na(data))
cat("Missing values per column:\n")
print(missing_values)

# Step 3: Distribution of Target Variable (`Mortality.Rate....`)
ggplot(data, aes(x = Mortality.Rate....)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  theme_minimal() +
  ggtitle("Distribution of Target Variable (Mortality Rate)") +
  xlab("Mortality Rate") +
  ylab("Frequency")

###--------------------- DATA PREPROCESSING ---------------------###

# Step 4: Handle Missing Values
# Impute missing values with median for numeric variables
numeric_columns <- sapply(data, is.numeric)
data[, numeric_columns] <- lapply(data[, numeric_columns], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# For categorical variables, impute with the mode
categorical_columns <- sapply(data, function(x) is.character(x) | is.factor(x))
data[, categorical_columns] <- lapply(data[, categorical_columns], function(x) {
  ifelse(is.na(x), as.character(names(sort(table(x), decreasing = TRUE))[1]), x)
})

# Step 5: Convert Categorical Variables to Factors
data[, categorical_columns] <- lapply(data[, categorical_columns], as.factor)

# Step 6: Dummy Encoding for Categorical Variables
data <- dummy_cols(data, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Step 7: Scale Numeric Variables
data[, numeric_columns] <- lapply(data[, numeric_columns], function(x) scale(x, center = TRUE, scale = TRUE))

# Step 8: Handle Outliers
cap_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}
data[, numeric_columns] <- lapply(data[, numeric_columns], cap_outliers)

# Step 9: Remove Constant and Duplicate Columns
constant_columns <- sapply(data, function(x) sd(x, na.rm = TRUE) == 0)
data <- data[, !constant_columns]

data <- data[, !duplicated(as.list(data))]

cat("Constant and duplicate columns removed.\n")

###--------------------------Correlation Heatmap--------------------------------
# Load additional library for heatmap
library(reshape2)

# Step 10: Correlation Heatmap
# Select numeric columns
numeric_data <- data[, numeric_columns]

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Melt the correlation matrix for ggplot2
melted_cor_matrix <- melt(cor_matrix)

# Plot the correlation heatmap
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "yellow", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  ggtitle("Correlation Heatmap") +
  xlab("") +
  ylab("")



###--------------------- DIMENSIONALITY REDUCTION USING PCA ---------------------###

# Step 10: Perform PCA
numeric_data <- data %>% select_if(is.numeric)  # Select numeric features
numeric_data <- numeric_data[, !colnames(numeric_data) %in% c("Mortality.Rate....")]  # Exclude target variable

pca <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

# Calculate cumulative variance explained
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2)) * 100

# Determine number of components explaining at least 70% variance
threshold <- 70
num_components <- which(explained_variance >= threshold)[1]
cat("Number of components explaining at least", threshold, "% variance:", num_components, "\n")


# Transform the dataset using selected components
pca_data <- as.data.frame(pca$x[, 1:num_components])

# Add the target variable back to the PCA-transformed dataset
pca_data$Mortality_Class <- as.factor(ifelse(data$Mortality.Rate.... > median(data$Mortality.Rate....), "High", "Low"))  # Binary classification example

# Select numeric columns from the dataset
numeric_data <- data %>% select_if(is.numeric)

# Exclude target variable (replace 'Mortality.Rate....' with your actual target variable name, if different)
numeric_data <- numeric_data[, !colnames(numeric_data) %in% c("Mortality.Rate....")]
# Perform PCA
pca <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

# Calculate cumulative variance explained
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2)) * 100
# Create a data frame for explained variance
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2)) * 100
scree_data <- data.frame(
  Component = 1:length(explained_variance),
  Variance_Explained = explained_variance
)

# Plot Scree Plot with ggplot2
library(ggplot2)
ggplot(scree_data, aes(x = Component, y = Variance_Explained)) +
  geom_line(color = "steelblue", size = 1) +                # Line plot
  geom_point(size = 3, color = "darkblue") +               # Points
  geom_hline(yintercept = 70, linetype = "dashed", color = "red", size = 1) +  # Threshold line
  ggtitle("Scree Plot: Cumulative Variance Explained") +
  xlab("Number of Principal Components") +
  ylab("Cumulative Variance (%)") +
  theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

###--------------------- OUTPUT ---------------------###
cat("Preprocessing complete. PCA-transformed data is ready for modeling.\n")


cat("Dimensions of PCA-transformed dataset:", dim(pca_data), "\n")

###--------------------- RANDOM FOREST FOR CLASSIFICATION ---------------------###

# Step 11: Split Dataset into Training and Testing Sets
set.seed(123)  # For reproducibility
split <- sample.split(pca_data$Mortality_Class, SplitRatio = 0.7)

train_data <- subset(pca_data, split == TRUE)
test_data <- subset(pca_data, split == FALSE)

# Step 12: Train Random Forest Classification Model
rf_model <- randomForest(
  Mortality_Class ~ .,
  data = train_data,
  ntree = 100,  # Number of trees
  mtry = floor(sqrt(ncol(train_data) - 1)),  # Number of predictors to consider at each split
  importance = TRUE
)


# Print Model Summary
print(rf_model)

###--------------------- MODEL EVALUATION ---------------------###

# Predict on the test set
predictions <- predict(rf_model, newdata = test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test_data$Mortality_Class)
print(conf_matrix)

# Feature Importance
importance <- importance(rf_model)
png("Feature_Importance_Plot.png", width = 1200, height = 800)
barplot(importance_df$MeanDecreaseGini, 
        names.arg = rownames(importance_df), 
        las = 2, 
        main = "Feature Importance", 
        col = "blue")
dev.off()


###--------------------- RANDOM FOREST TUNING ---------------------###

# Load necessary libraries for tuning
library(caret)

# Define a grid of hyperparameters to tune
tune_grid <- expand.grid(
  mtry = seq(2, floor(sqrt(ncol(train_data) - 1)), by = 1)  # Number of predictors at each split
)

# Set up cross-validation
control <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5-fold CV
  classProbs = TRUE,       # Enable class probabilities
  summaryFunction = twoClassSummary  # Use summary function for binary classification
)

# Train the model with grid search
set.seed(123)
rf_tuned <- train(
  Mortality_Class ~ .,
  data = train_data,
  method = "rf",
  metric = "ROC",         # Use ROC as the evaluation metric
  tuneGrid = tune_grid,
  trControl = control,
  ntree = 200             # Use a larger number of trees for stability
)

# Print the best tuning parameters
cat("Best parameters:\n")
print(rf_tuned$bestTune)

# Step 12: Evaluate Tuned Model
# Predict on the test set
predictions <- predict(rf_tuned, newdata = test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test_data$Mortality_Class)
print(conf_matrix)

# Step 13: Feature Importance
importance <- varImp(rf_tuned)
plot(importance, top = 10, main = "Top 10 Feature Importance")

###--------------------- ENHANCED RANDOM FOREST TUNING ---------------------###

# Load necessary libraries
library(caret)
library(randomForest)

# Define a broader grid of hyperparameters to tune
tune_grid <- expand.grid(
  mtry = seq(2, floor(sqrt(ncol(train_data) - 1)), by = 1),  # Number of predictors at each split
  nodesize = c(1, 5, 10, 15),                               # Minimum size of terminal nodes
  maxnodes = c(10, 20, 30, 50, 100)                         # Maximum number of terminal nodes
)

# Create a custom training function for Random Forest
custom_rf <- function(train_data, test_data, tune_grid) {
  # Placeholder to store results
  results <- data.frame()
  
  # Loop through hyperparameters
  for (i in 1:nrow(tune_grid)) {
    # Extract parameters
    params <- tune_grid[i, ]
    
    # Train Random Forest with current parameters
    rf_model <- randomForest(
      Mortality_Class ~ .,
      data = train_data,
      mtry = params$mtry,
      nodesize = params$nodesize,
      maxnodes = params$maxnodes,
      ntree = 200,   # Set a fixed number of trees
      importance = TRUE
    )
    
    # Predict on the test set
    predictions <- predict(rf_model, newdata = test_data)
    
    # Evaluate model using accuracy
    conf_matrix <- confusionMatrix(predictions, test_data$Mortality_Class)
    accuracy <- conf_matrix$overall["Accuracy"]
    
    # Store the results
    results <- rbind(results, cbind(params, Accuracy = accuracy))
  }
  
  # Return the results sorted by accuracy
  return(results[order(-results$Accuracy), ])
}

# Perform hyperparameter tuning
set.seed(123)
tuning_results <- custom_rf(train_data, test_data, tune_grid)

# Print the best hyperparameter combination
cat("Best Hyperparameters:\n")
print(tuning_results[1, ])

# Train final model using the best hyperparameters
best_params <- tuning_results[1, ]
final_rf_model <- randomForest(
  Mortality_Class ~ .,
  data = train_data,
  mtry = best_params$mtry,
  nodesize = best_params$nodesize,
  maxnodes = best_params$maxnodes,
  ntree = 200,
  importance = TRUE
)

# Predict on the test set with the final model
final_predictions <- predict(final_rf_model, newdata = test_data)

# Confusion Matrix for the final model
final_conf_matrix <- confusionMatrix(final_predictions, test_data$Mortality_Class)
cat("\nConfusion Matrix for Final Model:\n")
print(final_conf_matrix)

# Feature Importance
importance <- importance(final_rf_model)
varImpPlot(final_rf_model, main = "Feature Importance (Final Random Forest Model)")




####------------------------------BOOSTING------------------------------------
# Install and load xgboost
if (!require("xgboost")) install.packages("xgboost")
library(xgboost)

# Convert data into matrix format required by xgboost
train_matrix <- as.matrix(train_data[, -ncol(train_data)])  # Exclude target variable
train_labels <- as.numeric(train_data$Mortality_Class) - 1  # Convert to binary 0/1 for xgboost

test_matrix <- as.matrix(test_data[, -ncol(test_data)])

test_labels <- as.numeric(test_data$Mortality_Class) - 1

# Train xgboost model
set.seed(123)
xgb_model <- xgboost(
  data = train_matrix,
  label = train_labels,
  nrounds = 100,               # Number of boosting iterations
  objective = "binary:logistic",  # Binary classification
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.3
)

# Predict on test data
xgb_predictions <- predict(xgb_model, test_matrix)
xgb_class <- ifelse(xgb_predictions > 0.5, "High", "Low")
xgb_class <- factor(xgb_class, levels = c("High", "Low"))

# Confusion Matrix for xgboost
xgb_conf_matrix <- confusionMatrix(xgb_class, test_data$Mortality_Class)
print(xgb_conf_matrix)


# Install and load required libraries
if (!require("xgboost")) install.packages("xgboost")
if (!require("caret")) install.packages("caret")
library(xgboost)
library(caret)

# Prepare data for xgboost
train_matrix <- as.matrix(train_data[, -ncol(train_data)])  # Exclude target variable
train_labels <- as.numeric(train_data$Mortality_Class) - 1  # Convert target to binary 0/1

test_matrix <- as.matrix(test_data[, -ncol(test_data)])
test_labels <- as.numeric(test_data$Mortality_Class) - 1

# Define a comprehensive grid of hyperparameters for tuning
tune_grid <- expand.grid(
  nrounds = c(50, 100, 150),            # Number of boosting iterations
  max_depth = c(3, 6, 9),               # Maximum tree depth
  eta = c(0.01, 0.1, 0.3),              # Learning rate
  gamma = c(0, 1, 5),                   # Minimum loss reduction
  colsample_bytree = c(0.6, 0.8, 1),    # Feature subsampling
  min_child_weight = c(1, 5, 10),       # Minimum sum of weights in a child
  subsample = c(0.6, 0.8, 1)            # Row subsampling
)

# TrainControl for cross-validation
control <- trainControl(
  method = "cv",                  # Cross-validation
  number = 5,                     # 5-fold CV
  classProbs = TRUE,              # Enable class probabilities
  summaryFunction = twoClassSummary,  # Use ROC for evaluation
  verboseIter = TRUE              # Show progress
)

# Train the xgboost model with the new grid
set.seed(123)
xgb_tuned <- train(
  x = train_matrix,
  y = factor(train_labels, labels = c("Low", "High")),
  method = "xgbTree",
  metric = "ROC",                  # Optimize for ROC
  tuneGrid = tune_grid,
  trControl = control
)

# Print the best tuning parameters
cat("Best parameters:\n")
print(xgb_tuned$bestTune)

# Predict on the test set using the best model
xgb_predictions <- predict(xgb_tuned, newdata = test_matrix)
xgb_conf_matrix <- confusionMatrix(xgb_predictions, factor(test_labels, labels = c("Low", "High")))
print(xgb_conf_matrix)

###-------------------------------------KNN--------------------------------------
###--------------------- KNN IMPLEMENTATION ---------------------###

# Load necessary library
if (!require("class")) install.packages("class")
library(class)

# Step 1: Scale Data for KNN
# Scale features (excluding target variable) for both training and test sets
knn_train <- scale(train_data[, -ncol(train_data)])  # Exclude target variable
knn_test <- scale(test_data[, -ncol(test_data)])

train_labels <- train_data$Mortality_Class  # Target variable for training
test_labels <- test_data$Mortality_Class    # Target variable for testing


# Step 2: Implement KNN for Multiple Values of k
# Define a range of k values to test
k_values <- seq(1, 15, by = 2)  # Test odd values of k

# Initialize storage for accuracy results
knn_results <- data.frame(k = k_values, Accuracy = NA)

for (i in 1:length(k_values)) {
  # Apply KNN with current k
  knn_predictions <- knn(
    train = knn_train,
    test = knn_test,
    cl = train_labels,
    k = k_values[i]
  )
  
  # Calculate accuracy
  accuracy <- sum(knn_predictions == test_labels) / length(test_labels)
  knn_results$Accuracy[i] <- accuracy
}

# Print accuracy results for all k
print(knn_results)

# Find the best k based on accuracy
best_k <- knn_results$k[which.max(knn_results$Accuracy)]
cat("Best k:", best_k, "\n")

# Step 3: Apply KNN with Best k
knn_best_predictions <- knn(
  train = knn_train,
  test = knn_test,
  cl = train_labels,
  k = best_k
)

# Confusion Matrix for KNN
knn_conf_matrix <- confusionMatrix(knn_best_predictions, test_labels)
print(knn_conf_matrix)
k_values <- seq(1, 30, by = 1)  # Test more values of k
knn_model <- train(
  Mortality_Class ~ .,
  data = train_data,
  method = "knn",
  tuneGrid = expand.grid(k = 1:30),
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

# Define a range of k values to test
k_values <- seq(1, 30, by = 1)

# Train the KNN model with cross-validation
set.seed(123)
knn_model <- train(
  Mortality_Class ~ .,
  data = train_data,
  method = "knn",
  tuneGrid = expand.grid(k = k_values),
  trControl = trainControl(method = "cv", number = 5),  # 5-fold CV
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

# Print the results for each k
cat("Results for each k:\n")
print(knn_model$results)

# Plot accuracy vs. k
plot(knn_model, main = "Accuracy vs. Number of Neighbors (k)")

# Extract the best k
best_k <- knn_model$bestTune$k
cat("Best k based on cross-validation:", best_k, "\n")

# Predict on the test set using the best k
knn_predictions <- predict(knn_model, newdata = test_data)

# Confusion Matrix for KNN
knn_conf_matrix <- confusionMatrix(knn_predictions, test_data$Mortality_Class)
print(knn_conf_matrix)

# Additional Performance Metrics
cat("Accuracy of the best KNN model:", knn_conf_matrix$overall["Accuracy"], "\n")
cat("Sensitivity (True Positive Rate):", knn_conf_matrix$byClass["Sensitivity"], "\n")
cat("Specificity (True Negative Rate):", knn_conf_matrix$byClass["Specificity"], "\n")
cat("Balanced Accuracy:", knn_conf_matrix$byClass["Balanced Accuracy"], "\n")

###--------------------- SVM IMPLEMENTATION ---------------------###

# Load necessary libraries
if (!require("e1071")) install.packages("e1071")
library(e1071)
library(caret)

# Ensure the target variable 'Mortality_Class' is correctly defined
pca_data$Mortality_Class <- as.factor(ifelse(data$BMI > median(data$BMI, na.rm = TRUE), "High", "Low"))  # Use 'BMI' as a proxy target
# Split the dataset into training and testing sets
set.seed(123)
split <- sample.split(pca_data$Mortality_Class, SplitRatio = 0.7)
train_data <- subset(pca_data, split == TRUE)
test_data <- subset(pca_data, split == FALSE)

# Step 1: Define the training control for cross-validation
control <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5-fold CV
  classProbs = TRUE,       # Enable class probabilities
  summaryFunction = twoClassSummary  # Use ROC as the evaluation metric
)

# Step 2: Define a grid of hyperparameters to tune
tune_grid <- expand.grid(
  C = c(0.1, 1, 10, 100),       # Regularization parameter
  sigma = c(0.01, 0.05, 0.1, 0.2)  # RBF kernel parameter
)

# Step 3: Train the SVM model using caret
set.seed(123)
svm_model <- train(
  Mortality_Class ~ .,
  data = train_data,
  method = "svmRadial",   # Radial Basis Function kernel
  tuneGrid = tune_grid,
  trControl = control,
  metric = "ROC"          # Optimize for ROC
)

# Step 4: Print the best tuning parameters
cat("Best parameters:\n")
print(svm_model$bestTune)

# Step 5: Predict on the test set using the best model
svm_predictions <- predict(svm_model, newdata = test_data)

# Confusion Matrix for SVM
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$Mortality_Class)
print(svm_conf_matrix)

# Additional Performance Metrics
cat("Accuracy of the best SVM model:", svm_conf_matrix$overall["Accuracy"], "\n")
cat("Sensitivity (True Positive Rate):", svm_conf_matrix$byClass["Sensitivity"], "\n")
cat("Specificity (True Negative Rate):", svm_conf_matrix$byClass["Specificity"], "\n")
cat("Balanced Accuracy:", svm_conf_matrix$byClass["Balanced Accuracy"], "\n")

###--------------------- SVM IMPLEMENTATION ---------------------###

# Load necessary library for SVM
library(e1071)
library(caTools)
# Perform PCA
pca <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

# Select components explaining at least 70% variance
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2)) * 100
num_components <- which(explained_variance >= 70)[1]

# Transform data with selected components
pca_data <- as.data.frame(pca$x[, 1:num_components])

# Add target variable
pca_data$Mortality_Class <- as.factor(ifelse(data$Mortality.Rate.... > median(data$Mortality.Rate....), "High", "Low"))

# Step 11: Split Dataset into Training and Testing Sets
set.seed(123)  # For reproducibility
split <- sample.split(pca_data$Mortality_Class, SplitRatio = 0.7)

train_data <- subset(pca_data, split == TRUE)
test_data <- subset(pca_data, split == FALSE)

# Step 12: Train SVM Classification Model
svm_model <- svm(
  Mortality_Class ~ .,
  data = train_data,
  type = "C-classification",  # For classification tasks
  kernel = "radial",          # Radial Basis Function kernel (default)
  cost = 1,                   # Regularization parameter
  gamma = 0.1                 # Kernel coefficient
)

# Print Model Summary
print(svm_model)

###--------------------- MODEL EVALUATION ---------------------###

# Predict on the test set
svm_predictions <- predict(svm_model, newdata = test_data)

# Confusion Matrix
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$Mortality_Class)
print(svm_conf_matrix)

# Step 13: Perform Hyperparameter Tuning for SVM

tune_result <- tune(
  svm,
  Mortality_Class ~ .,
  data = train_data,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 10, 100),
    gamma = c(0.01, 0.1, 1, 10)
  )
)

# Best Parameters
cat("Best parameters after tuning:\n")
print(tune_result$best.parameters)

# Step 14: Train SVM with Best Parameters
svm_tuned_model <- svm(
  Mortality_Class ~ .,
  data = train_data,
  type = "C-classification",
  kernel = "radial",
  cost = tune_result$best.parameters$cost,
  gamma = tune_result$best.parameters$gamma
)

# Predict on the test set with tuned model
svm_tuned_predictions <- predict(svm_tuned_model, newdata = test_data)

# Confusion Matrix for Tuned Model
svm_tuned_conf_matrix <- confusionMatrix(svm_tuned_predictions, test_data$Mortality_Class)
print(svm_tuned_conf_matrix)

# Step 15: Visualization of SVM Decision Boundary (optional for 2D)
# If applicable, plot SVM decision boundary
if (ncol(train_data) - 1 == 2) {  # Only works if there are 2 predictors
  plot(svm_tuned_model, train_data, V1 ~ V2, main = "SVM Decision Boundary")
} else {
  cat("Visualization of decision boundary is only applicable for 2D data.")
}

###--------------------- SVM IMPLEMENTATION ---------------------###

# Load necessary libraries for SVM
library(e1071)
library(caTools)
library(caret)  # For confusion matrix and cross-validation

# Perform PCA
pca <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

# Select components explaining at least 70% variance
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2)) * 100
num_components <- which(explained_variance >= 70)[1]

# Transform data with selected components
pca_data <- as.data.frame(pca$x[, 1:num_components])

# Add target variable
pca_data$Mortality_Class <- as.factor(ifelse(data$Mortality.Rate.... > median(data$Mortality.Rate....), "High", "Low"))

# Step 11: Split Dataset into Training and Testing Sets
set.seed(123)  # For reproducibility
split <- sample.split(pca_data$Mortality_Class, SplitRatio = 0.7)

train_data <- subset(pca_data, split == TRUE)
test_data <- subset(pca_data, split == FALSE)

# Step 12: Train and Evaluate SVM Models with Different Kernels
kernels <- c("linear", "polynomial", "radial", "sigmoid")
results <- list()

for (kernel in kernels) {
  cat("\nTraining SVM with", kernel, "kernel...\n")
  
  # Train SVM model
  svm_model <- svm(
    Mortality_Class ~ .,
    data = train_data,
    type = "C-classification",
    kernel = kernel,
    cost = 1,
    gamma = 0.1  # For polynomial, radial, and sigmoid kernels
  )
  
  # Predict on test set
  predictions <- predict(svm_model, newdata = test_data)
  
  # Confusion matrix
  conf_matrix <- confusionMatrix(predictions, test_data$Mortality_Class)
  results[[kernel]] <- conf_matrix
  
  cat("Confusion Matrix for", kernel, "kernel:\n")
  print(conf_matrix)
}

# Compare the Accuracy of Different Kernels
accuracy_comparison <- sapply(results, function(cm) cm$overall["Accuracy"])
cat("\nAccuracy for Different Kernels:\n")
print(accuracy_comparison)

# Step 13: Hyperparameter Tuning for the Best Kernel (Radial as Example)
tune_result <- tune(
  svm,
  Mortality_Class ~ .,
  data = train_data,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 10, 100),
    gamma = c(0.01, 0.1, 1, 10)
  )
)

# Best Parameters
cat("Best parameters for radial kernel:\n")
print(tune_result$best.parameters)

# Step 14: Train SVM with Tuned Parameters for Best Kernel
svm_tuned_model <- svm(
  Mortality_Class ~ .,
  data = train_data,
  type = "C-classification",
  kernel = "radial",
  cost = tune_result$best.parameters$cost,
  gamma = tune_result$best.parameters$gamma
)

# Predict on the test set with tuned model
tuned_predictions <- predict(svm_tuned_model, newdata = test_data)

# Confusion Matrix for Tuned Model
tuned_conf_matrix <- confusionMatrix(tuned_predictions, test_data$Mortality_Class)
cat("\nConfusion Matrix for Tuned SVM Model:\n")
print(tuned_conf_matrix)
