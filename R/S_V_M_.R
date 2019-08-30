install.packages("e1071")
library(DescTools)
library(e1071)
setwd("C:/Users/RAMAN/Documents/R")
TrainRaw = read.csv("R_Module_Day_7.2_Credit_Risk_Train_data.csv",na.strings = "") 

TestRaw = read.csv("R_Module_Day_8.2_Credit_Risk_Test_data.csv", na.strings = "")
TrainRaw$Source = "Train"
TestRaw$Source = "Test"

# Combine both datasets. Call it FullRaw
FullRaw = rbind(TrainRaw,TestRaw)

# Verify the results
dim(FullRaw)
colSums(is.na(FullRaw))
str(FullRaw)
levels(FullRaw$Dependents)

# Convert "Dependents" categorical indep var into a continuous variable

FullRaw$Dependents <- ifelse(FullRaw$Dependents == "3+", 3, c(0,1,2))
str(FullRaw)

# Perform missing value imputation for categorical variables by imputing the NAs with a new category called "Unknown"

# Gender
FullRaw$Gender <- factor(FullRaw$Gender, levels = c(levels(FullRaw$Gender), "Unknown"))
FullRaw$Gender[is.na(FullRaw$Gender)] = "Unknown"

# Married
FullRaw$Married <- as.character(FullRaw$Married)
FullRaw$Married[is.na(FullRaw$Married)] <- "Unknown"
FullRaw$Married <- as.factor(FullRaw$Married)

# Self-Employed
FullRaw$Self_Employed <- as.character(FullRaw$Self_Employed)
FullRaw$Self_Employed[is.na(FullRaw$Self_Employed)] <- "Unknown"
FullRaw$Self_Employed <- as.factor(FullRaw$Self_Employed)

str(FullRaw)
colSums(is.na(FullRaw))

# Perform missing value imputation for continuous variables
# Loan Amount
Mymedian <- median(FullRaw$LoanAmount[FullRaw$Source == "Train"], na.rm = TRUE)
FullRaw$LoanAmount[is.na(FullRaw$LoanAmount)] <- Mymedian

# Loan_Amount_Term
Mymedian <- median(FullRaw$Loan_Amount_Term[FullRaw$Source == "Train"], na.rm = TRUE)
FullRaw$Loan_Amount_Term[is.na(FullRaw$Loan_Amount_Term)] <- Mymedian

# Credit_History
Mymedian <- median(FullRaw$Credit_History[FullRaw$Source == "Train"], na.rm = TRUE)
FullRaw$Credit_History[is.na(FullRaw$Credit_History)] <- Mymedian

# Dependents
Mymedian <- median(FullRaw$Dependents[FullRaw$Source == "Train"], na.rm = TRUE)
FullRaw$Dependents[is.na(FullRaw$Dependents)] <- Mymedian
str(FullRaw)

Dummy_df = model.matrix(~Gender+Married+Education+Self_Employed+Property_Area, data = FullRaw)
View(Dummy_Df)
Dummy_Df = Dummy_df[,-c(1)]

FullRaw <- subset(FullRaw, select = -c(Gender, Married, Education, Self_Employed, Property_Area, Loan_ID))

Data = cbind(FullRaw, Dummy_Df)
 dim(Data)
 Data$Loan_Status = ifelse(Data$Loan_Status == "N", 1, 0)
str(Data)

TrainData <- subset(Data, Source == "Train")
TrainData <- subset(TrainData, select = -c(Source))

TestData <- subset(Data, Source == "Test")
TestData <- subset(TestData, select = -c(Source))
dim(TestData) ; dim(TrainData)

# SVM Modeling

# Model
M1 <- svm(as.factor(Loan_Status) ~ ., kernel = "linear", data = TrainData)

# Prediction on TestSet
M1_Test_Pred <- predict(M1, TestData)
vif(M1_Test_Pred)

# Confusion Matrix
table(M1_Test_Pred, TestData$Loan_Status)

# Tune svm on hyperparameters
obj <- tune.svm(as.factor(Loan_Status)~., data = TestData, gamma = c(0, 0.3, 0.6, 0.9), cost = c(1, 10, 100) )

# grid_search
kernel <- c("radial", "polynomial")
gamma <- c(0, 0.3, 0.6, 0.9, 1)
cost <- c(1, 10, 100)
mentalTable <- c()
for (i in kernel) {
  for (j in gamma) {
    for (k in cost) {
      M1 <- svm(as.factor(Loan_Status) ~ ., kernel = i, gamma = j, cost = k, data = TrainData)
      M1_Pred_Test <- predict(M1, TestData)
      confusion_matrix <- table(M1_Pred_Test, TestData$Loan_Status)
      accuracy <- sum(diag(confusion_matrix))/nrow(TestData)
      mentalTable <- rbind(mentalTable, c(kernel = i,gamma= j,cost=  k, accuracy =  accuracy))
    }
  }
}
View(mentalTable)
max(mentalTable$accuracy)
