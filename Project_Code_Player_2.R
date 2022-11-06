
## Football angles project code for one of the players


## Importing data

mydata <- read.csv("RC1gcGrp123.csv")

## Explanatory Data Analysis (boxplots)

## Boxplots were generated to observe the variation in the data between the different outcomes

# Creating a new variable in the data that categorizes each outcome together.
# This is used for the Ordinal Logistic Regression Models in addition to the boxplots.
mydata$outcome <- 0
mydata$outcome[mydata$Outcome.y.45to90==1] <- 1
mydata$outcome[mydata$Outcome.y.90to135==1] <- 2
mydata$outcome[mydata$Outcome.y.135to180==1] <- 3


boxplot(mydata$CODangle~mydata$outcome, xlab='Outcome',ylab='COD angle')
boxplot(mydata$Mean_Speed~mydata$outcome, xlab='Outcome',ylab='Mean Speed')
boxplot(mydata$Min_AcclY~mydata$outcome, xlab='Outcome',ylab='Min Accl Y')
boxplot(mydata$Max_GyroX~mydata$outcome, xlab='Outcome',ylab='Max Gyro X')
boxplot(mydata$Mean_IAI~mydata$outcome, xlab='Outcome',ylab='Mean_IAI')
boxplot(mydata$Mean_AcclX~mydata$outcome, xlab='Outcome',ylab='Mean_AcclX')
boxplot(mydata$Mean_AcclY~mydata$outcome, xlab='Outcome',ylab='Mean_AcclY')
boxplot(mydata$Mean_AcclZ~mydata$outcome, xlab='Outcome',ylab='Mean_AcclZ')
boxplot(mydata$Mean_GyroX~mydata$outcome, xlab='Outcome',ylab='Mean_GyroX')
boxplot(mydata$Mean_GyroY~mydata$outcome, xlab='Outcome',ylab='Mean_GyroY')
boxplot(mydata$Mean_GyroZ~mydata$outcome, xlab='Outcome',ylab='Mean_GyroZ')
boxplot(mydata$Min_Speed~mydata$outcome, xlab='Outcome',ylab='Min_Speed')
boxplot(mydata$Min_IAI~mydata$outcome, xlab='Outcome',ylab='Min_IAI')
boxplot(mydata$Min_AcclX~mydata$outcome, xlab='Outcome',ylab='Min_AcclX')
boxplot(mydata$Min_AcclZ~mydata$outcome, xlab='Outcome',ylab='Min_AcclZ')
boxplot(mydata$Min_GyroX~mydata$outcome, xlab='Outcome',ylab='Min_GyroX')
boxplot(mydata$Min_GyroY~mydata$outcome, xlab='Outcome',ylab='Min_GyroY')
boxplot(mydata$Min_GyroZ~mydata$outcome, xlab='Outcome',ylab='Min_GyroZ')
boxplot(mydata$Max_Speed~mydata$outcome, xlab='Outcome',ylab='Max_Speed')
boxplot(mydata$Max_IAI~mydata$outcome, xlab='Outcome',ylab='Max_IAI')
boxplot(mydata$Max_AcclX~mydata$outcome, xlab='Outcome',ylab='Max_AcclX')
boxplot(mydata$Max_AcclY~mydata$outcome, xlab='Outcome',ylab='Max_AcclY')
boxplot(mydata$Max_AcclZ~mydata$outcome, xlab='Outcome',ylab='Max_AcclZ')
boxplot(mydata$Max_GyroY~mydata$outcome, xlab='Outcome',ylab='Max_GyroY')
boxplot(mydata$Max_GyroZ~mydata$outcome, xlab='Outcome',ylab='Max_GyroZ')

## Explanatory Data Analysis (t-tests)

## t-tests were conducted to observe all of the relationships between angles and other variables

# t-tests for angle 45 to 90

t.test(mydata$CODangle~mydata$Outcome.y.45to90)
t.test(mydata$Mean_Speed~mydata$Outcome.y.45to90)
t.test(mydata$Mean_IAI~mydata$Outcome.y.45to90)
t.test(mydata$Mean_AcclX~mydata$Outcome.y.45to90)
t.test(mydata$Mean_AcclY~mydata$Outcome.y.45to90)
t.test(mydata$Mean_AcclZ~mydata$Outcome.y.45to90)
t.test(mydata$Mean_GyroX~mydata$Outcome.y.45to90)
t.test(mydata$Mean_GyroY~mydata$Outcome.y.45to90)
t.test(mydata$Mean_GyroZ~mydata$Outcome.y.45to90)
t.test(mydata$Min_Speed~mydata$Outcome.y.45to90)
t.test(mydata$Min_IAI~mydata$Outcome.y.45to90)
t.test(mydata$Min_AcclX~mydata$Outcome.y.45to90)
t.test(mydata$Min_AcclY~mydata$Outcome.y.45to90)
t.test(mydata$Min_AcclZ~mydata$Outcome.y.45to90)
t.test(mydata$Min_GyroX~mydata$Outcome.y.45to90)
t.test(mydata$Min_GyroY~mydata$Outcome.y.45to90)
t.test(mydata$Min_GyroZ~mydata$Outcome.y.45to90)
t.test(mydata$Max_Speed~mydata$Outcome.y.45to90)
t.test(mydata$Max_IAI~mydata$Outcome.y.45to90)
t.test(mydata$Max_AcclX~mydata$Outcome.y.45to90)
t.test(mydata$Max_AcclY~mydata$Outcome.y.45to90)
t.test(mydata$Max_AcclZ~mydata$Outcome.y.45to90)
t.test(mydata$Max_GyroX~mydata$Outcome.y.45to90)
t.test(mydata$Max_GyroY~mydata$Outcome.y.45to90)
t.test(mydata$Max_GyroZ~mydata$Outcome.y.45to90)


# t-tests for angle 90 to 135

t.test(mydata$CODangle~mydata$Outcome.y.90to135)
t.test(mydata$Mean_Speed~mydata$Outcome.y.90to135)
t.test(mydata$Mean_IAI~mydata$Outcome.y.90to135)
t.test(mydata$Mean_AcclX~mydata$Outcome.y.90to135)
t.test(mydata$Mean_AcclY~mydata$Outcome.y.90to135)
t.test(mydata$Mean_AcclZ~mydata$Outcome.y.90to135)
t.test(mydata$Mean_GyroX~mydata$Outcome.y.90to135)
t.test(mydata$Mean_GyroY~mydata$Outcome.y.90to135)
t.test(mydata$Mean_GyroZ~mydata$Outcome.y.90to135)
t.test(mydata$Min_Speed~mydata$Outcome.y.90to135)
t.test(mydata$Min_IAI~mydata$Outcome.y.90to135)
t.test(mydata$Min_AcclX~mydata$Outcome.y.90to135)
t.test(mydata$Min_AcclY~mydata$Outcome.y.90to135)
t.test(mydata$Min_AcclZ~mydata$Outcome.y.90to135)
t.test(mydata$Min_GyroX~mydata$Outcome.y.90to135)
t.test(mydata$Min_GyroY~mydata$Outcome.y.90to135)
t.test(mydata$Min_GyroZ~mydata$Outcome.y.90to135)
t.test(mydata$Max_Speed~mydata$Outcome.y.90to135)
t.test(mydata$Max_IAI~mydata$Outcome.y.90to135)
t.test(mydata$Max_AcclX~mydata$Outcome.y.90to135)
t.test(mydata$Max_AcclY~mydata$Outcome.y.90to135)
t.test(mydata$Max_AcclZ~mydata$Outcome.y.90to135)
t.test(mydata$Max_GyroX~mydata$Outcome.y.90to135)
t.test(mydata$Max_GyroY~mydata$Outcome.y.90to135)
t.test(mydata$Max_GyroZ~mydata$Outcome.y.90to135)


# t-tests for angle 135 to 180

t.test(mydata$CODangle~mydata$Outcome.y.135to180)
t.test(mydata$Mean_Speed~mydata$Outcome.y.135to180)
t.test(mydata$Mean_IAI~mydata$Outcome.y.135to180)
t.test(mydata$Mean_AcclX~mydata$Outcome.y.135to180)
t.test(mydata$Mean_AcclY~mydata$Outcome.y.135to180)
t.test(mydata$Mean_AcclZ~mydata$Outcome.y.135to180)
t.test(mydata$Mean_GyroX~mydata$Outcome.y.135to180)
t.test(mydata$Mean_GyroY~mydata$Outcome.y.135to180)
t.test(mydata$Mean_GyroZ~mydata$Outcome.y.135to180)
t.test(mydata$Min_Speed~mydata$Outcome.y.135to180)
t.test(mydata$Min_IAI~mydata$Outcome.y.135to180)
t.test(mydata$Min_AcclX~mydata$Outcome.y.135to180)
t.test(mydata$Min_AcclY~mydata$Outcome.y.135to180)
t.test(mydata$Min_AcclZ~mydata$Outcome.y.135to180)
t.test(mydata$Min_GyroX~mydata$Outcome.y.135to180)
t.test(mydata$Min_GyroY~mydata$Outcome.y.135to180)
t.test(mydata$Min_GyroZ~mydata$Outcome.y.135to180)
t.test(mydata$Max_Speed~mydata$Outcome.y.135to180)
t.test(mydata$Max_IAI~mydata$Outcome.y.135to180)
t.test(mydata$Max_AcclX~mydata$Outcome.y.135to180)
t.test(mydata$Max_AcclY~mydata$Outcome.y.135to180)
t.test(mydata$Max_AcclZ~mydata$Outcome.y.135to180)
t.test(mydata$Max_GyroX~mydata$Outcome.y.135to180)
t.test(mydata$Max_GyroY~mydata$Outcome.y.135to180)
t.test(mydata$Max_GyroZ~mydata$Outcome.y.135to180)


## Libraries required for ROC curves & confusion matrices

library(pROC)
library(caret)
library(e1071)


## Training and test data (70% & 30% split)

train <- mydata[1:767,]


test <- mydata[768:1095,]


## Binary Logistic Regression


#45 to 90 angle

# Removing time and ID variables from training data. 
# Also removing other angle outcomes except for 45-90.
mypredictors <- subset(train,select=c(4:28,30))
head(mypredictors)
 
# Changing angle outcome to a categorical variable
mypredictors$Outcome.y.45to90<-as.factor(mypredictors$Outcome.y.45to90)
str(mypredictors)

# Generating a Binary Logistic Regression (BLR) model trained on the training data.
myglm <- glm(Outcome.y.45to90 ~ .,data=mypredictors, family = "binomial")
summary(myglm)

# Performing backwards stepwise on the model to improve accuracy.
backwards = step(myglm)
summary(backwards)

# Removing time, ID and other angle variables from test data for 45 to 90.
subdata <- subset(test,select=c(4:28,30))

# Changing angle outcome to a categorical variable
subdata$Outcome.y.45to90 <- as.factor(subdata$Outcome.y.45to90)
str(subdata)

# Applying the BLR model to the test data to create a set of predictions for the angle 45 to 90.
mypredict <- predict(backwards, subdata, type="response")

# Generating and plotting a Reciever Operating Characteristic (ROC) curve of the predictors.
myroc <- roc(subdata$Outcome.y.45to90, mypredict)
plot(myroc,main='Outcome 45 to 90')


# Confusion matrix

# Splitting the predictors between outcome results 1 and 0 based on the threshold.
# In this case the threshold is 0.05.
mypredict[mypredict>0.05] <- 1
mypredict[mypredict<0.05] <- 0

# Setting the predictor and outcome to categorical variables.
X <- as.factor(mypredict)
Y <- as.factor(test$Outcome.y.45to90)

# Generating a confusion matrix of the outcomes vs predictions.
Con <- confusionMatrix(X,Y,positive="1")
Con


#90 to 135 angle

# Creating another subset removing time and ID variables from training data. 
# Also removing other angle outcomes except for 90-135.
mypredictors2 <- subset(train,select=c(4:28,31))
head(mypredictors2)

# Changing angle outcome to a categorical variable.
mypredictors2$Outcome.y.90to135<-as.factor(mypredictors2$Outcome.y.90to135)
str(mypredictors2)

# Generating a Binary Logistic Regression (BLR) model trained on the training data.
myglm2 <- glm(Outcome.y.90to135 ~ .,data=mypredictors2, family = "binomial")
summary(myglm2)

# Performing backwards stepwise on the model to improve accuracy.
backwards2 = step(myglm2)
summary(backwards2)

# Removing time, ID and other angle variables from test data for 90-135.
subdata2 <- subset(test,select=c(4:28,31))

# Setting outcome to categorical variable.
subdata2$Outcome.y.90to135 <- as.factor(subdata2$Outcome.y.90to135)
str(subdata2)

# Applying the BLR model to the 90-135 test data. 
mypredict2 <- predict(backwards2, subdata2, type="response")

# Plotting an ROC curve of the predictors.
myroc2 <- roc(subdata2$Outcome.y.90to135, mypredict2)
plot(myroc2,main='Outcome 90 to 135')


# Confusion matrix

# Splitting the outcome by the threshold 0.005
mypredict2[mypredict2>0.005] <- 1
mypredict2[mypredict2<0.005] <- 0

# Setting predictor and outcome to categorical variables.
X2 <- as.factor(mypredict2)
Y2 <- as.factor(test$Outcome.y.90to135)

# Generating a confusion matrix of outcomes vs predictions.
Con2 <- confusionMatrix(X2,Y2,positive="1")
Con2


#135 to 180 angle

# Creating subset of training data for 135-180 angle, with time, ID & other angles removed.
mypredictors3 <- subset(train,select=c(4:28,32))
head(mypredictors3)

# Changing outcome to categorical variable.
mypredictors3$Outcome.y.135to180<-as.factor(mypredictors3$Outcome.y.135to180)
str(mypredictors3)

# Training BLR model on training data.
myglm3 <- glm(Outcome.y.135to180 ~ .,data=mypredictors3, family = "binomial")
summary(myglm3)

# Performing backwards stepwise.
backwards3 = step(myglm3)
summary(backwards3)

# Subset of test data for 135-180 angle.
subdata3 <- subset(test,select=c(4:28,32))

# Setting outcome to categorical variable.
subdata3$Outcome.y.135to180 <- as.factor(subdata3$Outcome.y.135to180)
str(subdata3)

# Running BLR model on test data.
mypredict3 <- predict(backwards3, subdata3, type="response")

# Generating ROC curve.
myroc3 <- roc(subdata3$Outcome.y.135to180,mypredict3)
plot(myroc3,main='Outcome 135 to 180')

# Confusion matrix

# Splitting predictions by threshold 0.015.
mypredict3[mypredict3>0.015] <- 1
mypredict3[mypredict3<0.015] <- 0

# Setting predictor and outcome to categorical variables.
X3 <- as.factor(mypredict3)
Y3 <- as.factor(test$Outcome.y.135to180)

# Creating confusion matrix of outcome vs prediction.
Con3 <- confusionMatrix(X3,Y3,positive="1")
Con3




## Ordinal Logistic Regression (OLR)


table(train$outcome)

library(MASS)

# Generating an OLR model using the training data with multiple outcome variable.
mymodel <- polr(as.factor(outcome) ~ CODangle + Mean_Speed + Mean_IAI
                + Mean_AcclX + Mean_AcclY + Mean_AcclZ
                + Mean_GyroX + Mean_GyroY + Mean_GyroZ + Min_Speed 
                + Min_IAI + Min_AcclX + Min_AcclY + Min_AcclZ + Min_GyroX
                + Min_GyroY + Min_GyroZ + Max_Speed + Max_IAI + Max_AcclX
                + Max_AcclY + Max_AcclZ + Max_GyroX + Max_GyroY + Max_GyroZ
                , data = train, Hess=TRUE)
mymodel
summary(mymodel)

# Backwards stepwise of the model.
backwards4 <- step(mymodel)

# Creating a table of the coefficients.
table = coef(summary(backwards4))

# Using the t-values in the table, the p-values for the variables are calculated.
p = pnorm(abs(table[,"t value"]), lower.tail = FALSE)*2

# The p-values are added to the table.
(table <- cbind(table, "p value" = p))

# Setting outcome in test data to categorical.
test$outcome <- as.factor(test$outcome)
str(test$outcome)

# Applying the OLR model to the test data.
pred <- predict(backwards4, test, type="prob")


# Multi-class ROC was generated but, unfortunately, could not be plotted.
myroc5 <- multiclass.roc(test$outcome, pred)

# Because of this, separate ROC curves were generated and plotted for each angle outcome.

# ROC for 45-90.
B <- roc(test$Outcome.y.45to90 ~ pred[,2])

# ROC for 90-135.
B1 <- roc(test$Outcome.y.90to135 ~ pred[,3])

# ROC for 135-180.
B2 <- roc(test$Outcome.y.135to180 ~ pred[,4])


plot(B, main='Outcome 45 to 90')
plot(B1, main='Outcome 90 to 135')
plot(B2, main='Outcome 135 to 180')


# The predicted probabilities are separated by variables from the predictor.

# 45-90 angle
P <- pred[,2]

# 90-135 angle
P1 <- pred[,3]

# 135-180 angle.
P2 <- pred[,4]


# Confusion matrices were formed, using the same principle as those created for BLR above.

# Predictor split by threshold 0.06
P[P>0.06] <- 1
P[P<0.06] <- 0

# Setting predictor and outcome as categorical variables.
X7 <- as.factor(P)
Y7 <- as.factor(test$Outcome.y.45to90)

# Building confusion matrix
Con7 <- confusionMatrix(X7,Y7,positive="1")
Con7



# Predictor split by threshold 0.01
P1[P1>0.01] <- 1
P1[P1<0.01] <- 0

# Setting predictor & outcome to categorical variables.
X8 <- as.factor(P1)
Y8 <- as.factor(test$Outcome.y.90to135)

# Building confusion matrix.
Con8 <- confusionMatrix(X8,Y8,positive="1")
Con8



# Predictor split by threshold 0.03
P2[P2>0.03] <- 1
P2[P2<0.03] <- 0

# Setting predictor & outcome to categorical variables.
X9 <- as.factor(P2)
Y9 <- as.factor(test$Outcome.y.135to180)

# Building confusion matrix.
Con9 <- confusionMatrix(X9,Y9,positive="1")
Con9



## The logistic regression models were generated again, 
# but using all of the variables in the training data (No backwards stepwise).


## Binary Logistic Regression using ALL variables.


# 45-90 angle.

# Applying the original BLR model (before backwards stepwise) to the test data.
# Then plotting its respective ROC curve.
mypred <- predict(myglm, subdata, type="response")
myproc <- roc(subdata$Outcome.y.45to90, mypred)
plot(myproc,main='Outcome 45 to 90')

# Setting the threshold to 0.06
mypred[mypred>0.06] <- 1
mypred[mypred<0.06] <- 0

# Setting outcome & predictor variables to categorical.
X4 <- as.factor(mypred)
Y4 <- as.factor(test$Outcome.y.45to90)

# Generating the confusion matrix.
pCon <- confusionMatrix(X4,Y4,positive="1")
pCon


# 90-135 angle.

# Applying model to test data
mypred2 <- predict(myglm2, subdata2, type="response")
myproc2 <- roc(subdata2$Outcome.y.90to135, mypred2)
plot(myproc2,main='Outcome 90 to 135')

# Setting threshold at 0.001
mypred2[mypred2>0.001] <- 1
mypred2[mypred2<0.001] <- 0

# Setting variables to categorical.
X5 <- as.factor(mypred2)
Y5 <- as.factor(test$Outcome.y.90to135)

# Generating confusion matrix.
pCon2 <- confusionMatrix(X5,Y5,positive="1")
pCon2


# 135-180 angle.

# Applying model to test data
mypred3 <- predict(myglm3, subdata3, type="response")
myproc3 <- roc(subdata3$Outcome.y.135to180, mypred3)
plot(myproc3,main='Outcome 135 to 180')

# Setting threshold at 0.015
mypred3[mypred3>0.015] <- 1
mypred3[mypred3<0.015] <- 0

# Setting variables to categorical.
X6 <- as.factor(mypred3)
Y6 <- as.factor(test$Outcome.y.135to180)

# Generating confusion matrix.
pCon3 <- confusionMatrix(X6,Y6,positive="1")
pCon3




## Ordinal Logistic Regression using ALL variables

# Applying original OLR model (before backwards stepwise) to test data.
# Then generating its multiclass ROC curve as well as plotting the separate ROCs.
mypred4 <- predict(mymodel, test, type="prob")
myproc4 <- multiclass.roc(test$outcome, mypred4)

# ROC for 45-90.
R <- roc(test$Outcome.y.45to90 ~ mypred4[,2])

# ROC for 90-135.
R2 <- roc(test$Outcome.y.90to135 ~ mypred4[,3])

# ROC for 135-180.
R3 <- roc(test$Outcome.y.135to180 ~ mypred4[,4])

# Changing the graphical parameters so 3 plots can be observed together.
par(mfrow = c(1,3))


plot(R, main='Outcome 45 to 90')
plot(R2, main='Outcome 90 to 135')
plot(R3, main='Outcome 135 to 180')

# Changing the graphical parameters back to default.
par(mfrow = c(1,1))


Mypred4 <- predict(mymodel, test, type="class")


(Tab <- table(Mypred4,test$outcome))


# Taking probabilities for each outcome and separating them into variables.

# 45-90.
P3 <- mypred4[,2]

# 90-135.
P4 <- mypred4[,3]

# 135-180.
P5 <- mypred4[,4]


# Confusion matrices were generated

# 45-90.

# Setting threshold at 0.06
P3[P3>0.06] <- 1
P3[P3<0.06] <- 0

# Setting predictor & outcome as categorical
X10 <- as.factor(P3)
Y10 <- as.factor(test$Outcome.y.45to90)

# Creating the confusion matrix.
Con10 <- confusionMatrix(X10,Y10,positive="1")
Con10

# 90-135.

# Setting the threshold at 0.01
P4[P4>0.01] <- 1
P4[P4<0.01] <- 0

# Setting predictor & outcome as categorical.
X11 <- as.factor(P4)
Y11 <- as.factor(test$Outcome.y.90to135)

# Creating confusion matrix.
Con11 <- confusionMatrix(X11,Y11,positive="1")
Con11

# 135-180.

# Setting threshold at 0.02
P5[P5>0.02] <- 1
P5[P5<0.02] <- 0

# Setting predictor & categorical as categorical.
X12 <- as.factor(P5)
Y12 <- as.factor(test$Outcome.y.135to180)

# Creating confusion matrix.
Con12 <- confusionMatrix(X12,Y12,positive="1")
Con12
