# Check that we have the Titanic data, and read it in.

if (!file.exists("train.csv") | !file.exists("test.csv")) {
    stop("Please download the Kaggle Titanic data train.csv and test.csv\n",
         "and set the working directory to the directory containing them\n",
         "then re-run this script.\n", call.=FALSE)
}

train.colClasses <- c(PassengerId="integer",
                      Survived="integer",
                      Pclass="integer",
                      Name="character",
                      Sex="factor",
                      Age="numeric",
                      SibSp="integer",
                      Parch="integer",
                      Ticket="character",
                      Fare="numeric",
                      Cabin="character",
                      Embarked="factor")
test.colClasses <-  c(PassengerId="integer",
                      Pclass="integer",
                      Name="character",
                      Sex="factor",
                      Age="numeric",
                      SibSp="integer",
                      Parch="integer",
                      Ticket="character",
                      Fare="numeric",
                      Cabin="character",
                      Embarked="factor")
na.strings <- c("", "NA")

# Cannot use "train" as the name of the dataset, as it will collide with the
# caret train() function.  This in spite of R supposedly having separate
# namespaces for functions and variables.  This happens when train is run via
# do.call.  Change the name of test for consistency.
train.set <- read.csv("train.csv", colClasses=train.colClasses, na.strings=na.strings)
test.set <- read.csv("test.csv", colClasses=test.colClasses, na.strings=na.strings)
train.colnames <- colnames(train.set)
test.colnames <- colnames(test.set)