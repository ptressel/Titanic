# Set up the data that will be used for classifier evaluation and training.
# This has no extra features, as there is a lot of data copying during CV.
# Make a similar set for the test data.
# Note the work of data cleanup and imputation has been done in examine_data.R.

source("settings.R")

if (!exists("train.set")) {
    train.set <- readRDS("train_set.Rds")
}
if (!exists("test.set")) {
    test.set <- readRDS("test_set.Rds")
}

# caret and some classifiers rely on checking whether the outcome is a factor,
# so make it an explicit factor.
train.minimal <- train.set[, c("Survived", features.all, "Cabin")]
train.minimal$Survived <- as.factor(train.minimal$Survived)
# The level names are "0" and "1" which are not legal R identifiers.  This can
# cause problems for some classifiers, that convert those to column names, and
# that gets them made into identifiers, maybe "X0" and "X1".  In any case,
# change the names to "N" and "Y".
levels(train.minimal$Survived) <- c("N", "Y")

# Also make a set with only complete cases.
train.no.na <- train.minimal[!is.na(train.minimal$Age), ]

# The test data has no outcome column.
test.minimal <- test.set[, c(features.all, "Cabin")]
test.no.na <- test.minimal[!is.na(test.minimal$Age), ]