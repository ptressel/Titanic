# These are definitions and settings that do not take a lot of time to run,
# and are needed by multiple scripts.

require(caret, quietly=TRUE)
require(mboost, quietly=TRUE)  # for the Binomial function

features.all <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age")
formula.all <-
    as.formula(paste("Survived ~ ",
                     paste(features.all, collapse="+")))
features.all.but.age <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked")
formula.all.but.age <-
    as.formula(paste("Survived ~ ",
                     paste(features.all.but.age, collapse="+")))

# caret trainControl options for CV, and to prevent it from storing the entire
# dataset and intermediate results.
trctrl_repeatedcv <-
    trainControl(method="repeatedcv",
                 returnData=FALSE, returnResamp="none", savePredictions="none")
# oob is for random forest only.
trctrl_rf_oob <-
    trainControl(method="oob",
                 returnData=FALSE, returnResamp="none", savePredictions="none")
# Use this for preliminary code testing, so training doesn't take too long.
trctrl_repeatedcv_5_1 <-
    trainControl(method="repeatedcv", number=5, repeats=1,
                 returnData=FALSE, returnResamp="none", savePredictions="none")
# Use this for estimating accuracy.
trctrl_repeatedcv_10_1 <-
    trainControl(method="repeatedcv", number=10, repeats=1,
                 returnData=FALSE, returnResamp="none", savePredictions="none")
# Use this for the actual classifier training.
trctrl_repeatedcv_20_5 <-
    trainControl(method="repeatedcv", number=20, repeats=5,
                 returnData=FALSE, returnResamp="none", savePredictions="none")

# Options for specific classifiers.
params_glmboost <- list(family=Binomial())
params_gbm_bernoulli <- list(distribution="bernoulli", verbose=FALSE)
params_gbm_adaboost <- list(distribution="adaboost", verbose=FALSE)

# List keys for classifiers without Age, to be used on all data:
models.no.age <- c(
    "rf.no.age",
    "glmboost.no.age",
    "avnnet.no.age",
    "gbm_bernoulli.no.age",
    "gbm_adaboost.no.age"
)
# List keys for classifiers with Age, to be used on complete cases (i.e. rows
# that have Age):
models.age <- c(
    "rf.age",
    "glmboost.age",
    "avnnet.age",
    "gbm_bernoulli.age",
    "gbm_adaboost.age"
)