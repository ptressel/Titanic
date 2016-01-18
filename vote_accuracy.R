# This is a sanity check -- run the voting procedure on the training data.
# This is not intended to be a viable estimate of the out-of-sample accuracy.
# But it will exercise the voting functions, and at least hint at whether the
# voting process is viable.  It had better not *lose* accuracy versus the
# estimated out-of-sample performance per classifier...  Run this after data
# setup, assess_classifiers, and train_classifiers.

if (!exists("accuracies")) {
    accuracies <- readRDS("accuracies.Rds")
}
if (!exists("models")) {
    models <- readRDS("models.Rds")
}

vote.probs.on.train <- vote.no.age.or.age(train.minimal, models, accuracies)
head(vote.probs.on.train)
# Use a threshold of 0.5.  Should look at ROC curve.  Is this the best
# threshold?  It should be, because the classifiers were asked to return the
# probability.  Also, voting mooshes them all together, so if some classifiers
# are skewing their probabilities, there's no reason to expect them to all do
# that the same way.  So each would need its threshold adjusted individually.
vote.preds.on.train <- as.numeric(vote.probs.on.train > 0.5)
head(vote.preds.on.train)
vote.accuracy.on.train <-
    sum((vote.preds.on.train == 1) == (train.minimal$Survived == "Y")) / nrow(train.minimal)
# > vote.accuracy.on.train
# [1] 0.8776655
# However, when submitted to Kaggle, this gets 0.77033.  This suggests
# overfitting.

# The baseline for the Titanic data appears to be a gender model, that predicts
# survival if female, not if male.  This model scores 0.76555 on the public
# leaderboard on Kaggle.  What is the accuracy of the gender model on the
# training data?  If it is on a par with its performance on the public test
# data, that gives weight to the idea that the voting model is overfitting.
# But if the gender model likewise does better on the training data, then that
# may point to a skewed split between test and train.  So...get the accuracy
# of the gender model on the training data.
gender.model.train.table <- table(train.set$Survived, train.set$Sex)
# > gender.model.train.table
#     female male
#   0     81  468
#   1    233  109
gender.model.train.accuracy <-
    (gender.model.train.table["0", "male"] +
     gender.model.train.table["1", "female"]) / nrow(train.set)
# > gender.model.train.accuracy
# [1] 0.7867565
# So, that does only a little better on the training data, which implies I
# should be looking at ways to simplfy / constrain my voting model, or switch
# to a simpler model altogether.