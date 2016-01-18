# Vote several classifiers, weighted by an out-of-sample accuracy, averaged
# over multiple CV runs.  There are two sets of classifiers -- one uses the
# Age feature, and should be applied to records that have Age.  The other
# does not use Age, and should be applied to records that do not have Age.
# We're predicting a binary outcome, but the results may be more accurate
# if we have the classifiers return a probability, as that will show how
# confident the classifier is.  Then we can take a weighted average of the
# probabilities, and threshold it at 0.5.  @ToDo, the threshold should be a
# parameter that is tuned by means of CV.

# Define caret and model settings.
source("settings.R")

# preds is a list in which there is one vector of probability predictions per
# classifier.
# accuracies are the performance values per classifier, in the same order as
# the elements in the preds.
vote <- function(preds.per.classifier, accuracies.per.classifier) {
    norm <- sum(accuracies.per.classifier)
    preds.matrix <- as.matrix(as.data.frame(preds.per.classifier))
    votes <- apply(preds.matrix, 1,
                   function(row) sum(row * accuracies.per.classifier) / norm)
    votes
}

# Parameters:
# dataset -- the data to be predicted.
# models -- the trained classifier models.
# accuracies -- this is the output of running assess_classifiers.R, which
# includes the estimated out-of-sample accuracies for the various
# classifiers.  These are not assumed to be normalized.
# Values from settings -- these should be in the environment:
# features.all -- all the features we're using.
# features.all.but.age -- all features except Age.
# models.no.age -- keys in lists of models, accuracies for classifiers that do
# not use Age.
# models.age -- keys for classifiers that do use Age.
vote.no.age.or.age <- function(dataset, models, accuracies) {
    # Rows that are missing Age.
    rows.no.age <- is.na(dataset$Age)
    # Select the rows that are missing Age.
    dataset.no.age <- dataset[rows.no.age, ]
    # Select the complete rows (i.e. the rows with Age).
    dataset.age <- dataset[!rows.no.age, ]
    # Separate the accuracies.
    accuracies.only <- sapply(accuracies, `[[`, "avg_accuracy")
    accuracies.no.age <- accuracies.only[models.no.age]
    accuracies.age <- accuracies.only[models.age]
    # For each classifier that does not use Age, predict on only the rows that
    # are missing Age.
    probs.no.age.per.classifier <-
#         lapply(models[models.no.age],
#                predict,
#                newdata=dataset.no.age,
#                type="prob")
        lapply(models[models.no.age],
               function(model, d) {
                   p <- predict(model, newdata=d, type="prob")
                   # Return only the probabilities for survival.
                   p$Y
               },
               dataset.no.age)
    # For each classifier that does use Age, predict on only the rows that have
    # Age.
    probs.age.per.classifier <-
#         lapply(models[models.age],
#                predict,
#                newdata=dataset.age,
#                type="prob")
        lapply(models[models.age],
               function(model, d) {
                   p <- predict(model, newdata=d, type="prob")
                   # Return only the probabilities for survival.
                   p$Y
               },
               dataset.age)
    # Combine the predictions without Age by accuracy-weighted voting.
    probs.no.age <- vote(probs.no.age.per.classifier, accuracies.no.age)
    # Combine the predictions with Age by accuracy-weighted voting.
    probs.age <- vote(probs.age.per.classifier, accuracies.age)
    # Re-assemble the predictions in their original order.
    probs <- numeric(nrow(dataset))
    probs[rows.no.age] <- probs.no.age
    probs[!rows.no.age] <- probs.age
    probs
}