# Get vote predictions on the test data.
vote.probs.on.test <- vote.no.age.or.age(test.minimal, models, accuracies)
head(vote.probs.on.test)
vote.preds.on.test <- as.numeric(vote.probs.on.test > 0.5)
head(vote.preds.on.test)

# Prepare a CSV file for Kaggle submission.  This must have the PassengerId
# column from the test data, and the predictions in a column called Survived,
# with values 1 for survived or 0 for not, which is what is in
# vote.preds.on.test.
test.submit <- data.frame(PassengerId=test.set$PassengerId,
                          Survived=vote.preds.on.test)
write.csv(test.submit, "vote_submission.csv", quote=FALSE, row.names=FALSE)