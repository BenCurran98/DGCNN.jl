function accuracy(predicted, target)
    correct_preds = predicted .== target
    return mean(correct_preds)
end