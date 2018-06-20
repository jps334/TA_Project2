

def Classification(model, c_index, dataset_target):
    mnb_model = model.fit(c_index, dataset_target)

    return mnb_model

def prediction(model, dataset_test):
    predicted = model.predict(dataset_test)

    return predicted








