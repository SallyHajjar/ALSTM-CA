def SupportVectorMachineAlgo(x_train_vft, y_train, x_validation, y_validation, x_test, yTrue,yPrevious):
    print("Support Vector Classifier")
    lsvc = svm.LinearSVC(C=0.1,class_weight=class_weight)
    lsvc.fit(x_train_vft, y_train)
    yPred = predict(lsvc, x_test, array_to_table(yPrevious))
    analyze_prediction(yPred,  yTrue, yPrevious, 'Model')
    plotMap(yPred.reshape(yTrue.shape[0], yTrue.shape[1])[200:400, 300:588])
def MultinomialLRAlgo(x_train_vft, y_train, x_validation, y_validation, x_test, yTrue,yPrevious):
    print("Multinomial Logistic Regression")
    mlr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mlr.fit(x_train_vft, y_train)
    yPred = predict(mlr, x_test, array_to_table(yPrevious))
    analyze_prediction(yPred,  yTrue, yPrevious, 'Model')
    plotMap(yPred.reshape(yTrue.shape[0], yTrue.shape[1])[200:400, 300:588])
