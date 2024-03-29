model = Sequential()
model.add(LSTM(56, input_shape=(1,21), return_sequences=True))
model.add(Dropout(0.3))
model.add(attention())
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(4))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, batch_size=100, epochs=100,validation_data=(xTest, yTest), verbose=1)
yPred = predict(model, xTest, array_to_table(dens_map_2010_c))
np.savetxt(outputFolder + "yPred.csv", yPred, delimiter=",")
np.savetxt(outputFolder + "yIn.csv", array_to_table(dens_map_2010_c), delimiter=",")
analyze_prediction(yPred, dens_map_2020_c, dens_map_2010_c, 'Model')
plotMap(dens_map_2010_c[200:400, 300:588])
plotMap(dens_map_2020_c[200:400, 300:588])
plotMap(yPred.reshape(dens_map_2020_c.shape[0], dens_map_2020_c.shape[1])[200:400, 300:588])
