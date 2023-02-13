featuresFolder = '/content/drive/MyDrive/SusDens/dataandcodes/features/'
outputFolder = '/content/drive/MyDrive/SusDens/output/'
_, dens_map_2000 = raster.read(featuresFolder + 'BU2000.tif', bands=1)
_, dens_map_2010 = raster.read(featuresFolder + 'BU2010.tif', bands=1)
_, dens_map_2020= raster.read(featuresFolder + 'BU2020.tif', bands=1)

dens_map_2000_c = convertToClasses(dens_map_2000)
dens_map_2010_c = convertToClasses(dens_map_2010)
dens_map_2020_c = convertToClasses(dens_map_2020)
xtrain = convertToOneColumnArray(dens_map_2000)
xtest = convertToOneColumnArray(dens_map_2010)

featuresArray_train = [xtrain]
featuresArray_test = [xtest]
for featureDict in featuresDict:
	if featureDict['enabled']:
		_, arr = raster.read(featuresFolder + featureDict['filename'], bands = featureDict['bands'])
		if featureDict['pixels_to_remove']['0'] is not None:
			arr = np.delete(arr, featureDict['pixels_to_remove']['0'], 0)
		if featureDict['pixels_to_remove']['1'] is not None:
			arr = np.delete(arr, featureDict['pixels_to_remove']['1'], 1)
		arr = convertToOneColumnArray(arr)
		featureDict['arr'] = arr
		featuresArray_train.append(arr)
		featuresArray_test.append(arr)
dens_map_2000_n = calc_neighbors_c(dens_map_2000_c, 4)
dens_map_2010_n = calc_neighbors_c(dens_map_2010_c, 4)

for i in range(4):
  featuresArray_train.append(convertToOneColumnArray(dens_map_2000_n[i]))
  featuresArray_test.append(convertToOneColumnArray(dens_map_2010_n[i]))
featuresArray_train = np.concatenate(np.array(featuresArray_train), 1)
featuresArray_test = np.concatenate(np.array(featuresArray_test), 1)
outputFolder = '/content/drive/MyDrive/SusDens/output/'

# print(featuresArray_train.shape)
# np.savetxt(outputFolder + "featuresArray_train.csv", featuresArray_train, delimiter=",")
# np.savetxt(outputFolder + "featuresArray_test.csv", featuresArray_test, delimiter=",")

X = convertToOneColumnArray(dens_map_2000_c)
y = array_to_table(dens_map_2010_c)

X_transitions = convertToOneColumnArray(computeTransitions(dens_map_2000_c, dens_map_2010_c))

featuresX = np.concatenate([featuresArray_train, X_transitions], 1)
xTrain, xValidation, yTrain, yValidation = train_test_split(featuresX, y, test_size=0.2, random_state=22, stratify = X_transitions)
xTrain=np.delete(xTrain, xTrain.shape[1] - 1,1)
xValidation=np.delete(xValidation, xValidation.shape[1] - 1,1)
print(xTrain.shape, xValidation.shape, yTrain.shape, yValidation.shape)
# Normalise the data
xTrain = xTrain / 188914
xValidation = xValidation / 188914
xTest = featuresArray_test / 188914

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1])) #Another additional pre-processing step is to reshape the features from two-dimensions to three-dimensions, such that each row represents an individual pixel.
xValidation = xValidation.reshape((xValidation.shape[0], 1, xValidation.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
