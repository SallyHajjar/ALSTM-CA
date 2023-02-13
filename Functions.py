def rst(file):
    Raster = gdal.Open(file)
    Band=Raster.GetRasterBand(1)
    Array=Band.ReadAsArray()
    return(Array)

def create_mask(radius, omit_center):
	mask_size = 2*radius+1
	mask = np.ones((mask_size, mask_size))
	if omit_center:
		mask_center = radius
		mask[mask_center, mask_center] = np.nan
	return mask

def find_neighbors(p_arr, radius, row_number, column_number):
	return [[p_arr[i][j] if  i >= 0 and i < len(p_arr) and j >= 0 and j < len(p_arr[0]) else np.nan
		for j in range(column_number-radius, column_number+radius+1)]
		for i in range(row_number-radius, row_number+radius+1)]

def calc_neighbors(p_original_arr):
	neighbors_radius = 3
	nb_rows = p_original_arr.shape[0]
	nb_cols = p_original_arr.shape[1]
	result = np.zeros(shape=(nb_rows, nb_cols))
	mask = create_mask(neighbors_radius, True)
	for i in range(0, nb_rows):
		for j in range(0, nb_cols):
			neighbors_array = np.array(find_neighbors(p_original_arr, neighbors_radius, i, j))
			cell_res = neighbors_array*mask
			sum = np.nansum(cell_res)
			count = np.count_nonzero(~np.isnan(cell_res))
			cell_val = sum/count if count > 0 else 0
			result[i, j] = cell_val
			
	return result

def calc_neighbors_c(p_original_arr, number_classes):
	neighbors_radius = 3
	nb_rows = p_original_arr.shape[0]
	nb_cols = p_original_arr.shape[1]
	results = np.zeros(shape=(number_classes, nb_rows, nb_cols))
	mask = create_mask(neighbors_radius, True)
	for i in range(0, nb_rows):
		for j in range(0, nb_cols):
			neighbors_array = np.array(find_neighbors(p_original_arr, neighbors_radius, i, j))
			cell_res = neighbors_array*mask
			count = np.count_nonzero(~np.isnan(cell_res))
			sums = np.zeros(number_classes)
			for a in range(0,cell_res.shape[0]):
				for b in range(0,cell_res.shape[1]):
					if cell_res[a,b] is not None and not np.isnan(cell_res[a,b]):
						cell_val = int(cell_res[a,b])
						sums[cell_val] = sums[cell_val] + 1

			for k in range(0, 4):
				results[k, i, j] = sums[k]/count if count > 0 else 0
			
	return results

def convertToClasses(arr):
	arr_classes=arr
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]): 
			if arr[i,j] >= 0  and arr[i,j]  <= 24:
				arr_classes[i,j]=0
			elif arr[i,j] >= 25  and arr[i,j]  <= 102:
				arr_classes[i,j]=1
			elif arr[i,j] >= 103  and arr[i,j]  <= 499:
				arr_classes[i,j]=2
			elif arr[i,j] >= 500  and arr[i,j]  <= 2500:
				arr_classes[i,j]=3
	return arr_classes

def convertToOneColumnArray(arr):
	arr = array_to_table(arr)
	arr = arr.reshape((arr.shape[0], 1))
	return arr

def computeTransitions(p_x, p_y):	
	a=[]
	for i in range(p_x.shape[0]):
		for j in range(p_x.shape[1]): 
			if p_x[i,j] != p_y[i,j]:
				a.append(abs(p_y[i,j] - p_x[i,j])+(p_x[i,j]+3)*10)
			else:
				a.append(p_x[i,j])
	a=np.reshape(a,(p_x.shape[0], p_x.shape[1]))
	return a

def convertModelOutputToLabels(outPred):
	return np.argmax(np.array(outPred), axis=-1)

def calculatePlotConfusionMatrix(yTrue, yPred, yPrev):
	classes = np.unique(yTrue)
	target_names = ["Class {}".format(i) for i in classes]
	cMatrix = confusion_matrix(yTrue, yPred)
	classificationReport = classification_report(yTrue, yPred, target_names=target_names)
	accuracyScore = accuracy_score(yTrue, yPred)
	precisionScore = precision_score(yTrue, yPred, average='macro') #weighted
	recallScore = recall_score(yTrue, yPred, average='macro')

	A=0
	B=0
	C=0
	D=0
	for i in range(len(yPred)):
		if yPrev[i] != yTrue[i]  and yPrev[i] == yPred[i]:
			A=A+1
		elif yPrev[i] != yTrue[i]  and yPrev[i] != yPred[i]:
			B=B+1
		if yTrue[i]  == yPred[i]:
			C=C+1
		elif yPrev[i] == yTrue[i]  and yPrev[i] != yPred[i]:
			D=D+1
	
	FoM = (B/(A+B+C+D))

	print("Confusion matrix:\n", cMatrix)
	print("Classification Report:\n", classificationReport)
	print("Accuracy:", accuracyScore)
	print("Precision Score : ", precisionScore)
	print("Recall Score : ", recallScore)
	print("FoM : ", FoM)
 
	# plot confusion matrix
	plt.imshow(cMatrix, interpolation='nearest', cmap=plt.cm.Greens)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = cMatrix.max() / 2.
	for i, j in itertools.product(range(cMatrix.shape[0]), range(cMatrix.shape[1])):
			plt.text(j, i, format(cMatrix[i, j], fmt),
							horizontalalignment="center",
							color="white" if cMatrix[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	return cMatrix, classificationReport, accuracyScore, precisionScore, recallScore

def calculateTransitionMatrix(yIn, yOut):
	nb_unique_classes = len(np.unique(yOut))
	transitionMatrix = np.zeros((nb_unique_classes, nb_unique_classes))

	for i in range(yOut.shape[0]):
		transitionMatrix[yIn[i], yOut[i]] = transitionMatrix[yIn[i], yOut[i]] + 1

	for i in range(transitionMatrix.shape[0]):
		for j in range(transitionMatrix.shape[1]):
			print("Class" + str(i) + " to " + str(j) + ": ", int(transitionMatrix[i, j]))
			
	return transitionMatrix

def plotMap(map):
	show(map)
 
def predict(model, xTest, yPrevious):
	yTestPredicted = model.predict(xTest)
	yTestPredicted_labels = convertModelOutputToLabels(yTestPredicted)
 
	for i in range(0, len(yTestPredicted_labels)):
			if yTestPredicted_labels[i] < yPrevious[i]:
				yTestPredicted_labels[i]= yPrevious[i]
	return yTestPredicted_labels

def analyze_prediction(yPred, yTrue, yPrevious, printTitle = None):
	if printTitle is not None:
		print(printTitle + ":\n")

	yTrue_t = array_to_table(yTrue)
	yPrevious_t = array_to_table(yPrevious)

	calculatePlotConfusionMatrix(yTrue_t, yPred, yPrevious_t)
	calculateTransitionMatrix(yPrevious_t, yPred)
