from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils import np_utils


# alphabets dictionary 
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
						14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 

file = 'handwritten.csv'
data = pd.read_csv(file).astype('float32')
data.rename(columns={'0':'label'}, inplace=True)
# feature are splited 
x = data.drop('label', axis = 1)
y = data['label']

(x_train, x_test, y_train, y_test) = train_test_split(x, y)
standard_scale = MinMaxScaler()
standard_scale.fit(x)

# test images are preprocess for text classification
x_test = standard_scale.transform(x_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
y_test = np_utils.to_categorical(y_test)
# loading the trained model
model = load_model('handWritten.h5')

# data for prediction
plt.imshow(x_test[2].reshape(28, 28), cmap='Greys')
plt.show()

test_img = x_test[2]
test_img = test_img.reshape(-1, 28, 28, 1)
predicted_data = np.argmax(model.predict(test_img))
print("Predicted data{} actual data{}", alphabets_mapper[predicted_data], y_test[2])

