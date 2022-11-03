import cv2 # OpenCV para computer vision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Para graficar
#import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split

#Cargando datos rostros Pascual
Ruta_dataset = './Dataset'
Filas=128
Columnas=128
Dataset=np.zeros((20,Filas*Columnas+1))

for i in range(0,20,1):
  Ruta=Ruta_dataset + '/' + str(i+1) + '.jpg'
  img=cv2.imread(Ruta)
  I_gris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  I_gris=cv2.resize(I_gris, (Filas,Columnas), interpolation = cv2.INTER_AREA)
  Dataset[i,0:Filas*Columnas]=I_gris.reshape((1,Filas*Columnas))
  if i>=0 and i<=5:
    Dataset[i,Filas*Columnas]=1
  else:
    if i>=6 and i<=10:
      Dataset[i,Filas*Columnas]=2
    else:
      if i>=11 and i<=15:
        Dataset[i,Filas*Columnas]=3
      else:
        if i>=16 and i<=20:
          Dataset[i,Filas*Columnas]=4
     
#print(Dataset.shape)

#2. Dividing dataset into input (X) and output (Y) variables
X = Dataset[:,0:Filas*Columnas]
Y = Dataset[:,Filas*Columnas]
#print(X.shape)
#print(Y.shape)
#print(Y)

X_train, X_test,Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=14541)
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)


# Showing the dataset images
Index=5
Imagen=X_train[Index,:]
Imagen=Imagen.reshape((Filas,Columnas))
plt.imshow(Imagen.astype('uint8'),cmap='gray',vmin=0, vmax=255)
#print('Este es el sugeto: ',Y_train[Index])

# Data normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#5. Evaluando casos mediante todos los clasificadores

Modelo_0 = KNeighborsClassifier(3)
Modelo_0.fit(X_train, Y_train)
Y_pred_0 =Modelo_0.predict (X_test)
#print("Accuracy KNN",accuracy_score(Y_test, Y_pred_0))

Modelo_1 = GaussianNB()
Modelo_1.fit(X_train, Y_train)
Y_pred =Modelo_1.predict (X_test)
#print("Accuracy Bayes",accuracy_score(Y_test, Y_pred))

Modelo_2 = LinearDiscriminantAnalysis()
Modelo_2.fit(X_train, Y_train)
Y_pred_2 =Modelo_2.predict (X_test)
#print("Accuracy LDA",accuracy_score(Y_test, Y_pred_2))

Modelo_3 = QuadraticDiscriminantAnalysis()
Modelo_3.fit(X_train, Y_train)
Y_pred_3 =Modelo_3.predict (X_test)
#print("Accuracy QDA",accuracy_score(Y_test, Y_pred_3))

Modelo_4 = DecisionTreeClassifier()
Modelo_4.fit(X_train, Y_train)
Y_pred_4 =Modelo_4.predict (X_test)
#print("Accuracy Tree",accuracy_score(Y_test, Y_pred_4))

Modelo_5 = SVC()
Modelo_5.fit(X_train, Y_train)
Y_pred_5 =Modelo_5.predict (X_test)
#print("Accuracy SVM",accuracy_score(Y_test, Y_pred_5))

#Reviewing an specific dataset target
Test=3
Target=np.zeros((1,Filas*Columnas))
Target[0,:]=X_test[Test,:]
Target_im=Target[0,:].reshape((Filas,Columnas))*255
plt.imshow(Target_im.astype('uint8'),cmap='gray',vmin=0, vmax=255)
Prediction_0 =Modelo_0.predict (Target)
Prediction_1 =Modelo_1.predict (Target)
Prediction_2 =Modelo_2.predict (Target)
Prediction_3 =Modelo_3.predict (Target)
Prediction_4 =Modelo_4.predict (Target)
Prediction_5 =Modelo_5.predict (Target)
print("La predicción de KNN es:",Prediction_0,', y debería ser: ',Y_test[Test])
print("La predicción de Bayes es:",Prediction_1,', y debería ser: ',Y_test[Test])
print("La predicción de LDA es:",Prediction_2,', y debería ser: ',Y_test[Test])
print("La predicción de QDA es:",Prediction_3,', y debería ser: ',Y_test[Test])
print("La predicción de Tree es:",Prediction_4,', y debería ser: ',Y_test[Test])
print("La predicción de SVM es:",Prediction_5,', y debería ser: ',Y_test[Test])

#Cargando datos rostros Pascual
Ruta_dataset = './Dataset_val'
Test=3
Ruta=Ruta_dataset + '/' + str(Test) + '.jpg'
img=cv2.imread(Ruta)
#plt.imshow(img[:,:,[2,1,0]].astype('uint8'),cmap='gray',vmin=0, vmax=255)
Filas=128
Columnas=128
Target=np.zeros((1,Filas*Columnas))
I_gris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
I_gris=cv2.resize(I_gris, (Filas,Columnas), interpolation = cv2.INTER_AREA)
Target[0,0:Filas*Columnas]=I_gris.reshape((1,Filas*Columnas))
Target = scaler.transform(Target)
Prediction_0 =Modelo_0.predict (Target)
if Prediction_0==1:
   print("La predicción de KNN es: Edwin")
else:
  if Prediction_0==2:
   print("La predicción de KNN es: Laura ")
  else:
    if Prediction_0==3:
      print("La predicción de KNN es: Sofia")
    else:
      if Prediction_0==4:
        print("La predicción de KNN es: Gabriela")