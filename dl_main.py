
import module_dl3D_c3 as mdl
from pathlib import Path
import keras.utils.np_utils
import argparse
import matplotlib.pyplot as plt
import scikitplot as skplt
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Deep Learning Algorith - projet ")


parser.add_argument("-i", help='Path to the 4 lists of protein complexes that are : heme, control, nucleotide and steroid',

                 metavar="path_to_lists", required=True)

parser.add_argument("-n", help="Path leading to the .npy files"

                , metavar="npy_files", required=True)

parser.add_argument("-s", help='Size of the whole Data you want to use --OPTIONNE -- Default value is 450',

                 dest = "data_size", metavar="data_size", type = int , default = 450)

parser.add_argument("-e", help='Number of epochs to use --OPTIONNEL -- Default value is 20',

                 dest = "Epochs_size", metavar="Epochs_size", type = int , default = 20)

parser.add_argument("-b", help='The size of batch you want to use --OPTIONNE -- Default value is 15',

                 dest = "Batch_size",metavar="Batch_size", type = int , default = 15)



args = parser.parse_args()

print("##################### TENSERFLOW PRINTS ######################")


path_to_data = args.n

print(' path_to_data you entered is : ', path_to_data)

path_to_lists =  args.i

print("path_to_lists you entered is : ", path_to_lists)

data_size = args.data_size

Epochs_size = args.Epochs_size

Batch_size = args.Batch_size






#definir la tailles des échantillons :

taille_sup,taille = mdl.Resize_data(data_size)

data_total,heme,steroid,nucleotide,control = mdl.read_lists(path_to_lists,taille,taille_sup)


#find the shape of data :

new_shape = mdl.make_shape(data_size,path_to_data)



while len(new_shape) != 5:

    new_shape = mdl.make_shape(data_size,path_to_data)

    print("Function make shape is called !!")

print("the input shape of your data is :", new_shape)


#create data :

x_array, y_array = mdl.create_data(data_size,heme, nucleotide, control, steroid,data_total,path_to_data)

#split data :

x_train, x_test, y_train, y_test, x_val, y_val = mdl.split_data(x_array,y_array)


vald_array = keras.utils.np_utils.to_categorical(y_val, 3)

n_y_array = keras.utils.np_utils.to_categorical(y_train, 3)

print("val shape: ",x_val.shape, " ; ", y_val.shape)
print("train shape: ",x_train.shape, " ; ", y_train.shape)
print("test shape: ",x_test.shape, " ; ", y_test.shape)



#create model :

model = mdl.create_model(new_shape)

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#training:

history = mdl.train_model(model,x_train, n_y_array,x_val, vald_array, Epochs_size, Batch_size)


#evaluating the model :

mdl.model_evaluate(model,x_train,n_y_array,x_val, vald_array)


#plot history :

mdl.plot_metrics(history)

#predicting :


prdd, predict_class =mdl.model_predict(model,x_test,y_test)


#plot roc : 

skplt.metrics.plot_roc_curve(y_test, prdd)

plt.show()

# table : 

variable1 = predict_class

variable2 = y_test.tolist()


df = pd.DataFrame({'prd' : variable1,
                  'real' : variable2})

df = pd.crosstab(df.prd, df.real)

print(df)


##Prédire les stéroid :

os.chdir(path_to_lists)

steroid = [i.strip() for i in open("steroid.list").readlines()]

x_steroid =  np.zeros(shape = (len(steroid),14,32,32,32))

y_steroid =  np.zeros(shape = (len(steroid)))


os.chdir(path_to_data)

file_count = 0

for file in steroid:

    y_steroid[file_count]= 2

    x_steroid[file_count] = np.load(str(file+".npy"))

    file_count+=1

#predire : 


prd_str = model.predict(x_steroid)

predict_class_st = np.argmax(prd_str, axis=1)

predict_class_str = predict_class_st.tolist()

print("prd str", predict_class_str)

pr2 = predict_class_str.count(2)/len(predict_class_str)
pr1 = predict_class_str.count(1)/len(predict_class_str)
pr0 = predict_class_str.count(0)/len(predict_class_str)

print("pourcentage des stéroids prédits comme control : ", pr2,"%")
print("pourcentage des stéroids prédits comme heme : ", pr0,"%")
print("pourcentage des stéroids prédits comme nucleotide : ", pr1,"%")
