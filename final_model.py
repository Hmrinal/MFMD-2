print('****************************************************************************')
print('\nApplication\t :\t HealthCare Analysis')
print('\nDeveloper\t :\t Mrinal Tyagi, Baljit Kaur Gill')
print('\nDate\t\t :\t 08/07/2019')
print('\nDescription\t :\t HealthCare Analysis')
print('\n****************************************************************************')

#importing the libraries required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import seaborn as sns

#initializing the widgets and access the root window
root=Tk()
root.title('Healthcare Analysis')
root.configure(bg="black")
root.resizable(width=False, height=False)

#setting the global variables with appropriate datatype
Pregnancies = IntVar(root)
Glucose = IntVar(root)
Diastolic = IntVar(root)
Triceps= IntVar(root)
Insulin =  IntVar(root)
Bmi = DoubleVar(root)
Dpf =  DoubleVar(root)
Age =  IntVar(root)
#Diabetes =  IntVar(root)

cur_preg = 0
cur_glucose = 0
cur_diastolic= 0
cur_triceps = 0
cur_insulin = 0
cur_bmi= 0.0
cur_dpf= 0.0
cur_age = 0
#cur_diabetes = 0

Preg_List = ['0', '1','2','3','4','5','6','7','8','9']
Pregnancies.set('-')

labelframe = LabelFrame(root, text = "Provide Inputs",fg="blue")
labelframe.pack(fill = "both")

label1 = Label( labelframe, text = 'Pregnancies',justify=LEFT )
label1.grid(row = 1,column = 1)
label2 = Label( labelframe, text = 'Glucose',justify=LEFT )
label2.grid(row = 2,column = 1)
label3 = Label( labelframe, text = 'Diastolic',justify=LEFT )
label3.grid(row = 3,column = 1)
label4 = Label( labelframe, text = 'Triceps',justify=LEFT )
label4.grid(row = 4,column = 1)
label5 = Label( labelframe, text = 'Insulin',justify=LEFT )
label5.grid(row = 5,column = 1)
label6 = Label( labelframe, text = 'Bmi',justify=LEFT )
label6.grid(row = 6,column = 1)
label7 = Label( labelframe, text = 'Dpf',justify=LEFT )
label7.grid(row = 7,column = 1)
label8 = Label( labelframe, text = 'Age',justify=LEFT )
label8.grid(row = 8,column = 1)
#label9 = Label( labelframe, text = 'Diabetes',justify=LEFT )
#label9.grid(row = 9,column = 1)

a1 = ttk.Entry(labelframe, textvariable=Pregnancies,width =20)
a2 = ttk.Entry(labelframe, textvariable=Glucose,width =20)
a3 = ttk.Entry(labelframe, textvariable=Diastolic,width =20)
a4 = ttk.Entry(labelframe, textvariable=Triceps,width =20)
a5 = ttk.Entry(labelframe, textvariable=Insulin,width =20)
a6 = ttk.Entry(labelframe, textvariable=Bmi,width =20)
a7 = ttk.Entry(labelframe, textvariable=Dpf,width =20)
a8 = ttk.Entry(labelframe, textvariable=Age,width =20)
#a9 = ttk.Entry(labelframe, textvariable=Diabetes,width =20)

a1.grid(row=1, column=8)
a2.grid(row=2, column=8)
a3.grid(row=3, column=8)
a4.grid(row=4, column=8)
a5.grid(row=5, column=8)
a6.grid(row=6, column=8)
a7.grid(row=7, column=8)
a8.grid(row=8, column=8)
#a9.grid(row=9, column=8)

popupMenu1 = ttk.Combobox(labelframe, textvariable= Pregnancies, values = Preg_List, state = 'readonly',width =17)  
popupMenu1.grid(row = 1,column = 8)

def clearsel() :
    Pregnancies.set(0)
    Glucose.set('-')
    Diastolic.set('')
    Triceps.set('-')
    Insulin.set('-')
    Bmi.set('-')
    Dpf.set('')
    Age.set(0)
    #Diabetes.set('-')
    

def checkcmbo():
    global cur_preg,cur_glucose,cur_diastolic,cur_triceps,cur_insulin,cur_bmi,cur_dpf,cur_age#,cur_diabetes
    Sel = ''
    cur_preg = popupMenu1.get()
    cur_glucose=a2.get()
    cur_diastolic=a3.get()
    cur_triceps= a4.get()
    cur_insulin= a5.get()
    cur_bmi=a6.get()
    cur_dpf=a7.get()    
    cur_age=a8.get()
    #cur_diabetes=a9.get()
    Sel = "Patients :\t" + str(cur_preg)
    Sel = Sel + "\nGlucose\t\t\t:\t" + str(cur_glucose) + "\nDiastolic\t\t\t:\t"+ str(cur_diastolic) + "\nTriceps\t\t\t:\t" + str(cur_triceps) + "\nInsulin \t\t\t:\t" + str(cur_insulin) + "\nBmi\t\t\t:\t" + str(cur_bmi) + "\nDpf\t\t\t:\t" + str(cur_dpf) + "\nAge\t\t\t:\t" + str(cur_age)#+ "\nDiabetes\t\t\t:\t" + str(cur_diabetes)
    
    messagebox.showinfo( "Current Selection", Sel)
    
def NN():
    global cur_preg,cur_glucose,cur_diastolic,cur_triceps,cur_insulin,cur_bmi,cur_dpf,cur_age#,cur_diabetes
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
     #import seaborn as sns
    cur_preg = popupMenu1.get()
    cur_glucose=a2.get()
    cur_diastolic=a3.get()
    cur_triceps= a4.get()
    cur_insulin= a5.get()
    cur_bmi=a6.get()
    cur_dpf=a7.get()    
    cur_age=a8.get() 
    # Importing the dataset
    train_data = pd.read_csv('diabetes_data.csv')
    X = train_data.iloc[:,:-1].values
    y = train_data.iloc[:, -1].values
    #dataset[:,4].hist()
    #dataset.describe()
    #dataset.head()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from tensorflow.contrib.keras.api.keras import models
    from tensorflow.contrib.keras.api.keras import layers
    
    classifier = models.Sequential()
    # Adding the input layer and the first hidden layer
    #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(layers.Dense(units = 12,  activation = 'relu', input_dim = 8))
    classifier.add(layers.Dense(units = 22,  activation = 'relu', input_dim = 8))
    # Adding the second hidden layer
    #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(layers.Dense(units = 8, activation = 'relu'))
    

    # Adding the output layer
    #classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.add(layers.Dense(units = 1, activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    #classifier.fit(X_train, y_train, batch_size = 10, epoch = 100)
    classifier.fit(X_train, y_train, batch_size = 5, epochs = 30, validation_data=(X_test, y_test))
    # Part 3 - Making the predictions and evaluating the model
   
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
   
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print (cm)

    cur_preg = popupMenu1.get()
    cur_glucose=a2.get()
    cur_diastolic=a3.get()
    cur_triceps= a4.get()
    cur_insulin= a5.get()
    cur_bmi=a6.get()
    cur_dpf=a7.get()    
    cur_age=a8.get()
    
    new_prediction=classifier.predict(np.array([[popupMenu1.get(),a2.get(),a3.get(),a4.get(),a5.get(),a6.get(),a7.get(),a8.get()]]))
    new_prediction=(new_prediction > 0.5)
    print (new_prediction)
    if new_prediction==True:
        asy= "Sorry, You have probable chances of suffering from diabetes"    
        messagebox.showinfo( "PREDICTION DONE",asy)
        
    else:
        asy= "Congratulations! You are healthy"    
        messagebox.showinfo( "PREDICTION DONE",asy)
        

        
    
    
"""
    from keras.models import load_model
    global cur_preg,cur_glucose,cur_diastolic,cur_triceps,cur_insulin,cur_bmi,cur_dpf,cur_age
    classifier = load_model('classifier.h5')
    summarize model.
    classifier.summary()
     
    
    
    e1 = popupMenu1.get()
    e2 = a2.get()
    e3 = a3.get()
    e4 = a4.get()
    e5 = a5.get()
    e6 = a6.get()
    e7 = a7.get()
    e8 = a8.get()
    
    root.destroy()
    
    global params
    global arr
    params = [e1,e2,e3,e4,e5,e6,e7,e8]
    arr = np.asarray(params)
    arr = arr.reshape(1,-1)
    
   
    new_prediction=classifier.predict(sc.transform(np.array([[popupMenu1.get(),a2.get(),a3.get(),a4.get(),a5.get(),a6.get(),a7.get(),a8.get()]])))
    new_prediction=(new_prediction>0.5)
    print (new_prediction)
    new_prediction=classifier.predict(sc.transform(np.array([[cur_preg,cur_glucose,cur_diastolic,cur_triceps,cur_insulin,cur_bmi,cur_dpf,cur_age]])))
    new_prediction=(new_prediction>0.5)
    print (new_prediction)
   
    asy="YAY, DONE WITH TRAINING"    
    messagebox.showinfo( "TRAINING FINISHED",asy)
    
"""   

btn1=Button(root,text="Clear",command=clearsel)
btn1.pack(side = BOTTOM)
btn2=Button(root,text="Check",command=checkcmbo)
btn2.pack(side = BOTTOM)
btn3=Button(root,text="Test",command=NN)
btn3.pack(side = BOTTOM)

root.geometry("340x300+1050+150")
root.mainloop()


