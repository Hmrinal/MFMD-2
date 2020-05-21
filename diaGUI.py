print('****************************************************************************')
print('\nApplication\t :\t HealthCare Analysis')
print('\nDeveloper\t :\t Mrinal Tyagi, Baljit Kaur Gill')
print('\nDate\t\t :\t 08/07/2019')
print('\nDescription\t :\t HealthCare Analysis In TechCompany')
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
Diabetes =  IntVar(root)

cur_preg = 0
cur_glucose = 0
cur_diastolic= 0
cur_triceps = 0
cur_insulin = 0
cur_bmi= 0.0
cur_dpf= 0.0
cur_age = 0
cur_diabetes = 0

Preg_List = ['0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
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
label9 = Label( labelframe, text = 'Diabetes',justify=LEFT )
label9.grid(row = 9,column = 1)

a1 = ttk.Entry(labelframe, textvariable=Pregnancies,width =20)
a2 = ttk.Entry(labelframe, textvariable=Glucose,width =20)
a3 = ttk.Entry(labelframe, textvariable=Diastolic,width =20)
a4 = ttk.Entry(labelframe, textvariable=Triceps,width =20)
a5 = ttk.Entry(labelframe, textvariable=Insulin,width =20)
a6 = ttk.Entry(labelframe, textvariable=Bmi,width =20)
a7 = ttk.Entry(labelframe, textvariable=Dpf,width =20)
a8 = ttk.Entry(labelframe, textvariable=Age,width =20)
a9 = ttk.Entry(labelframe, textvariable=Diabetes,width =20)

a1.grid(row=1, column=8)
a2.grid(row=2, column=8)
a3.grid(row=3, column=8)
a4.grid(row=4, column=8)
a5.grid(row=5, column=8)
a6.grid(row=6, column=8)
a7.grid(row=7, column=8)
a8.grid(row=8, column=8)
a9.grid(row=9, column=8)

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
    Diabetes.set('-')
    

def checkcmbo():
    global cur_preg,cur_glucose,cur_diastolic,cur_triceps,cur_insulin,cur_bmi,cur_dpf,cur_age,cur_diabetes
    Sel = ''
    cur_preg = popupMenu1.get()
    cur_glucose=a2.get()
    cur_diastolic=a3.get()
    cur_triceps= a4.get()
    cur_insulin= a5.get()
    cur_bmi=a6.get()
    cur_dpf=a7.get()    
    cur_age=a8.get()
    cur_diabetes=a9.get()
    Sel = "Patients :\t" + str(cur_preg)
    Sel = Sel + "\nGlucose\t\t\t\t:\t" + str(cur_glucose) + "\nDiastolic\t\t\t\t:\t"+ str(cur_diastolic) + "\nTriceps\t\t\t:\t" + str(cur_triceps) + "\nInsulin \t\t\t:\t" + str(cur_insulin) + "\nBmi\t\t\t:\t" + str(cur_bmi) + "\nDpf\t\t\t:\t" + str(cur_dpf) + "\nAge\t\t\t:\t" + str(cur_age)+ "\nDiabetes\t\t\t:\t" + str(cur_diabetes)
    
    messagebox.showinfo( "Current Selection", Sel)
    
    
btn1=Button(root,text="Clear",command=clearsel)
btn1.pack(side = BOTTOM)
btn2=Button(root,text="Check",command=checkcmbo)
btn2.pack(side = BOTTOM)


root.geometry("340x300+1050+150")
root.mainloop()
