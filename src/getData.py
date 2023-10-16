import pandas as pd
import numpy as np

def change_label_to_flair_format(string):
    '''Transforms text mine's default labels into flair format.'''
    if string == "aucun":
        return "O"
    if " " in string:
        return "-".join(string.split(" "))
    else:
        return string

def split_data(file,data_folder,proba):
    '''
    Randomly distribute the sentences in the train/test/dev files according to the 
    distribution provided by the proba variable.
    Train/test/dev are stored in data_folder.
    '''
    df = pd.read_csv(file)
    df["Label2"] = df["Label"].apply(change_label_to_flair_format)
    fTrain = open(data_folder+"/train.txt","w",encoding='utf-8')
    fTest = open(data_folder+"/test.txt","w",encoding='utf-8')
    fDev = open(data_folder+"/dev.txt","w",encoding='utf-8')

    file = np.random.choice([0,1,2],1,p=proba)
    for _,row in df.iterrows():
        word = row["Token"][1:-1]
        if file == 0:
            fTrain.write(word+" "+row["Label2"]+"\n")
            if word == ".":
                fTrain.write("\n")
        elif file == 1:
            fTest.write(word+" "+row["Label2"]+"\n")
            if word == ".":
                fTest.write("\n")
        else:
            fDev.write(word+" "+row["Label2"]+"\n")
            if word == ".":
                fDev.write("\n")
        
        if word == ".":
            file = np.random.choice([0,1,2],1,p=proba) #train-test-dev

file = "train.csv"
data_folder = "train_data"
proba = [0.85,0.03,0.12] #Train/Test/Dev
