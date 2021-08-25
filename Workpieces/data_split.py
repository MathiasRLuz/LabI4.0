import os
from pathlib import Path
import random
	
pasta=os.getcwd()
pasta=os.path.join(pasta,'Workpieces')

CATEGORIES = ["black", "metal", "red"]
for category in CATEGORIES:    
    folder=os.path.join(pasta,category)
    print(folder)        
    included_extensions = ['jpg']
    file_names = [fn for fn in os.listdir(folder) if any(fn.endswith(ext) for ext in included_extensions)]
    train_folder = os.path.join(pasta,'train',category)
    test_folder = os.path.join(pasta,'test',category)
    val_folder = os.path.join(pasta,'validation',category)
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)
    Path(val_folder).mkdir(parents=True, exist_ok=True)

    ### Datasplit  

    train_size,test_size,val_size=[0.7,0.15,0.15]
    n_train=round(train_size*len(file_names))
    n_val=round(len(file_names)*val_size)
    n_test=len(file_names)-n_val-n_train        
    random.shuffle(file_names)
    train=file_names[:n_train]
    val=file_names[n_train:n_train+n_val]
    test=file_names[n_train+n_val:]    
    for i in range(len(train)):
        for file in file_names:
            if file.startswith(train[i]):
                os.rename(folder+'\\'+file, train_folder+'\\'+file)
    for i in range(len(test)):
        for file in file_names:
            if file.startswith(test[i]):
                os.rename(folder+'\\'+file, test_folder+'\\'+file)
    for i in range(len(val)):
        for file in file_names:
            if file.startswith(val[i]):
                os.rename(folder+'\\'+file, val_folder+'\\'+file)