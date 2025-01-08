This repository contains code for
data preparation,
model training,
model testing.

Very detailed comments are mentioned in the code for your reference.

It is very important to follow the directory structure while training the model because the folder names are taken as labels for classification.

You can alter model accuracy and model size by changing the number of neurons in every layer of neural network model. It can be found in training code.

There is already a pretrained model available in this folder you can use it by installing same version packages as mentioned in requirements.txt file

You can also train your own model since I provided train and test scripts here.

# Below is the directory structure of data
/data/
├──         
├── train/          
│   └── column/    
│   └── noncolumn/    
├── test/          
│   └── column/    
│   └── noncolumn/    
├           
├── images_column/           
└── images_noncolumn/ 
