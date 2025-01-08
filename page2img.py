from pdf2image import convert_from_path
import os,shutil

def img_convert():
    # This works only for PDF files
    for root,subdir,files in os.walk('/home/Desktop/issue_resumes'):
        for each_file in files:
            src = root+'/'+each_file
            dst = '/home/Desktop/issue_resumes_image/'+(each_file.split('.')[0])+'.jpg'
            print(dst)
            images = convert_from_path(src)
            images[0].save(dst, 'JPEG')

def split(dir):
    root_path = './data/'+dir
    imgs_list = os.listdir(root_path)
    train_folder, test_folder = './data/train/'+dir, './data/test/'+dir
    # 0.80 in below line mean 80% data go into train folder and 20% into test
    train_size = int(len(imgs_list) * 0.80)
    test_size = int(len(imgs_list) * 0.20)
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        else:
            dest_folder = test_folder
        src = os.path.join(root_path, f)
        dst = os.path.join(dest_folder, f)
        shutil.copy(src, dst)

# To convert first page of pdf resume to image and save them in different folder
# img_convert()

# To split the images into test and train folders. Check the directory structure provided below
# split('images_column/')

# Below is the directory structure of data
'''
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
'''