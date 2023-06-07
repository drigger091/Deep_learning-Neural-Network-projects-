import zipfile
import subprocess

import kaggle

command = 'kaggle datasets download -d omkargurav/face-mask-dataset'

subprocess.run(command,shell=True)

zip_path = 'face-mask-dataset.zip'

with zipfile.ZipFile(zip_path,'r') as zip_ref:
    zip_ref.extractall('E:\GITHUB\Deep_learning-Neural-Network-projects-')