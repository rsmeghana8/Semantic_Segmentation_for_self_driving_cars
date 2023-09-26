import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s : %(message)s:]')


list_of_files =[
    ".github/workflows/.gitkeep",
    f'src/__init__.py',
    f'src/components/__init__.py',
    f'src/utils/__init__.py',
    f'src/config/__init__.py',
    f'src/config/configuration.py',
    f'src/pipeline/__init__.py',
    f'src/entity/__init__.py',
    f'src/constants/__init__.py',
    "config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py" ,
    "research/trails.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    #creating the directory if there is one
    if filedir !="": #(if filedir is not an empty string, means the path is a file inside a directory not just a file then we need to create a new directory)
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Directory {filedir} created")

    #creating the files inside the directory as well as the files without any directories in the filepath
    if (not os.path.exists(filepath)) or ( os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"empty file {filepath} created")

    else:
        logging.info(f"file {filepath} already exists")