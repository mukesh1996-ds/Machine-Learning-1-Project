import os 
from setuptools import setup,find_packages
from typing import List


## Create Variable for setup file 
PROJCT_NAME="Machine Learning 1"
VERSION="0.0.1"
DESCRIPTION="This Is Our Machine Learning Project 1"
AUTHOR_NAME="Mukesh Kumar"
AUTHOR_EMAIL="mks.mukesh1996@gamil.com"
REQUIREMENTS_FILE_NAME='requirements.txt'
HYPHEN_E_DOT="-e ."

## Locate the requirements.txt file and read it 
def get_requirements_list()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as reqirement_file:
        reqirement_list=reqirement_file.readlines()
        reqirement_list=[requirement_name.replace("\n","") for requirement_name in reqirement_list]

        if HYPHEN_E_DOT in reqirement_list:
            reqirement_list.remove()
            return reqirement_list

setup(name=PROJCT_NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR_NAME,
      author_email=AUTHOR_EMAIL,
      packages=find_packages(),
      install_requirements=get_requirements_list()
     )

