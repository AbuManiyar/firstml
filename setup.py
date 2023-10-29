from setuptools import setup, find_packages
from typing import List


HED = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [i.replace('\n', '') for i in requirements]
        if HED in requirements:
            requirements.remove(HED)
    print(requirements,'\n','\n','\n','\n')        
    return requirements


setup(
    name='first',
    version= '0.0.1',
    author= 'abutalha',
    author_email= 'maniyarabutalha00@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )