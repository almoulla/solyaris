from setuptools import find_packages, setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='solyaris',
    version='0.2.0',
    author='Khaled Al Moulla',
    author_email='khaled.almoulla@gmail.com',
    description='Order-by-order radial-velocity extraction and differential-extinction correction of Sun-as-a-star observations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/almoulla/solyaris',
    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['astropy>=8.0.1'     ,
                      'barycorrpy>=0.4.4'  ,
                      'iCCF>=0.4.15'       ,
                      'numpy>=2.5.2'       ,
                      'pandas>=3.0.5'      ,
                      'PyAstronomy>=0.25.0',
                      'scipy>=1.18.1'      ,
                      'tqdm>=4.70.0'       ],
    classifiers=["Development Status :: 1 - Planning"    ,
                 "Intended Audience :: Science/Research" ,
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3"   ]
)