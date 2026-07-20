from setuptools import setup, find_packages

setup(
    name="chanmerge",                            
    version="1.1.0",                             
    author="Ahmet Sercan Kıyak & Berra Dülgerbaki",
    author_email="a.sercankyk@gmail.com",
    description="Automated Chandra X-ray Data Pipeline and Merger",
    long_description=open('README.md').read(),   
    long_description_content_type="text/markdown",
    packages=["chanmerge"],                    
    install_requires=[                           
        "astropy",
        "astroquery"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.12',
)
