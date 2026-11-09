from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chanmerge",
    version="2.0.0",
    author="Ahmet Sercan Kıyak & Berra Dülgerbaki",
    author_email="a.sercankyk@gmail.com",
    description="Chandra search, reprocessing and time-grouped merged flux images with documented corrections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["chanmerge"],
    install_requires=["astropy", "astroquery", "numpy", "matplotlib", "scipy"],
    entry_points={"console_scripts": ["chanmerge=chanmerge.cli:main", "chanmerge-gui=chanmerge.gui:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
)
