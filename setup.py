import setuptools

setuptools.setup(
    name="puffins",
    version="0.1",
    packages=setuptools.find_packages(),
    author="Spencer A. Hill",
    author_email="shill@ldeo.columbia.edu",
    description="Functions for computing things in climate science",
    install_requires=[
        "eofs",
        "faceted",
        "gitpython",
        "ipython",
        "matplotlib",
        "numpy",
        "scipy",
        "scikit-learn",
        "xarray",
    ],
    scripts=["puffins/scripts/set_proj_puff_branch.py"],
    license="Apache",
    keywords="climate science",
    url="https://github.com/spencerahill/puffins",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ]
)
