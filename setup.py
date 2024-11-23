from setuptools import setup, find_packages

setup(
    name="st_dc",  # Your package name
    version="1.0.0",  # Package version
    description="Same Text, Different Context: A package for visualizing contextual word embeddings",
    long_description=open("README.md").read(),  # Use README.md for detailed description
    long_description_content_type="text/markdown",
    author="Suneel Nadipalli",  # Replace with your name
    author_email="nsuneel89@gmail.com",  # Replace with your email
    url="https://github.com/suneel-nadipalli/st-dc",  # GitHub repo URL
    packages=find_packages(),  # Automatically find all sub-packages
    install_requires=[
        "transformers>=4.0.0",
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "plotly>=5.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
