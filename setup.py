from setuptools import setup, find_packages

setup(
    name='KmerTokenizer',
    version='0.1.0',
    packages=find_packages(),
    description='A package for k-mer tokenization.',
    author='Saleh Refahi',
    author_email='sr3622@drexel.edu',
    url='https://github.com/MsAlEhR/KmerTokenizer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'transformers',  # Add other dependencies here
    ],
    python_requires='>=3.6',
)
