from setuptools import setup, find_packages

setup(
    name='DataAnalyzer',
    version='0.1',
    packages=find_packages(),
    description='A library for analyzing data influences in models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Billy Shi',
    author_email='lsa84@sfu.com',
    url='https://github.com/B1llySh1/DataAnalyeTool/tree/main',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
