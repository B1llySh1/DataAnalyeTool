from setuptools import find_packages, setup

setup(
    name='datainfluence',
    packages=find_packages(include=['datainfluence']),
    version='0.0.0',
    description='Data influence analyzer',
    # Absolutely needed packages
    install_requires=['numpy'],
    # Second needed packages
    setup_requires=['pandas'],
    author='Billy',
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)