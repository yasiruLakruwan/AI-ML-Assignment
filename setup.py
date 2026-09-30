from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="AI/ML Assignment",
    version="0.1",
    author="Yasiru Lakruwan",
    install_requires=requirements,
    packages=find_packages()
)


