from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="distributed-rl-llm",
    version="0.1.0",
    author="NAMEEEEE PLEASE BRO",
    author_email="lmaohellnah@gmail.com",
    description="Distributed Reinforcement Learning system for training large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnthonyJi123/DistributedML",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-llm=scripts.train:main",
            "eval-llm=scripts.evaluate:main",
            "monitor-llm=scripts.monitor:main",
            "example-llm=scripts.example:main",
        ],
    },
) 