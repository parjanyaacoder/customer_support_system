from setuptools import find_packages, setup

setup(name="customer_chat_bot",
    version="0.0.1",
    author="parjanya",
    author_email="padityashukla26@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb', 'langchain']
)