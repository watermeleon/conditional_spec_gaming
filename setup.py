from setuptools import setup, find_packages

setup(
    name="conditional-spec-gaming",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    author="Anonymous",
    author_email="",
    description="Investigating harmful misalignment when training LLMs via RL with LLM-as-a-judge rewards",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
