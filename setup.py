import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mvseg3d",
    version="0.0.1",
    author="darrenwang",
    author_email="wangyang9113@gmail.com",
    description="mvsegmentation3d",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boundles/mvsegmentation3d",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6"
)
