import setuptools

with open("README.md", "r",encoding="utf-8") as f:
    long_description = f.read()

version="0.0.0"

REPO_NAME="Semantic_Segmentation_for_self_driving_cars"
AUTHOR_USER_NAME="rsmeghana8"
SRC_REPO="src"
AUTHOR_EMAIL="meghana.reppala@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=version,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A samll python package for semantic segmentation of self driving cars",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"))