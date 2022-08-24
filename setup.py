import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {'qa': 'transformers',
          'summarization': 'transformers',
          'unsupervised': 'sentence_transformers'}

extras['all'] = list(set(extras.values()))

install_requires = ["sentence_transformers"]
setuptools.setup(
    name="nfmodelapis",
    version="0.0.1",
    author="Abhijith Neil Abraham",
    author_email="abhijithneilabrahampk@gmail.com",
    description="Data Curation over Time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    url="https://github.com/nfflow/nfmodelapis",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    extras_require=extras,
    python_requires='>=3.6',
    include_package_data=True
)
