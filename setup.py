from setuptools import find_packages, setup

setup(name='crying_cat_generator',
      version='0.1.0.dev',
      author="Iago Martinelli Lopes",
      description="Crying cat generator app",
      long_description=open("README.md", "r").read(),
      long_description_content_type="text/markdown",
      keywords="crying cat generator",
      package_dir={"": "src"},
      packages=find_packages("src"),
      python_requires='>=3.6',
      license='MIT',
      )