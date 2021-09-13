from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='single_view_depth',
      version='0.0.2',
      author='Garima Samvedi',
      author_email='g.samvedi@qut.edu.au',
      description='Unsupervised CNN for Single View Depth Estimation.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['torch', 'torchvision', 'numpy', 'requests', 'pillow', 'opencv-python', 'matplotlib'],
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))