# Crying Cat Generator

Welcome to Crying Cat Generator.

![Alt Text](https://raw.githubusercontent.com/maicken/crying_cat_generator/main/crying_cat.gif)

## 1. Setup

If you wish to run this algo in your local machine, you can start by git cloning this repo and running the following commands inside the repo folder.

```
pip install -r requirements.txt
pip install -e .
```

## 2. Run

To create a crying cat from a cat image, you can run the following command:

```
!python src/crying_cat_generator.py -f PATH_TO_THE_FOLDER_WITH_CAT_IMAGES
```

The results will be saved in the folder "results" as default, you can change this option by passing as argument the output folder with the argument -o.

You can also select only one image by using -i followed by the path to the image.

## 3. Colab demo

A simple demo was created using colab. You don't need to install in your local machine, you just need to open the file "crying_cat_generator.ipynb" in the google colab and follow colab instructions.


