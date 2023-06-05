# GitHub Repository

https://github.com/nightmare224/brane-programming-project



# BRANE Experience

## Brane Package

- When developing package in Python, it seems that developer always has to write the boilerplate to select function as well as to read the variable from environment variables. Although Brane is not just for Python developer, I think it would be nice if there is Brane Python package to reduce the boilerplate. For example, maybe can develop a Python Decorator to let developer expose the function.

  ```python
  @brane_function
  def label_encoding(df: pd.DataFrame, name: str):
    pass
  ```

- I think when building the image with same tag can overwrite the old one. Or we will have to run `docker image rm` to delete the old one manually.

## Brane Script

- It would be nice if we can also pass array as parameter to Python as many machine learning related function would be more make sense if we can have array parameter. Currently my workaroud is to write `func(["'apple'", "'banana'"])` to pass the array to the Python function in package. And in Python I do `eval(varaible)` to make it become Python list. I have to do quote `"'apple'"` two times because Brane seems remove the double quote when passing variable into function.



# Member Responsibilities Summary

| Name               | Work Description                                             |
| ------------------ | ------------------------------------------------------------ |
| Hsiang-ling Tai    | - BRANE environment setup<br />- BRANE packages and scripts<br />- Visualization<br />- Report writing |
| Yung-sheng Tu      | - Machine Learning model training<br />- Data preprocessing<br />- Data analysis<br />- Visualization<br />- Report writing |
| Vishwamitra Mishra | - Machine Learning model training<br />- Report writing      |



