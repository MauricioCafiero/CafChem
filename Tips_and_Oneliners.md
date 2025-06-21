### Getting all of a particular file-type in a directory and making a list
```
import os

path = "CSV_files/"
files = os.listdir(path)            
filenames = [file for file in files if (os.path.splitext(file)[1]==".csv")]

print(filenames)
```
