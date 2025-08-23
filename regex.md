## Search with Regular Expressions

Regular expressions, or regex, all you to search text for specific strings. In this example we will search a Gaussian input file:
```
import re

sample = '''
%chk=myfile.chk
%mem=32GB
%nprocshared=32
# wB97XD def2-tzvpp opt scf=xqc

sample calc

0 1
H 0.0 0.0 1.4
H 0.0 0.0 0.0
'''

lines = sample.split("\n")
```
- First we can define how we search: 

| Character(s) | What they search for: |
| --- | --- |
| \d           | any number            |
| \D           | any non-number        |
| [wH]         | w and H               |
| [a-z]        | any lowercase letter  |
| \w           | any word              |
| +            | 1 or more             |
| *            | 0 or more             |
| ?            | 0 or 1                |
| {n}          | exactly n times       |
| \s           | any whitespace            |
| \S           | any non-whitespace        |
| ^            | at the begining of a line |


 
- Let's check each line for the presence of '0 1'. This will identify the start of the molecule specification.
- We will use re's match method, which return None is a match is not found, and returns a match object if a match is found. 
```
for i,line in enumerate(lines):
    m = re.search('\d', line)
    if m is not None:
        print(f"Match found in line {i}")

>>Match found in line 2
>>Match found in line 3
>>Match found in line 4
>>Match found in line 8
>>Match found in line 9
>>Match found in line 10
```
It found all the lines with any number in them. Let's be more specific and look for 1 or more numbers:
```
for i,line in enumerate(lines):
    m = re.search('\d+', line)
    if m is not None:
        print(f"Match found in line {i}")

>>Match found in line 2
>>Match found in line 3
>>Match found in line 4
>>Match found in line 8
>>Match found in line 9
>>Match found in line 10
```
It still found all lines containing numbers. Lets try to be even more specific:
```
for i,line in enumerate(lines):
    m = re.search('\d\s\d', line)
    if m is not None:
        print(f"Match found in line {i}")

>>Match found in line 8
>>Match found in line 9
>>Match found in line 10
``````
This did better, as it skipped the lines with no space between numbers. One more try:
```
for i,line in enumerate(lines):
    m = re.search('^\d\s\d', line)
    if m is not None:
        print(f"Match found in line {i}")

>>Match found in line 8
```
By specifying we wanted the begining of the line, we got it. 
- Lets look for wB97XD so we can replace it with B3LYP:
