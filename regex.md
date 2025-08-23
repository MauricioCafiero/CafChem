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
| \            | used before any special character |


- How would we specify a regex for a post code? For example: RG6 6AH
```
'\D\D\d\d*\s\d\D\D'
```
- Let's check each line in the sample text for the presence of '0 1'. This will identify the start of the molecule specification.
- We will use re's match method, which return None is a match is not found, and returns a match object if a match is found. If there are multiple matches, it only returns the first.
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
- In each case we got a match object. The match object contains some information about the search, so let's start saving them:
```
m = []
for i,line in enumerate(lines):
    m.append(re.search('^\d\s\d', line))
    if m[-1] is not None:
        print(f"Match found in line {i}")
>> Match found in line 8
```
Now we have an array of match objects. Let's look at the .group() and .span() methods for this object:
```
m[8].group()

>> '0 1'
```
The group() method returned the match. Now let's look at span():
```
m[8].span()

.. (0, 3)
```
span() tells us the match starts at position 0 and ends at position 3. 
- If we want to find more than one occurance of a string, we can use re's findall function. It returns a list of all the occurances, or an empty list of there are none.
- Let's look for every time '0.0' appears in a line:
```
for i,line in enumerate(lines):
    m = re.findall('0\.0', line)
    if len(m) != 0:
        print(f"found the string in line {i}")
        for result in m:
            print(result)

>> found the string in line 9
>> 0.0
>> 0.0
>> found the string in line 10
>> 0.0
>> 0.0
>> 0.0
```
It found '0.0' twice in line 9 and three times in line 10.

- Lets look for the amount of memory specified in the file, and change it to 64 GB. We'll pretend we don't know what the current memory specification is (it's 32). Thus we will have to allow for anywhere from 1-3 digits for the memory and either MB or GB for the units. We will also specify only 1 substitution. ONce we have made the substitution we will print out the new file.
- re's sub method allows you to substitute something for a regex one or more times:
```
new_lines = []
for i,line in enumerate(lines):
    new_line = re.sub('mem=\d\d*\d*[GM]B', 'mem=64GB', line, 1)
    new_lines.append(new_line)

new_file = "\n".join(new_lines)
print(new_file)

>> %chk=myfile.chk
>> %mem=64GB
>> %nprocshared=32
>> # wB97XD def2-tzvpp opt scf=xqc
>> 
>> sample calc
>> 
>> 0 1
>> H 0.0 0.0 1.4
>> H 0.0 0.0 0.0
```
- Finally, we can search a line for cartesian coordinates, and save each one as an x, y or z value.
- We will use re's split function, which returns a list of parts of the string split at the specified point. In this case we will split at a space.
```
x = []
y = [] 
z = []

for i,line in enumerate(lines):
    m = re.search('^\D\D*\s\d\d*\.\d+\s\d\d*\.\d+\s\d\d*\.\d+', line)
    if m is not None:
        parts = re.split('\s', line)
        x.append(parts[1])
        y.append(parts[2])
        z.append(parts[3])

print(x)
print(y)
print(z)

>> ['0.0', '0.0']
>> ['0.0', '0.0']
>> ['1.4', '0.0']
```
First we checked for a match. In order to capture any coordinates, we specified the line has to start with a non-number, followed by a possible second non-number (element names, 1-2 letters). Then a space, and then 1-2 numbers, a full stop, and and number of numbers following the decimal. This pattern is then repeated twice more with a space in between.
