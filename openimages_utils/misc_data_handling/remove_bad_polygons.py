list = [
"[1062, 1007, 1062, 1069], ",
"[328, 475, 329, 475], ",
"[752, 422, 753, 422], ",
"[1557, 1199, 1561, 1199], ",
"[558, 386, 559, 386], ",
"[1069, 1073, 1070, 1073], ",
"[793, 408, 799, 408], ",
"[1039, 572, 1040, 573], ",
"[1456, 1064, 1464, 1064], ",
]

infile = 'dirtyfile.txt'
outfile = 'cleanedfile.txt'

fin = open(infile)
fout = open(outfile, "w+")
for line in fin:
    for word in list:
        line = line.replace(word, "")
    fout.write(line)
fin.close()
fout.close()