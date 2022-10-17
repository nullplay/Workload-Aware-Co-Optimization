import sys
import re

if __name__ == "__main__":
    f = open(sys.argv[1])
    s = f.read()
    f.close()

    f = open(sys.argv[1], "w")
    apos = re.findall("A_vals\[(.*?)\]", s)[0]
    substitute_C_vals = re.sub("C_vals\[(.*?)\]", 'C_vals[{}]'.format(apos), s)
    commentout_zero = re.sub("C_vals\[.*\] = 0.0;", "", substitute_C_vals)
    f.write(commentout_zero)
    f.close()

