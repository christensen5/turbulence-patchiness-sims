#!/bin/bash
cd $1

find . -regextype "egrep" \
       -iregex './vor/in/vor_mm_[0-9][0-9][0-9].txt' \
       -exec voro++ -o -px -py -c "%v" 0 600 0 600 0 300 {} \;
       
cd ./vor/in

mv *.vol ../out
