#!/bin/bash
if [ $# -eq 0 ]; then
    files="*py misc/*py models/*py models/perceptual/*py datasets/*.py"
else
    files=$1
fi
#for i in *py misc/*py models/*py models/perceptual/*py datasets/*.py;
for i in $files
do
    echo "Formating $i"
    yapf -i $i
    autopep8 --in-place -a $i
    yapf -i $i
    flake8 $i
    echo "---"
done
