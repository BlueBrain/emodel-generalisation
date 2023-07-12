#!/bin/bash -l

# get data from harvard dataverse at https://doi.org/10.7910/DVN/FGWMUF
wget -nc https://dataverse.harvard.edu/api/access/datafile/7239342 -O cadpyr_l5.zip
unzip cadpyr_l5.zip
