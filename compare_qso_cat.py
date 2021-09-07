#!/usr/bin/env python3
import fitsio
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Plot QSO catalogs')
parser.add_argument('-i1',
                    '--input1', help='Input QSO catalog',required=True)
parser.add_argument('-i2',
                    '--input2', help='Input QSO catalog',required=False)
parser.add_argument('-legend', help='legend', default=["1","2"],required=False)

args = vars(parser.parse_args())
try:
    ra1 = fitsio.FITS(args["input1"])[1]["TARGET_RA"][:]
    dec1 = fitsio.FITS(args["input1"])[1]["TARGET_DEC"][:]
except:
    ra1 = fitsio.FITS(args["input1"])[1]["RA"][:]
    dec1 = fitsio.FITS(args["input1"])[1]["DEC"][:]

try:
    ra2 = fitsio.FITS(args["input2"])[1]["TARGET_RA"][:]
    dec2 = fitsio.FITS(args["input2"])[1]["TARGET_DEC"][:]
except:
    ra2 = fitsio.FITS(args["input2"])[1]["RA"][:]
    dec2 = fitsio.FITS(args["input2"])[1]["DEC"][:]

plt.figure(figsize=(10,10))
plt.scatter(ra1,dec1,s=1,alpha=0.5)
plt.scatter(ra2,dec2,s=1,alpha=0.5)
plt.legend(args["legend"])
plt.show()
