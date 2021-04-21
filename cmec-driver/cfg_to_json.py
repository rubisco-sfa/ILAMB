"""
cfg_to_json.py

This script writes the contents of an ILAMB configuration file
to the default parameters section of a cmec-driver settings json.

Usage
-----
    python cfg_to_json.py <filename> <settings_file>

Parameters
----------
    filename: str
        path to ILAMB config file to import
    settings_file: str
        path to cmec settings JSON to write config to

"""

import json
import os
import re
import sys

filename = sys.argv[1]
settings_file = sys.argv[2]

my_json = {}
# Get command line arguments
for line in open(filename).readlines():
    m1 = re.search(r"#!(.*)=(.*)",line)
    if m1:
        keyword = m1.group(1).strip()
        value   = m1.group(2).strip().replace('"','')
        my_json.update({keyword: value})

# Get obs configurations
cfg = {}
for line in open(filename).readlines():
    line = line.strip()
    if line.startswith("#"): continue
    m1 = re.search(r"\[h(\d):\s+(.*)\]",line)
    m2 = re.search(r"\[(.*)\]",line)
    m3 = re.search(r"(.*)=(.*)",line)
    if m1:
        level1 = m1.group(1)
        if int(level1) < 2:
            name1 = m1.group(2)
            cfg.update({name1: {}})
            name2 = ""
        elif int(level1) < 3:
            name2 = m1.group(2)
            cfg[name1].update({name2: {}})
        name3 = ""

    if not m1 and m2:
        name3 = m2.group(1)
        cfg[name1][name2].update({name3: {}})

    if m3:
        keyword = m3.group(1).strip()
        value   = m3.group(2).strip().replace('"','')
        if value.lower() in ["true","false"]:
            value = bool(value)
        elif value.isnumeric():
            try:
                value = int(value)
            except ValueError:
                pass
            try:
                value = float(value)
            except ValueError:
                pass

        if name3:
            cfg[name1][name2][name3].update({keyword: value})
        elif name2:
            cfg[name1][name2].update({keyword: value})
        else:
            cfg[name1].update({keyword: value})

my_json.update({"cfg": cfg})

# Write these settings to specified settings json
with open(settings_file,"r") as setfilename:
    config = json.load(setfilename)

config["default_parameters"] = my_json
with open(settings_file,"w") as setfilename:
    json.dump(config,setfilename,indent=2)
