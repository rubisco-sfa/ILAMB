#!/usr/bin/env python
# Turn cmec.json parameters into ilamb config file format

import json
import os
import subprocess
import sys

def write_ilamb_cfg(param_file,cfdict,lev=0):
    """Recursive function that writes ILAMB configuration file
    from dictionary in cmec.json.
    Each nested dictionary is treated as a new sub section (h1, h2,
    or a data source)

    Parameters
    ----------
    param_file: str
        open parameter file to write to
    cfdict: str
        dictionary with the configuration file contents
    lev: int
        counter that keeps track of the heading level

    """
    head = {0: "h1: ", 1: "h2: ", 2: ""}
    for key in cfdict:
        if isinstance(cfdict[key], str):
            param_file.write("{0} = \"{1}\"\n".format(key, cfdict[key]))
        elif isinstance(cfdict[key], (int,float,bool)):
            param_file.write("{0} = {1}\n".format(key, cfdict[key]))
        elif isinstance(cfdict[key], dict):
            param_file.write("[{0}{1}]\n".format(head[lev],key))
            write_ilamb_cfg(param_file,cfdict[key],lev=lev+1)
    return

if __name__ == '__main__':

    module = sys.argv[1]

    # Get CMEC environment variables
    code_dir = os.getenv("CMEC_CODE_DIR",default=None)
    config_dir = os.getenv("CMEC_CONFIG_DIR",default=None)
    obs_data = os.getenv("CMEC_OBS_DATA",default=None)
    model_data = os.getenv("CMEC_MODEL_DATA",default=None)
    wk_dir = os.getenv("CMEC_WK_DIR",default=None)

    if None in [config_dir, wk_dir, obs_data, model_data]:
        print("Error: A CMEC environment variable is missing")
        sys.exit(1)

    os.chdir(wk_dir)

    # Get ILAMB user settings from cmec config
    config_file = config_dir + "/cmec.json"
    try:
        with open(config_file) as cf:
            cfdict = json.load(cf)[module]
    except json.decoder.JSONDecodeError:
        print("Error: could not read " + config_file + ". File might not be valid JSON.")
        sys.exit(1)

    # Load CMIP confrontation info for custom configuration
    custom_keys = cfdict.pop("confrontations",None)

    if custom_keys is not None:
        # load cmip config
        try:
            with open(config_file) as cf:
                cmip = json.load(cf)["ILAMB/CMIP"]["cfg"]
        except KeyError:
            print("ILAMB/CMIP settings missing from cmec.json.")
            print("Loading CMIP obs information from ILAMB/cmec-driver/cmip.json.")
            with open(code_dir + "/cmip.json","r") as cf:
                cmip = json.load(cf)["default_parameters"]["cfg"]
        # Might need to initialize config dictionary if using custom
        if "cfg" not in cfdict:
            cfdict["cfg"] = {}
        for key in custom_keys:
            # Check H1 keys
            if key in cmip:
                cfdict["cfg"].update(cmip[key])
            else:
                # Check H2 keys
                for ckey in cmip:
                    if key in cmip[ckey]:
                        if ckey not in cfdict["cfg"]:
                            cfdict["cfg"].update({ckey: {}})
                        cfdict["cfg"][ckey].update({key:cmip[ckey][key]})

    # Populate ilamb.cfg file in CMEC_WIK_DIR
    param_file_name = os.path.join(wk_dir,"ilamb.cfg")
    with open(param_file_name, "w") as param_file:
        write_ilamb_cfg(param_file,cfdict["cfg"])

    # Generate ilamb-run command
    # Any variables that aren't "cfg" are treated as cmd line options
    cfdict.pop("cfg")
    base_cmd = ["ilamb-run", "--config", param_file_name, "--model_root", model_data, "--build_dir", wk_dir]
    for cmd_var in cfdict:
        base_cmd.append("--{0}".format(cmd_var))
        if isinstance(cfdict[cmd_var],list):
            for item in cfdict[cmd_var]:
                base_cmd.append(item)
            #base_cmd.append(cfdict[cmd_var])
        elif isinstance(cfdict[cmd_var],str):
            base_cmd.append("{0}".format(cfdict[cmd_var]))

    obs_dir_name = os.path.basename(obs_data)
    # Check if obs dir is present in curent directory or if
    # it needs to get linked here for ILAMB
    create_link=False
    if not os.path.isdir(obs_dir_name):
        os.symlink(obs_data,obs_dir_name)
    # Add ILAMB_ROOT to env and run
    myenv = os.environ.copy()
    myenv['ILAMB_ROOT'] = wk_dir
    subprocess.run(base_cmd, env=myenv)
    # Remove symlink if created
    if os.path.islink(obs_dir_name):
        os.unlink(obs_dir_name)
