import logging
import os
import pickle
from traceback import format_exc
import yaml
import inspect

from mpi4py import MPI

from ILAMB import ilamblib as il
from ILAMB.ModelResult import ModelResult
from matplotlib import colors as mplclrs

logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)
rank = 0

def InitializeModels(
    model_root,
    models=[],
    verbose=False,
    filter="",
    regex="",
    model_year=[],
    log=True,
    models_path="./",
    comm=MPI.COMM_WORLD,
):
    """Initializes a list of models

    Initializes a list of models where each model is the subdirectory
    beneath the given model root directory. The global list of models
    will exist on each processor.

    Parameters
    ----------
    model_root : str
        the directory whose subdirectories will become the model results
    models : list of str, optional
        only initialize a model whose name is in this list
    verbose : bool, optional
        enable to print information to the screen
    model_year : 2-tuple, optional
        shift model years from the first to the second part of the tuple

    Returns
    -------
    M : list of ILAMB.ModelResults.ModelsResults
       a list of the model results, sorted alphabetically by name

    """
    rank = comm.rank
    # initialize the models
    M = []
    if len(model_year) != 2:
        model_year = None
    max_model_name_len = 0
    if rank == 0 and verbose:
        print("\nSearching for model results in %s\n" % model_root)
    for subdir, dirs, files in os.walk(model_root):
        for mname in dirs:
            if len(models) > 0 and mname not in models:
                continue
            pkl_file = os.path.join(models_path, "%s.pkl" % mname)
            if os.path.isfile(pkl_file):
                with open(pkl_file, "rb") as infile:
                    m = pickle.load(infile)
            else:
                try:
                    m = ModelResult(
                        os.path.join(subdir, mname),
                        modelname=mname,
                        filter=filter,
                        regex=regex,
                        model_year=model_year,
                    )
                except Exception as ex:
                    if log:
                        logger.debug("[%s]" % mname, format_exc())
                    continue
            M.append(m)
            max_model_name_len = max(max_model_name_len, len(mname))
        break
    M = sorted(M, key=lambda m: m.name.upper())

    # assign unique colors
    clrs = il.GenerateDistinctColors(len(M))
    for m in M:
        clr = clrs.pop(0)
        m.color = clr

    # save model objects as pickle files
    comm.Barrier()
    if rank == 0:
        for m in M:
            pkl_file = os.path.join(models_path, "%s.pkl" % m.name)
            with open(pkl_file, "wb") as out:
                pickle.dump(m, out, pickle.HIGHEST_PROTOCOL)

    # optionally output models which were found
    if rank == 0 and verbose:
        for m in M:
            print(("    {0:>45}").format(m.name))

    if len(M) == 0:
        if verbose and rank == 0:
            print("No model results found")
        comm.Barrier()
        comm.Abort(0)

    return M


def _parse_model_yaml(filename: str, cache_path: str = "./", only_models: list = []):
    """Setup models using a yaml file."""
    model_classes = {
        "ModelResult": ModelResult,
    }
    models = []
    with open(filename, encoding="utf-8") as fin:
        yml = yaml.safe_load(fin)
    for name, opts in yml.items():
        # optionally filter models
        if len(only_models) > 0 and name not in only_models:
            continue

        if "name" not in opts:
            opts["name"] = name

        # if the model_year option is given, convert to lits of floats
        if "model_year" in opts:
            opts["model_year"] = [
                float(y.strip()) for y in opts["model_year"].split(",")
            ]

        # select the class type
        cls = model_classes[opts["type"]] if "type" in opts else ModelResult
        if cls is None:
            typ = opts["type"]
            raise ValueError(f"The model type '{typ}' is not available")
        fcns = dir(cls)

        # if the pickle file exists, just load it
        cache = os.path.join(cache_path, f"{name}.pkl")
        if os.path.exists(cache):
            if "read_pickle" in fcns:
                model = cls().read_pickle(cache)
            else:
                with open(cache, mode="rb") as fin:
                    model = pickle.load(fin)
            models.append(model)
            continue

        # call the constructor using keywords defined in the YAML file
        cls = model_classes[opts["type"]] if "type" in opts else ModelResult
        model = cls(
            **{
                key: opts[key]
                for key in inspect.getfullargspec(cls).args
                if key in opts
            }
        )

        # some model types have a find_files() method, call if present loading
        # proper keywords from the YAML file
        if "find_files" in fcns:
            model.find_files(
                **{
                    key: opts[key]
                    for key in inspect.getfullargspec(model.find_files).args
                    if key in opts
                }
            )

        # some model types allow you to specify snynonms
        if "add_synonym" in fcns and "synonyms" in opts:
            for mvar, syn in opts["synonyms"].items():
                model.add_synonym(mvar, syn)

        # cache the model result
        if rank == 0:
            if "read_pickle" in fcns:
                model.to_pickle(cache)
            else:
                with open(cache, mode="wb") as fin:
                    pickle.dump(model, fin)

        models.append(model)

    for model in models:
        if isinstance(model.color, str) and model.color.startswith("#"):
            model.color = mplclrs.hex2color(model.color)
    return models

def ParseModelSetup(
        model_setup, models=[], verbose=False, filter="", regex="", models_path="./", comm=MPI.COMM_WORLD
):
    """Initializes a list of models

    Initializes a list of models where each model is the subdirectory
    beneath the given model root directory. The global list of models
    will exist on each processor.

    Parameters
    ----------
    model_setup : str
        the directory whose subdirectories will become the model results
    models : list of str, optional
        only initialize a model whose name is in this list
    verbose : bool, optional
        enable to print information to the screen

    Returns
    -------
    M : list of ILAMB.ModelResults.ModelsResults
       a list of the model results, sorted alphabetically by name

    """
    if rank == 0 and verbose:
        print("\nSetting up model results from %s\n" % model_setup)

    # intercept if this is a yaml file
    if model_setup.endswith(".yaml"):
        M = _parse_model_yaml(model_setup, cache_path=models_path, only_models=models)
        if rank == 0 and verbose:
            for m in M:
                print(("    {0:>45}").format(m.name))
            if len(M) == 0:
                print("No model results found")
                comm.Barrier()
                comm.Abort(0)
        return M

    # initialize the models
    M = []
    max_model_name_len = 0
    with open(model_setup) as f:
        for line in f.readlines():
            if line.strip().startswith("#"):
                continue
            line = line.split(",")
            mname = None
            mdir = None
            model_year = None
            mgrp = ""
            if len(line) >= 2:
                mname = line[0].strip()
                mdir = line[1].strip()
                # if mdir not a directory, then maybe path is relative to ILAMB_ROOT
                if not os.path.isdir(mdir):
                    mdir = os.path.join(os.environ["ILAMB_ROOT"], mdir).strip()
                if len(line) == 3:
                    mgrp = line[2].strip()
            if len(line) == 4:
                model_year = [float(line[2].strip()), float(line[3].strip())]
            max_model_name_len = max(max_model_name_len, len(mname))
            if (len(models) > 0 and mname not in models) or (mname is None):
                continue
            pkl_file = os.path.join(models_path, "%s.pkl" % mname)
            if os.path.isfile(pkl_file):
                with open(pkl_file, "rb") as infile:
                    m = pickle.load(infile)
            else:
                try:
                    m = ModelResult(
                        mdir,
                        modelname=mname,
                        filter=filter,
                        regex=regex,
                        model_year=model_year,
                        group=mgrp,
                    )
                except Exception as ex:
                    logger.debug("[%s]" % mname, format_exc())
                    continue
            M.append(m)

    # assign unique colors
    clrs = il.GenerateDistinctColors(len(M))
    for m in M:
        m.color = clrs.pop(0)

    # save model objects as pickle files
    comm.Barrier()
    if rank == 0:
        for m in M:
            pkl_file = os.path.join(models_path, "%s.pkl" % m.name)
            with open(pkl_file, "wb") as out:
                pickle.dump(m, out, pickle.HIGHEST_PROTOCOL)

    # optionally output models which were found
    if rank == 0 and verbose:
        for m in M:
            print(("    {0:>45}").format(m.name))

    if len(M) == 0:
        if verbose and rank == 0:
            print("No model results found")
        comm.Barrier()
        comm.Abort(0)

    return M
