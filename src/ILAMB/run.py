from .ModelResult import ModelResult
import os
import pickle
from mpi4py import MPI
import logging
from . import ilamblib as il
from traceback import format_exc

logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)

def InitializeModels(model_root,models=[],verbose=False,filter="",regex="",model_year=[],log=True,models_path="./",comm=MPI.COMM_WORLD):
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
    if len(model_year) != 2: model_year = None
    max_model_name_len = 0
    if rank == 0 and verbose: print("\nSearching for model results in %s\n" % model_root)
    for subdir, dirs, files in os.walk(model_root):
        for mname in dirs:
            if len(models) > 0 and mname not in models: continue
            pkl_file = os.path.join(models_path,"%s.pkl" % mname)
            if os.path.isfile(pkl_file):
                with open(pkl_file,'rb') as infile:
                    m = pickle.load(infile)
            else:
                try:
                    m = ModelResult(os.path.join(subdir,mname), modelname = mname, filter=filter, regex=regex, model_year = model_year)
                except Exception as ex:
                    if log: logger.debug("[%s]" % mname,format_exc())
                    continue
            M.append(m)
            max_model_name_len = max(max_model_name_len,len(mname))
        break
    M = sorted(M,key=lambda m: m.name.upper())
    
    # assign unique colors
    clrs = il.GenerateDistinctColors(len(M))
    for m in M:
        clr     = clrs.pop(0)
        m.color = clr

    # save model objects as pickle files
    comm.Barrier()
    if rank == 0:
        for m in M:
            pkl_file = os.path.join(models_path,"%s.pkl" % m.name)
            with open(pkl_file,'wb') as out:
                pickle.dump(m,out,pickle.HIGHEST_PROTOCOL)
        
    # optionally output models which were found
    if rank == 0 and verbose:
        for m in M:
            print(("    {0:>45}").format(m.name))

    if len(M) == 0:
        if verbose and rank == 0: print("No model results found")
        comm.Barrier()
        comm.Abort(0)

    return M

def ParseModelSetup(model_setup,models=[],verbose=False,filter="",regex="",models_path="./",comm=MPI.COMM_WORLD):
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
    rank = comm.rank
    # initialize the models
    M = []
    max_model_name_len = 0
    if rank == 0 and verbose: print("\nSetting up model results from %s\n" % model_setup)
    with open(model_setup) as f:
        for line in f.readlines():
            if line.strip().startswith("#"): continue
            line       = line.split(",")
            mname      = None
            mdir       = None
            model_year = None
            if len(line) >= 2:
                mname  = line[0].strip()
                mdir   = line[1].strip()
                # if mdir not a directory, then maybe path is relative to ILAMB_ROOT
                if not os.path.isdir(mdir):
                    mdir = os.path.join(os.environ["ILAMB_ROOT"],mdir).strip()
            if len(line) == 4:
                model_year = [float(line[2].strip()),float(line[3].strip())]
            max_model_name_len = max(max_model_name_len,len(mname))
            if (len(models) > 0 and mname not in models) or (mname is None): continue
            pkl_file = os.path.join(models_path,"%s.pkl" % mname)
            if os.path.isfile(pkl_file):
                with open(pkl_file,'rb') as infile:
                    m = pickle.load(infile)
            else:
                try:
                    m = ModelResult(mdir, modelname = mname, filter=filter, regex=regex, model_year = model_year)
                except Exception as ex:
                    if log: logger.debug("[%s]" % mname,format_exc())
                    continue
            M.append(m)

    # assign unique colors
    clrs = il.GenerateDistinctColors(len(M))
    for m in M:
        clr     = clrs.pop(0)
        m.color = clr

    # save model objects as pickle files
    comm.Barrier()
    if rank == 0:
        for m in M:
            pkl_file = os.path.join(models_path,"%s.pkl" % m.name)
            with open(pkl_file,'wb') as out:
                pickle.dump(m,out,pickle.HIGHEST_PROTOCOL)
                
    # optionally output models which were found
    if rank == 0 and verbose:
        for m in M:
            print(("    {0:>45}").format(m.name))

    if len(M) == 0:
        if verbose and rank == 0: print("No model results found")
        comm.Barrier()
        comm.Abort(0)

    return M
