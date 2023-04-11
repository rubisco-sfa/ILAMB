"""A class for abstracting and managing raw E3SM model results. While applicable
to global raw E3SM output, this implementation was meant for small domains such
as watersheds."""
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Union

import cftime
import pandas as pd
import xarray as xr

from . import ilamblib as il
from .Variable import Variable


@dataclass
class E3SMResult:
    """A class for abstracting and managing raw E3SM model results."""

    name: str = "E3SM"
    color: tuple[float] = (0, 0, 0)
    synonyms: dict = field(init=False, repr=False, default_factory=dict)
    files: pd.DataFrame = field(init=False, repr=False, default_factory=lambda: None)
    variables: set = field(init=False, repr=False, default_factory=set)

    def find_files(self, path: Union[str, list[str]], pattern: str = ".*elm.h0.*nc"):
        """Given a path or list of paths, find all files that match the given
        pattern."""
        model_files = []
        if isinstance(path, str):
            path = [path]
        for file_path in path:
            for root, _, files in os.walk(file_path, followlinks=True):
                for filename in files:
                    match = re.search(pattern, filename)
                    if not match:
                        continue
                    filepath = os.path.join(root, filename)
                    with xr.open_dataset(filepath) as dset:
                        model_files.append(
                            {
                                "tmin": dset["time"].min(),
                                "tmax": dset["time"].max(),
                                "path": filepath,
                            }
                        )
                        self.variables = self.variables.union(dset.variables)
        self.files = pd.DataFrame(model_files)
        return self

    def get_variable(
        self,
        vname: str,
        synonyms: Union[str, list[str]] = None,
        time0: str = None,
        timef: str = None,
    ):
        """Search the model database for the specified variable.

        Assumes by the nature of h0 files that all variables are in all files
        and that the variable you want is time dependent."""
        # Synonym handling, possibly move to a separate function
        possible = [vname]
        if isinstance(synonyms, str):
            possible.append(synonyms)
        elif isinstance(synonyms, list):
            possible += synonyms
        possible_syms = [p for p in possible if p in self.synonyms]
        possible += [var for syms in possible_syms for var in self.synonyms[syms]]
        found = [p for p in possible if p in self.variables]
        if len(found) == 0:
            raise il.VarNotInModel(
                f"Variable '{vname}' not found in model '{self.name}'"
            )
        found = found[0]

        # Figure out what files
        files = self.files
        if time0:
            files = files[files["tmin"] >= time0]
        if timef:
            files = files[files["tmax"] <= timef]

        # Uses pandas.to_xarray() to handle possibly unstructured grids
        used = ["time_bounds", "lat", "lon", "area", "landfrac", found]
        dset = [xr.open_dataset(filename)[used] for filename in files["path"]]
        dset = xr.concat(dset, dim="time")
        rem_attrs = {v: dset[v].attrs for v in used}
        time_bounds = dset[used.pop(0)]
        series = {"time": pd.Series(dset["time"]).repeat(dset.dims["lndgrid"])}
        series.update({v: dset[v].values.flatten() for v in used})
        dset = pd.DataFrame(series).set_index(["time", "lat", "lon"]).to_xarray()
        dset["time_bounds"] = time_bounds
        for var in rem_attrs:
            dset[var].attrs = rem_attrs[var]
        dset = dset.rename({found: vname})
        return dset

    def extractTimeSeries(self, vname: str, **keywords):
        """Get the input variable.

        The get_variable() routine will work well with ilamb3, for ilamb2x
        compatibility get just call that and then construct a Variable object.
        """
        time0 = keywords.get("initial_time", None)
        timef = keywords.get("final_time", None)
        time0 = (
            cftime.num2date(time0, "days since 1850-1-1", calendar="noleap")
            if time0
            else None
        )
        timef = (
            cftime.num2date(timef, "days since 1850-1-1", calendar="noleap")
            if timef
            else None
        )
        dset = self.get_variable(vname, time0=time0, timef=timef)

        ns_to_d = 1e-9 / 3600 / 24
        var = Variable(
            name=vname,
            data=dset[vname].to_numpy(),
            unit=dset[vname].attrs["units"].replace("gC", "g"),
            time=(dset["time"] - cftime.DatetimeNoLeap(1850, 1, 1)).values.astype(float)
            * ns_to_d,
            time_bnds=(
                dset["time_bounds"] - cftime.DatetimeNoLeap(1850, 1, 1)
            ).values.astype(float)
            * ns_to_d,
            lat=dset["lat"].to_numpy(),
            lon=dset["lon"].to_numpy(),
            area=(dset["area"] * 1e6 * dset["landfrac"]).isel({"time": 0}),
        )
        return var

    def add_synonym(self, elm_variable: str, other_variable: str):
        """Add synonyms, preference given to earlier definitions."""
        assert elm_variable in self.variables
        if other_variable not in self.synonyms:
            self.synonyms[other_variable] = []
        self.synonyms[other_variable].append(elm_variable)
        return self

    def to_pickle(self, filename: str):
        """Dumps the model result to a pickle file."""
        with open(filename, mode="wb") as pkl:
            pickle.dump(self.__dict__, pkl)

    def read_pickle(self, filename: str):
        """Initializes the model result from a pickle file."""
        with open(filename, mode="rb") as pkl:
            # pylint: disable=no-value-for-parameter
            obj = self.__new__(self.__class__)
            obj.__dict__.update(pickle.load(pkl))
        return obj
