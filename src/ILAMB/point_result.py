"""A class for abstracting and managing point model results in the form of
observation point data."""

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
from dataretrieval import nwis

from ILAMB import ilamblib as il


def is_binary_file(filename: str) -> bool:
    """Tests if the file is binary, as opposed to text."""
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    with open(filename, "rb") as fin:
        return bool(fin.read(1024).translate(None, textchars))


@dataclass
class ModelPointResult:
    """A class for abstracting and managing point model results."""

    name: str = "none"
    color: tuple[float] = (0, 0, 0)
    synonyms: dict = field(init=False, repr=False, default_factory=dict)
    variables: pd.DataFrame = field(
        init=False, repr=False, default_factory=lambda: None
    )
    sites: pd.DataFrame = field(init=False, repr=True, default_factory=lambda: None)
    origin: pd.Timestamp = field(
        init=False,
        repr=False,
        default_factory=lambda: pd.Timestamp("1980-01-01 00:00:00"),
    )

    def find_files(self, path: Union[str, list[str]], file_to_site: dict[str] = None):
        """Given a path or list of paths, find all"""
        if isinstance(path, str):
            path = [path]
        model_data = []
        site_data = []
        for file_path in path:
            for root, _, files in os.walk(file_path, followlinks=True):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if is_binary_file(filepath):
                        continue
                    site = pd.NA
                    if file_to_site:
                        site = (
                            file_to_site[filename]
                            if filename in file_to_site
                            else pd.NA
                        )
                    if site:
                        rec = nwis.get_record(site=site, service="site")
                        assert len(rec) == 1
                        rec = rec.iloc[0]
                        site_data.append(
                            {
                                "sitecode": site,
                                "name": rec["station_nm"],
                                "lat": rec["dec_lat_va"],
                                "lon": rec["dec_long_va"],
                                "huc8": f"{rec['huc_cd']:08d}",
                                "path": filepath,
                            }
                        )
                    dfv = pd.read_csv(filepath, comment="#", nrows=0)
                    for key in dfv.columns:
                        match = re.search(r"(.*)\s\[(.*)\]", key)
                        if not match:
                            continue
                        model_data.append(
                            {
                                "variable": match.group(1),
                                "unit": match.group(2),
                                "column": key,
                                "sitecode": site,
                            }
                        )
        self.variables = pd.DataFrame(model_data)
        self.sites = pd.DataFrame(site_data).set_index("sitecode")
        return self

    def get_variable(
        self,
        vname: str,
        sitecode: str,
        synonyms: Union[str, list[str]] = None,
        frequency: str = "ununsed",
    ):
        """Search the model database for the specified variable.

        At the moment, the entire csv file is read and only the requested column
        is returned. We may want to cache read in dataframes to trade memory for
        time."""
        dfv = self.variables[(self.variables["sitecode"] == sitecode)]
        if len(dfv) == 0:
            raise ValueError("The given sitecode is not part of this model result.")

        # Synonym handling, possibly move to a separate function
        possible = [vname]
        if isinstance(synonyms, str):
            possible.append(synonyms)
        elif isinstance(synonyms, list):
            possible += synonyms
        possible_syms = [p for p in possible if p in self.synonyms]
        possible += [var for syms in possible_syms for var in self.synonyms[syms]]
        found = [p for p in possible if p in dfv["variable"].unique()]
        if len(found) == 0:
            raise ValueError(f"Variable '{vname}' not found in model '{self.name}'")
        found = found[0]
        dfv = dfv[dfv["variable"] == found].iloc[0]

        # Process csv file
        dfo = pd.read_csv(self.sites.loc[sitecode, "path"], comment="#")
        time_col = [c for c in dfo.columns if ("time" in c or "date" in c)]
        time_col = time_col[0]
        new_time_col = "time"
        try:
            dfo[new_time_col] = pd.to_datetime(
                dfo[time_col], unit="D", origin=self.origin
            )
        except (TypeError, ValueError):
            dfo[new_time_col] = pd.to_datetime(dfo[time_col])
        dfo = dfo.rename(columns={dfv["column"]: vname})
        dfo = dfo.set_index(new_time_col)
        dfo = dfo.tz_localize(None)
        dfo = dfo.groupby(pd.Grouper(freq="D")).mean(numeric_only=True)
        dfo.attrs["unit"] = dfv["unit"]
        dfo = dfo[vname]
        return dfo

    def add_synonym(self, ats_variable: str, other_variable: str):
        """Add synonyms, preference given to earlier definitions."""
        assert ats_variable in self.variables["variable"].unique()
        if other_variable not in self.synonyms:
            self.synonyms[other_variable] = []
        self.synonyms[other_variable].append(ats_variable)

    def to_pickle(self, filename: str):
        """."""
        with open(filename, mode="wb") as pkl:
            pickle.dump(self.__dict__, pkl)

    def read_pickle(self, filename: str):
        """."""
        with open(filename, mode="rb") as pkl:
            # pylint: disable=no-value-for-parameter
            obj = self.__new__(self.__class__)
            obj.__dict__.update(pickle.load(pkl))
        return obj

    def extractTimeSeries(self, *args, **kwargs):
        """."""
        raise il.VarNotInModel(
            "The point model object does not yet handle gridded output."
        )
