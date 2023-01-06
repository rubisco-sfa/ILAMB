"""."""
import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ILAMB.Regions import Regions
from ILAMB.Variable import Variable
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_threshold_database(
    results_root: str,
    error_names: Union[str, List[str]],
    regions: List[str],
    thresholds: Union[float, List[float]],
    ref_name: str = "Benchmark",
    verbose: bool = True,
):
    """."""
    if isinstance(error_names, str):
        error_names = [error_names]
    if isinstance(thresholds, str):
        thresholds = [thresholds]
    ilamb_regions = Regions()

    # dictionaries reference [error_name][variable][region]
    quantiles = {}
    units = {}
    if verbose:
        print("Searching for errors...")
    for root, _, files in os.walk(results_root, followlinks=True):

        print(f"  {root}")
        # we just need to check netcdf files
        files = [f for f in files if (f.endswith(".nc") and not f.startswith(ref_name))]
        if not files:
            continue

        # harvest the non-null error values from the ILAMB intermediate files
        for error_name in error_names:
            for fname in files:

                dset = xr.open_dataset(os.path.join(root, fname), group="MeanState")

                # is there a dataarray in this dataset that matches the error name?
                var = [name for name in dset if f"{error_name}_map" in name]
                if not var:
                    continue
                assert len(var) == 1
                var = var[0]
                vname = var.split("_")[-1]
                var = Variable(
                    filename=os.path.join(root, fname),
                    variable_name=var,
                    groupname="MeanState",
                )

                # handle initialization of nested dictionary
                if error_name not in quantiles:
                    quantiles[error_name] = {}
                    units[error_name] = {}
                if vname not in quantiles[error_name]:
                    quantiles[error_name][vname] = {region: [] for region in regions}

                # handle units
                if vname not in units[error_name]:
                    units[error_name][vname] = var.unit
                if var.unit not in [
                    "K",
                    "degC",
                ]:  # are there other units we should not convert?
                    var = var.convert(units[error_name][vname])

                # append arrays
                for region in regions:
                    mask = ilamb_regions.getMask(region, var)
                    rvar = np.ma.masked_array(var.data, mask=mask).compressed()
                    quantiles[error_name][vname][region].append(rvar)

    # compute the quantiles across all models and datasets, for each variable
    V = []
    R = []
    P = []
    T = []
    E = []
    U = []
    # pylint: disable=consider-using-dict-items
    for error_name in quantiles:
        for vname in quantiles[error_name]:
            for region in quantiles[error_name][vname]:
                data = np.abs(np.hstack(quantiles[error_name][vname][region]))
                if data.size == 0:
                    continue
                data = np.percentile(data, thresholds)
                for i, th in enumerate(thresholds):
                    V.append(vname)
                    R.append(region)
                    P.append(thresholds[i])
                    T.append(error_name)
                    E.append(data[i])
                    U.append(units[error_name][vname])
    df_out = pd.DataFrame(
        {
            "variable": V,
            "region": R,
            "quantile": P,
            "type": T,
            "value": E,
            "unit": U,
        }
    )
    return df_out


def plot_quantiles(
    tdf: pd.DataFrame, error_type: str, error_value: float, plotname: str
):
    """."""
    assert error_type in tdf["type"].unique()
    assert error_value in tdf["quantile"].unique()
    dfr = tdf[(tdf["type"] == error_type) & (tdf["quantile"] == error_value)]
    ilamb_regions = Regions()
    variables = sorted(list(dfr["variable"].unique()), key=lambda key: key.lower())
    numv = len(variables)
    numr = int(round(np.sqrt(numv)))
    numc = int(round(numv / numr))
    while numr * numc < numv:
        numc += 1
    fig = plt.figure(figsize=(numr * 4, numc * 3), tight_layout=True, dpi=200)
    for i, vname in enumerate(variables):
        mask = []
        values = []
        for rname in dfr["region"].unique():
            lat = ilamb_regions._regions[rname][1]
            lon = ilamb_regions._regions[rname][2]
            value = dfr.loc[
                (dfr["variable"] == vname) & (dfr["region"] == rname), "value"
            ]
            if len(value) > 0:
                mask.append(ilamb_regions._regions[rname][3])
                values.append((mask[-1] == False) * float(value))
                unit = dfr.loc[
                    (dfr["variable"] == vname) & (dfr["region"] == rname), "unit"
                ]

        values = np.ma.masked_array(
            np.array(values).sum(axis=0), mask=np.array(mask).all(axis=0)
        )
        axs = fig.add_subplot(numr, numc, i + 1)
        img = axs.pcolormesh(lon[30:], lat, values[:, 30:])
        axs.axis("off")
        ax_divider = make_axes_locatable(axs)
        cax = ax_divider.append_axes("bottom", size="7%", pad="2%")
        fig.colorbar(
            img,
            cax=cax,
            orientation="horizontal",
            label=f"{vname} [{unit.iloc[0]}]",
        )

    fig.savefig(plotname)


# load the regions we will use to define biomes
regions = Regions()
for region_set in ["Whittaker", "Koppen"]:
    score_regions = regions.addRegionNetCDF4(
        f"/var/www/www.ilamb.org/html/ILAMB-Data/DATA/regions/{region_set}.nc"
    )

    # have we done this already?
    outname = f"quantiles_{region_set}_cmip5v6.pkl"
    if os.path.isfile(outname):
        df = pd.read_pickle(outname)
    else:
        df = create_threshold_database(
            "/var/www/www.ilamb.org/html/CMIP5v6/historical/",
            ["bias", "rmse"],
            score_regions,
            [50, 60, 70, 80, 90, 95, 98],
        )
        df.to_pickle(outname)

    plot_quantiles(df, "bias", 98, f"quantiles_{region_set}_bias98.png")
