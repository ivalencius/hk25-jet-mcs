"""
Extract U and V after the jet detection algorithm (1_xshield.ipynb)
"""
## Standard Stuff
import numpy as np
import xarray as xr

## HEALPix Specific
import healpix as hp
import easygems.healpix as egh
import easygems.remap as egr

import intake     # For catalogs
import zarr

# Ilan
from icecream import ic
import nc_time_axis

# set up dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

def load_XSHIELD():
    # catfn='/home/tmerlis/hackathon/xsh24_scream_main.yaml'
    catfn='/home/tmerlis/hackathon/hackathon_cat_may14_main.yaml'
    
    combo_cat = intake.open_catalog(catfn)
    
    # select zoom level and the part of the combined catalog you're interested in
    # coarse stores are available at zoom 7 ~50km and lower
    zoom_select = 7 # Wind speeds are messed up for zoom 7 and less
    ds = combo_cat.xsh24_coarse(zoom=zoom_select).to_dask().drop_vars('cell')
    # attach coordinates; otherwise can't use lat and lon and selecting regions or taking a zonal mean won't work
    # geopotential height (basically height)
    ds = ds.pipe(egh.attach_coords).assign(height=lambda x: x['zg']) 
    return ds

def get_uvs(c_us, c_vs, c_heights, jet_height):
    """
    Input: one cell of data, dim is [plev] except for jet_height which is a scalar
    Output: dataset with 2 fields
        1. u [time, cell] -> u wind at the jet core
        2. v [time, cell] -> v wind at the jet core
    """
    if np.isnan(jet_height):
        return (np.nan, np.nan)
    # Need increasing heights for np.interp to work
    good_heights = c_heights >= 0
    c_heights, c_us, c_vs = c_heights[good_heights], c_us[good_heights], c_vs[good_heights]
    sort_idx = np.argsort(c_heights)
    c_heights, c_us, c_vs = c_heights[sort_idx], c_us[sort_idx], c_vs[sort_idx]
    # Now get u and v
    u = np.interp(jet_height, c_heights, c_us)
    v = np.interp(jet_height, c_heights, c_vs)
    return (u,v)
    
    
def apply_uvs(us, vs, heights, jet_heights):
    """
    llj -> mask, speed, height [time, cell]
    """
    u, v = xr.apply_ufunc(
        get_uvs,
        us.chunk({'plev':-1}),
        vs.chunk({'plev':-1}),
        heights.chunk({'plev': -1}),
        jet_heights,
        input_core_dims=[['plev'], ['plev'], ['plev'], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    return xr.merge([
        u.rename('jet_u'),
        v.rename('jet_v')
    ])
    
def main():
    llj = xr.open_dataset("/scratch/cimes/iv4111/hk25-data/llj_XSHIELD.h5")
    xshield = load_XSHIELD()
    xshield = xshield.reindex_like(llj)
    uv = apply_uvs(
        xshield.ua,
        xshield.va,
        xshield.height,
        llj.height
    )
    # need to copy attributes so healpix doesn't mess up
    uv['crs'].attrs = llj['crs'].attrs.copy()
    uv.to_netcdf("/scratch/cimes/iv4111/hk25-data/llj_XSHIELD_uv.h5")

if __name__ == '__main__':
    main()
    