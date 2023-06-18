# Regridding using ESMF library in python (xesmf)
# the new regrid script using the ESMF regridding framework

import numpy as np
import xarray as xr
import dask.array as da
import xesmf as xe
import time
import os
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import sys
import getopt

def get_args(argv):
    inputdir = ''
    outputdir = ''
    opts, _ = getopt.getopt(argv, "hi:o:",["idir=","odir="])
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <intputdir> -o <outputdir>')
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputdir = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg

    return inputdir, outputdir
    

# the main function to run
def main(inputdir, outputdir):
    with os.scandir(inputdir) as it:
        for entry in it:
            if entry.name.endswith('.nc') and entry.is_file() and not os.path.exists(outputdir + entry.name):
                print("Starting the Regridding of - " + entry.name + " ...")
                
                # loading the data
                ds = xr.open_dataset(entry.path, chunks=dict(time=2000))
                ds.unify_chunks()
                if 'nv' in ds.dims:
                    ds = ds.drop_dims('nv')
                ds = ds.transpose('time', 'lat', 'lon')

                print("Data is imported successfully ...")

                # prepare the regridding
                # TODO: change the lat and lon ranges according to your dataset
                ds_out = xr.Dataset({'lat': (['lat'], np.arange( 7.02, 19.06, 0.04)),
                    'lon': (['lon'], np.arange(66.47, 77.51, 0.04)),
                    }
                )
                regridder = xe.Regridder(ds, ds_out, 'bilinear')

                print("Data has been successfully prepared for regridding ...")

                # Do the regridding, change the name of variable
                dr_out = regridder(ds.chlor_a)

                print("The data has been regridded, now exporting ...")

                # Ooutput the data
                # TODO: change the name of the variable according to your dataset
                dr_out = dr_out.to_dataset(name = 'chlor_a') 
                delayedObj = dr_out.to_netcdf(outputdir + "/" + entry.name, compute=False)

                with ProgressBar():
                    results = delayedObj.compute()

                print(results)
                print("Finished Regridding and data exported " + entry.name)
                time.sleep(5)

if __name__ == "__main__":
    inputdir, outputdir = get_args(sys.argv[1:])
    main(inputdir, outputdir)
    print("Finished the Regridding process")
