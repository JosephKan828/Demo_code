# This code is for demonstrating how to apply space-time filter as Wheeler and Kiladis (1999)
# Import package
import numpy as np;
import netCDF4 as nc;
from matplotlib import pyplot as plt;

def main(  ):
    # Load data;
    fpath = "/work/DATA/Satellite/OLR/olr_anomaly.nc";
    
    with nc.Dataset( fpath, "r" ) as ds:
        dims = { 
                key: ds.variables[key][...]
                for key in ds.dimensions.keys()
                }; # load dimensions;

        lat_lim = np.where( ( dims["lat"] >= -15.0 ) & ( dims["lat"] <= 15.0 ) )[0];

        olr = ds["olr"][:1000, lat_lim, :].mean( axis=1 );
    
    # apply 2-dimensional FFT on the data;
    olr_fft = np.fft.fft( olr, axis=0 );
    olr_fft = np.fft.ifft( olr_fft, axis=-1 ) * olr.shape[-1];

    # setup axes and domain of reconstructing
    wn = np.fft.fftfreq( olr_fft.shape[1], d= )

    mask = np.where( 
            (  )
            )


if __name__ == "__main__":
    main();
