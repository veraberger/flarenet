import numpy as np

"""
how to cite the cosmic ray mitigation tutorial? 
"""


def load_cosmicray_extension(tpf):
    """Function to load the cosmic ray extension into a datacube
    
    Parameters:
    -----------
    tpf: lk.TargetPixelFile
        Input Target Pixel File
    
    Returns:
    -------
    cr: np.ndarray
        Array containing cosmic ray fluxes of shape tpf.shape
    """
    cadenceno = np.asarray(tpf.cadenceno)

    # Get cosmic ray information
    l = np.where([hdu.name == 'TARGET COSMIC RAY' for hdu in tpf.hdu])[0]
    if len(l) != 1:
        raise ValueError("TPF product has no cosmic ray extension.")
    hdu = tpf.hdu[l[0]]
    c, x, y, f = [hdu.data[attr].copy() for attr in ['CADENCENO', 'RAWX', 'RAWY', 'COSMIC_RAY']]
    if len(c) == 0:
        raise ValueError("No cosmic rays identified. Is this a 20-second dataset?")
    x -= tpf.column
    y -= tpf.row
    
    # Mask down to only cosmic rays in cadences within the TPF
    k = np.in1d(c, cadenceno)
    c, x, y, f = c[k], x[k], y[k], f[k]
        
    # Map cosmic ray cadences to array indices
    r = {cadenceno[idx]:idx for idx in range(tpf.shape[0])}
    inv = np.asarray([r[c1] for c1 in c])
    
    # Build a cube
    cr = np.zeros(tpf.shape)
    cr[inv, y, x] = f
    return cr

