import numpy as np


def proj_time(x, masks, data_clipped):
    
    proj = np.copy(x)
    # Debugging Checks
    assert len(x) == len(data_clipped), "Signal length mismatch"
    assert len(masks['Mr']) == len(x), "Mask 'Mr' length mismatch"
    assert len(masks['Mh']) == len(x), "Mask 'Mh' length mismatch"
    assert len(masks['Ml']) == len(x), "Mask 'Ml' length mismatch"

    proj[masks['Mr']] = data_clipped[masks['Mr']]
    proj[masks['Mh']] = np.maximum(x[masks['Mh']], data_clipped[masks['Mh']])
    proj[masks['Ml']] = np.minimum(x[masks['Ml']], data_clipped[masks['Ml']])
    
    return proj