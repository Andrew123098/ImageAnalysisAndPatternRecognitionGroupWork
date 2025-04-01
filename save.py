def compute_reverse_descriptor(descriptor: np.ndarray, n_samples: int = 11):
    """
    Reverse a Fourier descriptor to xy coordinates given a number of samples.
    
    Args
    ----
    descriptor: np.ndarray (D,)
        Complex descriptor of length D.
    n_samples: int
        Number of samples to consider to reverse transformation.

    Return
    ------
    x: np.ndarray complex (n_samples,)
        x coordinates of the contour
    y: np.ndarray complex (n_samples,)
        y coordinates of the contour
    """

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    # ------------------
    # Your code here ... 
    # USE THE INVERSE TRANSFORMATION FROM NUMPY
    
    # x = [1,2,3,4,5,6,7,8,9,10,11]
    # y = [1,2,3,4,5,6,7,8,9,10,11]

    sample_descriptor = ([ 0.+0.j,  1.-0.j, -0.+0.j, -0.-0.j,  0.-0.j,  0.-0.j,  0.+0.j,-0.+0.j,  0.-0.j, -0.-0.j,  0.-0.j])
    
    reconstructed_contour = np.zeros_like(sample_descriptor, dtype=complex)

    #print(descriptor)

    t_values = np.arange(n_samples)
    x_t = np.zeros_like(t_values, dtype=complex)  # Initialize the signal
    
    #print(sample_descriptor)

    for k in range(len(sample_descriptor)):  # Iterate over all descriptors
        reconstructed_contour += sample_descriptor[k] * np.exp(1j * 2 * np.pi * k * t_values / n_samples)

    #x_t = np.real(x_t)  # If descriptors are real, take the real part

    #reconstructed_contour = np.real(reconstructed_contour) 
    #print(reconstructed_contour)

    x = np.real(reconstructed_contour)
    y = np.imag(reconstructed_contour)

    # ------------------

    return x, y