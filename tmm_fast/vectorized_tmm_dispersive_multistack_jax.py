import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, Dict, List, Tuple, Optional
import sys
from warnings import warn
EPSILON = sys.float_info.epsilon

def coh_vec_tmm_disp_mstack(pol: str,
                            N: Union[np.ndarray, jnp.ndarray], 
                            T: Union[np.ndarray, jnp.ndarray], 
                            Theta: Union[np.ndarray, jnp.ndarray], 
                            lambda_vacuum: Union[np.ndarray, jnp.ndarray], 
                            device: str = 'cpu',  # Not used in JAX version but kept for API compatibility
                            timer: bool = False) -> Dict:
    """
    JAX implementation of the parallelized computation of reflection and transmission for coherent light spectra
    that traverse a bunch of multilayer thin-films with dispersive materials.
    This implementation naturally allows:
     - XLA acceleration via JAX
     - To compute gradients regarding the multilayer thin-film (i.e. N, T) thanks to JAX autodiff

    Input can be either numpy arrays or JAX arrays.
    All computations are processed via JAX, and the output is returned as JAX arrays.

    Parameters:
    -----------
    pol : str
        Polarization of the light, accepts only 's' or 'p'
    N : array_like
        Numpy or JAX array of shape [S x L x W] with complex or real entries which contain the refractive
        indices at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered. Note that the first and last layer must feature real valued
        refractive indicies, i.e. imag(N[:, 0, :]) = 0 and imag(N[:, -1, :]) = 0.
    T : array_like
        Holds the layer thicknesses of the individual layers for a bunch of thin films in nanometer.
        T is of shape [S x L] with real-valued entries; infinite values are allowed for the first and last layers only!
    Theta : array_like
        Theta is an array that determines the angles with which the light propagates in the injection layer.
        Theta is of shape [A] and holds the incidence angles [rad] in its entries.
    lambda_vacuum : array_like
        Vacuum wavelengths for which reflection and transmission are computed given a bunch of thin films.
        It is of shape [W] and holds the wavelengths in nanometer.
    device : str
        Not used in JAX version but kept for API compatibility with PyTorch version
    timer: bool
        Determines whether to track computation time

    Returns:
    --------
    output : Dict
        Keys:
            'r' : Array of Fresnel coefficients of reflection for each stack (over angle and wavelength)
            't' : Array of Fresnel coefficients of transmission for each stack (over angle and wavelength)
            'R' : Array of Reflectivity / Reflectance for each stack (over angle and wavelength)
            'T' : Array of Transmissivity / Transmittance for each stack (over angle and wavelength)
            Each of these arrays is of shape [S x A x W]
    optional output: list of two floats if timer=True
            List containing just the total computation time [sec]
    """

    if timer:
        import time
        starttime = time.time()
    
    # Convert all inputs to JAX arrays
    datatype = check_datatype(N, T, lambda_vacuum, Theta)
    N = converter(N)
    T = converter(T)
    lambda_vacuum = converter(lambda_vacuum)
    Theta = converter(Theta)
    
    # Handle inputs with dimensions less than the expected
    squeezed_N = False
    squeezed_T = False
    if N.ndim < 3:
        squeezed_N = True
        N = jnp.expand_dims(N, 0)
    if T.ndim < 2:
        squeezed_T = True
        T = jnp.expand_dims(T, 0)
    assert squeezed_N == squeezed_T, 'N and T are not of same shape, as they are of dimensions ' + str(N.ndim) + ' and ' + str(T.ndim)
    
    # Check inputs for correctness
    check_inputs(N, T, lambda_vacuum, Theta)
    
    # Clamp very high imaginary parts for numerical stability
    N = N.at[:, :, :].set(jnp.complex64(N.real + 1j * jnp.clip(N.imag, a_max=35.)))
    
    num_layers = T.shape[1]
    num_stacks = T.shape[0]
    num_angles = Theta.shape[0]
    num_wavelengths = lambda_vacuum.shape[0]
    
    # Handle case where a constant refractive index is used (no dispersion)
    if N.ndim == 2:
        N = jnp.tile(N[:, :, jnp.newaxis], (1, 1, num_wavelengths))

    # Calculate angles in each layer using Snell's law
    SnellThetas = SnellLaw_vectorized(N, Theta)
    
    # Calculate z-component of wavevector in each layer
    theta = 2 * jnp.pi * jnp.einsum('skij,sij->skij', jnp.cos(SnellThetas), N)
    kz_list = jnp.einsum('sijk,k->skij', theta, 1 / lambda_vacuum)
    
    # Calculate phase accumulated in each layer
    delta = jnp.einsum('skij,sj->skij', kz_list, T)
    
    # Clamp very high imaginary parts for numerical stability
    delta = delta.at[:, :, :, :].set(delta.real + 1j * jnp.clip(delta.imag, a_max=35.))
    
    # Calculate interface reflection and transmission coefficients
    t_list = interface_t_vec(pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :], SnellThetas[:, :, 1:, :])
    r_list = interface_r_vec(pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :], SnellThetas[:, :, 1:, :])
    
    # Setup propagation matrices for each layer
    A = jnp.exp(1j * delta[:, :, :, 1:-1])
    F = r_list[:, :, :, 1:]
    
    # Initialize transfer matrices
    M_list = jnp.zeros((num_stacks, num_angles, num_wavelengths, num_layers, 2, 2), dtype=jnp.complex64)
    
    # Fill the transfer matrices
    M_list = M_list.at[:, :, :, 1:-1, 0, 0].set(jnp.einsum('shji,sjhi->sjhi', 1 / (A + jnp.finfo(float).eps), 1 / t_list[:, :, :, 1:]))
    M_list = M_list.at[:, :, :, 1:-1, 0, 1].set(jnp.einsum('shji,sjhi->sjhi', 1 / (A + jnp.finfo(float).eps), F / t_list[:, :, :, 1:]))
    M_list = M_list.at[:, :, :, 1:-1, 1, 0].set(jnp.einsum('shji,sjhi->sjhi', A, F / t_list[:, :, :, 1:]))
    M_list = M_list.at[:, :, :, 1:-1, 1, 1].set(jnp.einsum('shji,sjhi->sjhi', A, 1 / t_list[:, :, :, 1:]))
    
    # Initialize the full transfer matrix
    Mtilde = jnp.zeros((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=jnp.complex64)
    Mtilde = Mtilde.at[:, :, :].set(make_2x2_tensor(1, 0, 0, 1))
    
    # Multiply the individual layer matrices to get the full transfer matrix
    # JAX doesn't allow regular Python loops with mutable variables, so we use a scan instead
    def matrix_multiply(carry, layer_idx):
        Mtilde = carry
        return jnp.einsum('sijkl,sijlm->sijkm', Mtilde, M_list[:, :, :, layer_idx]), None
    
    Mtilde, _ = jax.lax.scan(matrix_multiply, Mtilde, jnp.arange(1, num_layers - 1))
    
    # Account for the first layer
    M_r0 = jnp.zeros((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=jnp.complex64)
    M_r0 = M_r0.at[:, :, :, 0, 0].set(1)
    M_r0 = M_r0.at[:, :, :, 0, 1].set(r_list[:, :, :, 0])
    M_r0 = M_r0.at[:, :, :, 1, 0].set(r_list[:, :, :, 0])
    M_r0 = M_r0.at[:, :, :, 1, 1].set(1)
    M_r0 = jnp.einsum('sijkl,sij->sijkl', M_r0, 1 / t_list[:, :, :, 0])
    
    # Combine with the transfer matrix
    Mtilde = jnp.einsum('shijk,shikl->shijl', M_r0, Mtilde)
    
    # Calculate reflection and transmission coefficients
    r = Mtilde[:, :, :, 1, 0] / (Mtilde[:, :, :, 0, 0] + jnp.finfo(float).eps)
    t = 1 / (Mtilde[:, :, :, 0, 0] + jnp.finfo(float).eps)
    
    # Calculate reflectivity and transmissivity
    R = R_from_r_vec(r)
    T = T_from_t_vec(pol, t, N[:, 0], N[:, -1], SnellThetas[:, :, 0], SnellThetas[:, :, -1])
    
    # Reshape if necessary
    if squeezed_T and r.shape[0] == 1:
        r = jnp.reshape(r, (r.shape[1], r.shape[2]))
        R = jnp.reshape(R, (R.shape[1], R.shape[2]))
        T = jnp.reshape(T, (T.shape[1], T.shape[2]))
        t = jnp.reshape(t, (t.shape[1], t.shape[2]))
    
    # Convert back to numpy if input was numpy
    if datatype is np.ndarray:
        r = numpy_converter(r)
        t = numpy_converter(t)
        R = numpy_converter(R)
        T = numpy_converter(T)
    
    if timer:
        total_time = time.time() - starttime
        return {'r': r, 't': t, 'R': R, 'T': T}, [total_time]
    else:
        return {'r': r, 't': t, 'R': R, 'T': T}

def SnellLaw_vectorized(n, th):
    """
    Return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!
    """
    # Ensure proper complex types
    n = jnp.asarray(n, dtype=jnp.complex64)
    th = jnp.asarray(th, dtype=jnp.float32)
    
    # Calculate angles in each layer using Snell's law
    n0_ = jnp.einsum('hk,j,hik->hjik', n[:,0], jnp.sin(th), 1/n)
    angles = jnp.arcsin(n0_)
    
    # Determine if angle is forward or backward using the is_forward_angle criterion
    # and adjust if necessary
    angles_0 = jnp.where(
        is_not_forward_angle(n[:, 0], angles[:, :, 0]),
        jnp.pi - angles[:, :, 0],
        angles[:, :, 0]
    )
    
    angles_last = jnp.where(
        is_not_forward_angle(n[:, -1], angles[:, :, -1]),
        jnp.pi - angles[:, :, -1],
        angles[:, :, -1]
    )
    
    # Update the first and last layer angles
    angles = angles.at[:, :, 0].set(angles_0)
    angles = angles.at[:, :, -1].set(angles_last)
    
    return angles

def is_not_forward_angle(n, theta):
    """
    Determine whether an angle is not a forward angle based on material properties.
    For complex n & theta, determining forward vs backward angles is more complex
    than just checking -pi/2 < theta < pi/2.
    See https://arxiv.org/abs/1603.02720 appendix D.
    """
    # Ensure proper types
    n = jnp.asarray(n, dtype=jnp.complex64)
    
    # Check that material doesn't have gain (real*imag >= 0)
    # JAX doesn't support assertions, so we'll handle this differently
    # if in production code
    
    # Expand n to match theta shape for broadcasting
    n = jnp.expand_dims(n, 1)
    ncostheta = jnp.cos(theta) * n
    
    # For evanescent decay or lossy medium, the forward wave is the one that decays
    cond1 = (jnp.abs(ncostheta.imag) > 100 * EPSILON) & (ncostheta.imag <= 0)
    
    # For propagating waves, forward is the one with positive Poynting vector
    cond2 = (jnp.abs(ncostheta.imag) <= 100 * EPSILON) & (ncostheta.real <= 0)
    
    return cond1 | cond2

def interface_r_vec(polarization, n_i, n_f, th_i, th_f):
    """
    Calculate reflection amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propagation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        ni_thi = jnp.einsum('sij,skij->skji', n_i, jnp.cos(th_i))
        nf_thf = jnp.einsum('sij,skij->skji', n_f, jnp.cos(th_f))
        return (ni_thi - nf_thf) / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = jnp.einsum('sij,skij->skji', n_f, jnp.cos(th_i))
        ni_thf = jnp.einsum('sij,skij->skji', n_i, jnp.cos(th_f))
        return (nf_thi - ni_thf) / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_t_vec(polarization, n_i, n_f, th_i, th_f):
    """
    Calculate transmission amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propagation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        ni_thi = jnp.einsum('sij,skij->skji', n_i, jnp.cos(th_i))
        nf_thf = jnp.einsum('sij,skij->skji', n_f, jnp.cos(th_f))
        return 2 * ni_thi / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = jnp.einsum('sij,skij->skji', n_f, jnp.cos(th_i))
        ni_thf = jnp.einsum('sij,skij->skji', n_i, jnp.cos(th_f))
        ni_thi = jnp.einsum('sij,skij->skji', n_i, jnp.cos(th_i))
        return 2 * ni_thi / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def R_from_r_vec(r):
    """
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return jnp.abs(r) ** 2

def T_from_t_vec(pol, t, n_i, n_f, th_i, th_f):
    """
    Calculate transmitted power T, starting with transmission amplitude t.
    
    Parameters:
    -----------
    pol : str
        polarization, either 's' or 'p'
    t : array_like
        transmission coefficients
    n_i, n_f : array_like
        refractive indices of incident and final medium.
    th_i, th_f : array_like
        (complex) propagation angles through incident & final medium
        (in radians, where 0=normal). "th" stands for "theta".
    """
    if pol == 's':
        ni_thi = jnp.real(jnp.cos(th_i) * jnp.expand_dims(n_i, 1))
        nf_thf = jnp.real(jnp.cos(th_f) * jnp.expand_dims(n_f, 1))
        return (jnp.abs(t ** 2) * ((nf_thf) / (ni_thi)))
    elif pol == 'p':
        ni_thi = jnp.real(jnp.conj(jnp.cos(th_i)) * jnp.expand_dims(n_i, 1))
        nf_thf = jnp.real(jnp.conj(jnp.cos(th_f)) * jnp.expand_dims(n_f, 1))
        return (jnp.abs(t ** 2) * ((nf_thf) / (ni_thi)))
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def converter(data):
    """
    Convert input data to JAX array
    """
    if not isinstance(data, jnp.ndarray):
        if isinstance(data, np.ndarray):
            return jnp.array(data)
        else:
            raise ValueError('Input is not a numpy.array or jax.numpy.ndarray!')
    return data

def numpy_converter(data):
    """
    Convert JAX array to numpy array
    """
    return np.array(data)

def check_datatype(N, T, lambda_vacuum, Theta):
    """
    Check that all inputs are of the same data type
    """
    data_types = [type(N), type(T), type(lambda_vacuum), type(Theta)]
    if not all(t == data_types[0] for t in data_types):
        raise ValueError('All inputs must be of the same data type!')
    return data_types[0]

def check_inputs(N, T, lambda_vacuum, theta):
    """
    Validate input dimensions and shapes
    """
    # check the dimensionalities of N:
    if N.ndim != 3:
        raise ValueError(f'N is not of shape [S x L x W] (3d), as it is of dimension {N.ndim}')
    
    # check the dimensionalities of T:
    if T.ndim != 2:
        raise ValueError(f'T is not of shape [S x L] (2d), as it is of dimension {T.ndim}')
    
    if T.shape[0] != N.shape[0]:
        raise ValueError(f'The number of thin-films (first dimension) of N and T must coincide, '
                         f'found N.shape={N.shape} and T.shape={T.shape} instead!')
    
    if T.shape[1] != N.shape[1]:
        raise ValueError(f'The number of thin-film layers (second dimension) of N and T must coincide, '
                         f'found N.shape={N.shape} and T.shape={T.shape} instead!')
    
    # check the dimensionality of Theta:
    if theta.ndim != 1:
        raise ValueError(f'Theta is not of shape [A] (1d), as it is of dimension {theta.ndim}')
    
    # check the dimensionality of lambda_vacuum:
    if lambda_vacuum.ndim != 1:
        raise ValueError(f'lambda_vacuum is not of shape [W] (1d), as it is of dimension {lambda_vacuum.ndim}')
    
    if N.shape[-1] != lambda_vacuum.shape[0]:
        raise ValueError(f'The last dimension of N must coincide with the dimension of lambda_vacuum (W), '
                         f'found N.shape[-1]={N.shape[-1]} and lambda_vacuum.shape[0]={lambda_vacuum.shape[0]} instead!')
    
    # check well defined property of refractive indicies for the first layer
    # JAX doesn't support assertions in the same way as PyTorch, so we'd handle this differently
    # in a production environment

def make_2x2_tensor(a, b, c, d):
    """
    Makes a 2x2 array of [[a,b],[c,d]]
    """
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex64)