#!/usr/bin/env -S grimaldi --kernel bento_kernel_arvr
# fmt: off

""":md
### Use arvr (v2213) kernel
"""

""":py"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

""":md
### Check GPU availabiilty
"""

""":py"""
# some gpu testing code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_cached(0) / 1024**3, 1), "GB")

""":md
### Fresnel diffraction functions
"""

""":py"""
def circ(x, y):
    return np.where(np.sqrt(x**2 + y**2) <= 0.5, 1, 0)


def propForward(u1, L1, x1, wavelength, z, dx1, X, Y, FX, FY, method):
    """
    propagation - Fraunhofer pattern
    assumes uniform sampling
    u1 - source plane field
    L1 - source plane side length
    lambda - wavelength
    z - propagation distance
    L2 - observation plane side length
    u2 - observation plane field
    """
    k = 2 * math.pi / wavelength
    sampling_criteria = wavelength * z / (dx1 * L1)
    L2 = L1
    x2 = x1

    if z == 0:
        u2 = u1
    else:
        if not method:
            if sampling_criteria >= 1:
                method = "IR"
            else:
                method = "TF"
        # print(
        #     "sampling criteria (lambda*z/(dx*L)) = "
        #     + str(round(sampling_criteria, 4))
        #     + ". Using method "
        #     + method
        # )

        if method == "IR":
            # x1 = -L1/2:dx1:L1/2-dx1
            h = (
                1
                / (1j * wavelength * z)
                * torch.exp(1j * k / (2 * z) * (X**2 + Y**2))
            )
            U2 = fft.fft2(fft.fftshift(h)) * dx1**2 * fft.fft2(fft.fftshift(u1))
            u2 = fft.ifftshift(fft.ifft2(U2))
        elif method == "TF":
            # fx=-1/(2*dx1):1/L1:1/(2*dx1)-1/L1
            H = torch.exp(-1j * math.pi * wavelength * z * (FX**2 + FY**2))
            U2 = fft.fftshift(H) * fft.fft2(fft.fftshift(u1))
            u2 = fft.ifftshift(fft.ifft2(U2))
            
        elif method == "RS":
            # fx=-1/(2*dx1):1/L1:1/(2*dx1)-1/L1
            complex_argument = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
            complex_argument = complex_argument.type(torch.complex64)  # Convert to complex type
            H = torch.exp(
                1j
                * k
                * z
                * torch.sqrt(complex_argument)
            )
            U2 = fft.fftshift(H) * fft.fft2(fft.fftshift(u1))
            u2 = fft.ifftshift(fft.ifft2(U2))

    return u2, L2, x2


def scaled_FFT(u1, L1, x1, wavelength, z):
    dx1 = x1[1] - x1[0]
    fx1 = torch.linspace(-1 / (2 * dx1), 1 / (2 * dx1), steps=len(x1))
    x2 = fx1 * wavelength * z
    dx2 = x2[1] - x2[0]
    L2 = len(x2) * dx2
    u2 = fft.ifftshift(fft.fft2(fft.fftshift(u1)))

    return u2, L2, x2

""":md
### Phase mask generating functions
"""

""":py"""
@torch.jit.script
def SDF(pos: torch.Tensor, center: torch.Tensor, radius: float):
    """A signed-distance function (SDF) for a level set composed of an array of circles.

    Returns a scalar value for each grid point corresponding to the distance
    from the current point to the nearest interface. Consequently, a distance
    of `0` means the point lies on the interface. Positive values lie outside
    the circle and negative values lie inside the circle.

    As a performance optimization, we could batch (or window) the loop. This
    "quasi-vectorization" helps mitigate the memory issues of traditional broadcasting.

    [1] https://math.mit.edu/classes/18.086/2006/am57.pdf

    Args:
        pos: (Nx2) x-y pairs for each grid point
        center: (1x2) x-y emitter center
        radii: emitter radius

    Returns:
        d:
    """
    return (
        torch.sqrt((pos[:, 0] - center[0]) ** 2 + (pos[:, 1] - center[1]) ** 2) - radius
    )


@torch.jit.script
def sphere_volume_average(d: torch.Tensor, r: float) -> torch.Tensor:
    """Computes the volume average assuming a spherical convolution kernel.

    From a signed distance function, we must project the distance to the nearest interface
    to an actual weight value from 0 to 1 (where 0 corresponds to a point outside the circle
    and 1 corresponds to a shape inside the circle). This is particularly challenging for pixels
    that contain an "intersection" of the circle's contour. We get around this by performing an
    "average", such that these contour values smoothly range from 0 to 1. As a result, we can
    smoothly move the circle across the discrete domain and expect a continuos change (no snapping).

    To make things easy, we use a spherical averaging kernel, rather than the actual cube. The error introduced
    is first order at worst, and should be negligible compared to all other approximations.

    Importantly, this smoothing is essential for computing the gradient. These smoothed regions are the
    *only* pixels that will actually contribute to the gradient w.r.t. the circle centers (classic shape optimization).

    Args:
        d: distance from the current point to the nearest edge
        r: radius of current averaging sphere

    returns
    """
    return torch.where(
        d < 0.0,
        1.0,
        torch.where(
            d > r,
            0.0,
            1.0
            / (torch.pi * r**2)
            * (r**2 * torch.acos(d / r) - d * torch.sqrt(r**2 - d**2)),
        ),
    )


@torch.jit.script
def restrict_amplitude_and_phase(
    amplitude_batch: torch.Tensor,
    phase_batch: torch.Tensor,
    pos: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    phases: torch.Tensor,
    smoothing_sphere_radius: float,
) -> None:
    """Restricts (or projects) the geometry onto a discretized amplitude and phase mask.

    Args:
        pos: (Nx2) x-y pairs for each grid point
        centers: (Mx2) x-y emitter center
        radii: (Mx1) emitter radii
        phases: (Mx1) emitter phases
        smoothing_sphere_radius: essentially Δx/2, where Δx is the grid discretization
    """
    for ri, r in enumerate(radii):
        mask = sphere_volume_average(
            SDF(pos, centers[ri, :], r), smoothing_sphere_radius
        )
        # mask = SDF(pos, centers[ri, :], r)
        amplitude_batch += mask
        phase_batch += mask * phases[ri]


@torch.jit.script
def restrict_amplitude_and_phase_batch(
    pos: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    phases: torch.Tensor,
    smoothing_sphere_radius: float,
    device: torch.device,
    num_batches: int,
):
    """Restricts (or projects) the geometry onto a discretized amplitude and phase mask using a GPU batch approach.

    Args:
        pos: (Nx2) x-y pairs for each grid point
        centers: (Mx2) x-y emitter center
        radii: (Mx1) emitter radii
        phases: (Mx1) emitter phases
        smoothing_sphere_radius: essentially Δx/2, where Δx is the grid discretization
    """

    # initialize the global arrays on the CPU
    N = pos.shape[0]
    amplitude = torch.zeros((N,))
    phase = torch.zeros((N,))

    # Iterate over each batch
    block_size = N // (num_batches)
    start_indices = torch.arange(0, N, block_size)
    stop_indices = torch.concatenate((start_indices[1:], torch.tensor([N])))

    centers = centers.to(device)
    radii = radii.to(device)
    phases = phases.to(device)

    for b in range(num_batches):
        # print("Submitting batch {}".format(b))
        s = start_indices[b]
        e = stop_indices[b]
        Nb = int(e - s)

        amplitude_batch = torch.zeros((Nb,), device=device)
        phase_batch = torch.zeros((Nb,), device=device)

        restrict_amplitude_and_phase(
            amplitude_batch,
            phase_batch,
            (pos[s:e, :]).to(device),
            centers,
            radii,
            phases,
            smoothing_sphere_radius,
        )
        amplitude[s:e] = amplitude_batch
        phase[s:e] = phase_batch

    return (amplitude, phase)

""":md
### Define source parameters
"""

""":py"""
# Constants

from scipy.stats import norm

# Constants

c_speed = 3e8  # speed of light (m/s)
nm, um = 1e-9, 1e-6
N = {"r": 100, "g": 100, "b": 100}


def plot_spectra(wavelengths, spectra, colors):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[25, 7.5])
    for ax, color in zip(axes, colors):
        ax.stem(
            wavelengths[color] / nm,
            spectra[color],
            linefmt=color + "-",
            markerfmt=color + "o",
            basefmt=color + "-",
        )
        ax.plot(wavelengths[color] / nm, spectra[color], color)
        ax.set_title(f"{color.upper()} Spectrum")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity")
    plt.tight_layout()
    plt.show()


# Main execution

colors = ["r", "g", "b"]
S = {}
wavelengths = {}

# laser lineshape parameters

lambda0 = {"r": 638 * nm, "g": 520 * nm, "b": 450 * nm}
linewidth_group = {"r": 0.0015 * um, "g": 0.001 * um, "b": 0.001 * um}
linewidth_span = {
    "r": linewidth_group["r"] * 2,
    "g": linewidth_group["r"] * 2,
    "b": linewidth_group["r"] * 2,
}


def gaussmf(x, params):
    sigma, mean = params
    return norm.pdf(x, loc=mean, scale=sigma)


for color in colors:
    wv_list = np.linspace(
        lambda0[color] - linewidth_span[color],
        lambda0[color] + linewidth_span[color],
        N[color],
    )
    wavelengths[color] = wv_list
    wv_coef = gaussmf(wv_list, [linewidth_group[color] / 2, lambda0[color]])
    wv_coef = wv_coef / np.sum(wv_coef)
    S[color] = wv_coef
plot_spectra(wavelengths, S, colors)

""":py"""
# global variables

nm, um, mm, cm = 1e-9, 1e-6, 1e-3, 1e-2
GHz = 1e9

neff = {
    "r": torch.arange(1.491191, 1.490732, -(1.491191 - 1.490732) / N["r"]),
    "g": torch.arange(1.528777, 1.527924, -(1.528777 - 1.527924) / N["g"]),
    "b": torch.arange(1.565589, 1.564314, -(1.565589 - 1.564314) / N["b"]),
}

neff0 = {
    color: neff[color][int(np.floor(N[color] / 2))] for color in ["r", "g", "b"]
}  # effective index

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft as scfft, fftshift as scfftshift, ifft as scifft
from scipy.interpolate import interp1d


def wavelength_to_frequency(wavelength, intensity):
    frequency = c_speed / wavelength
    sorted_indices = np.argsort(frequency)
    return frequency[sorted_indices], intensity[sorted_indices]


def compute_coherence_function(frequency, intensity):
    spectrum_ft = scfft(intensity)
    c = 299792458
    autocorrelation = scifft(spectrum_ft * np.conj(spectrum_ft))
    autocorrelation = scfftshift(autocorrelation).real
    contrast = np.abs(spectrum_ft) / np.max(np.abs(spectrum_ft))
    delta_nu = np.abs(frequency[1] - frequency[0])
    opd = np.fft.fftfreq(frequency.size, d=delta_nu)
    indices = np.argsort(opd)
    opd = opd[indices]
    contrast = contrast[indices]
    return opd * c, contrast


color_map = {"r": "r-", "g": "g-", "b": "b-"}
for color in colors:
    frequency, sorted_intensity = wavelength_to_frequency(wavelengths[color], S[color])
    opd, contrast = compute_coherence_function(frequency, sorted_intensity)
    plt.plot(
        opd / um / neff0[color],
        contrast,
        color_map[color],
        label=f"{color} {linewidth_group[color]/nm} nm laser",
    )
plt.plot(opd / um, 0.04 * np.ones(len(opd)), "--")
plt.xlim(0, 300)
plt.grid(False)

plt.xlabel("Emitter pitch (um)", fontsize=20)
plt.ylabel("Coherency function", fontsize=20)
plt.legend(handlelength=0.25)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color("black")
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

# Enhance tick visibility

ax.tick_params(
    axis="both",
    which="major",
    labelsize=15,
    length=5,
    width=1,
    colors="black",
    direction="out",
)

plt.show()

""":md
### Define PIC simulation parameters
"""

""":py"""
def calculate_refractive_index(wavelengths):
    neff = {
        color: (
            torch.linspace(1.491191, 1.490732, len(wavelengths["r"]))
            if color == "r"
            else (
                torch.linspace(1.528777, 1.527924, len(wavelengths["g"]))
                if color == "g"
                else torch.linspace(1.565589, 1.564314, len(wavelengths["b"]))
            )
        )
        for color in wavelengths
    }
    wavelengths_tensors = {
        key: torch.from_numpy(value) for key, value in wavelengths.items()
    }
    n_glass = {
        key: torch.sqrt(
            0.6961663
            * (wavelengths_tensors[key] / um) ** 2
            / ((wavelengths_tensors[key] / um) ** 2 - 0.0684043**2)
            + 0.4079426
            * (wavelengths_tensors[key] / um) ** 2
            / ((wavelengths_tensors[key] / um) ** 2 - 0.1162414**2)
            + 0.8974794
            * (wavelengths_tensors[key] / um) ** 2
            / ((wavelengths_tensors[key] / um) ** 2 - 9.896161**2)
            + 1
        )
        for key in wavelengths
    }
    return neff, n_glass


neff, n_glass = calculate_refractive_index(wavelengths)

# beam parameters


grt_aperture = 2 * um
interleave_number = 1  # choose between 1,2,4

simulation_no = 21
pic_pixel_x, pic_pixel_y = 12, 24  # make it odd number
pic_pitch_x, pic_pitch_y = 80 * um, 40 * um
z = 1500 * um
resolution_factor = 4
dx1 = grt_aperture / resolution_factor

current_size = pic_pixel_x * pic_pitch_x / (grt_aperture / resolution_factor) + 1000
next_power_of_2 = 2 ** np.ceil(np.log2(current_size))
padding_size = next_power_of_2 - current_size
M = int(next_power_of_2)
L1 = M * dx1
x1 = torch.linspace(-L1 / 2, L1 / 2, steps=M)
y1 = x1
[X1, Y1] = torch.meshgrid(x1, y1)
large_grid = torch.vstack((X1.flatten(), Y1.flatten())).T

fx1 = torch.linspace(-1 / (2 * dx1), 1 / (2 * dx1), steps=int(M))
# fx1 = fft.fftshift(fx1)
[FX1, FY1] = torch.meshgrid(fx1, fx1)

print("simulation data size: x = " + str(len(x1)) + ", y = " + str(len(y1)))
print("M = " + str(M))
print("dx1 = " + str(dx1))
print("sample no. in w = " + str(grt_aperture / dx1))
print("sample no. in PIC pixel = " + str(pic_pitch_x / dx1))
print("grid size: ", large_grid.shape)

# Create meshgrid
Ex_pos, Ey_pos = torch.meshgrid(
    torch.arange(pic_pixel_x), torch.arange(pic_pixel_y), indexing="xy"
)
if pic_pixel_x % 2 == 1:
    Ex_pos = (Ex_pos.float() - torch.median(Ex_pos.float())) * pic_pitch_x
    Ey_pos = (Ey_pos.float() - torch.median(Ey_pos.float())) * pic_pitch_y
else:
    Ex_pos = (Ex_pos.float() - torch.mean(Ex_pos.float())) * pic_pitch_x
    Ey_pos = (Ey_pos.float() - torch.mean(Ey_pos.float())) * pic_pitch_y

""":py"""
# sampling_size = 300*um
# sampling_res = grt_aperture/resolution_factor/um
# sampling_npix = int(sampling_size / (grt_aperture / resolution_factor))
# sampling_img_size = sampling_size/um

sampling_size = 300*um
sampling_res = dx1/um
# sampling_npix = M
sampling_npix = sampling_size/dx1
# sampling_img_size = L1/um
sampling_img_size = sampling_size/um

# Projector parameters
lens_F = 1.6
lens_res = 2.0 * 1.22 * 0.52 * lens_F
projector_FoV = 52  # deg
LCoS_pix_size = 2
LCoS_size = 2560 * LCoS_pix_size  # um
um2deg = projector_FoV / LCoS_size

""":md
### PIC phase mask generation and beam propagation
"""

""":py"""
randomness = 0.02

speckle_contrasts = {color: [] for color in colors}
emitter_positions = []
phase_values = []
num_iterations = 25
speckle_contrast = {}

for iteration in range(num_iterations):
    # Apply bias

    Ex_pos += (
        pic_pitch_x
        * (1 + (10 - 1) * torch.rand(pic_pixel_y, pic_pixel_x))
        * randomness
        / 10
        * (2 * torch.randint(2, size=(pic_pixel_y, pic_pixel_x)) - 1)
    )
    Ey_pos_bias = (
        (
            torch.ones(pic_pixel_y)
            * pic_pitch_y
            * (1 + (10 - 1) * torch.rand(1, pic_pixel_y))
            * randomness
            / 10
            * (2 * torch.randint(2, size=(1, pic_pixel_y)) - 1)
        )
        .repeat(pic_pixel_x, 1)
        .T
    )
    Ey_pos += Ey_pos_bias

    # Create emitter positions

    emitter_pos = torch.vstack((Ex_pos.flatten(), Ey_pos.flatten())).T

    ### Phase delay for the emitter positions
    # Apply lateral propagation

    x_offset = 3.8 * mm
    lateral_prop_x = torch.zeros(pic_pixel_x, pic_pixel_y).T

    # Assign every other row starting from the first row

    lateral_prop_x[::2] = Ex_pos[::2] - Ex_pos[0, 0]

    # Assign every other row starting from the second row

    lateral_prop_x[1::2] = -Ex_pos[1::2] + Ex_pos[0, -1]
    lateral_prop_x[1::2] += x_offset

    # Apply lateral propagation in y direction

    lateral_prop_y = torch.zeros(pic_pixel_x, pic_pixel_y).T
    Ey_pos_bias = Ey_pos_bias.reshape(pic_pixel_x, pic_pixel_y).T
    num_rows = lateral_prop_y.shape[0]
    for i in range(0, num_rows, 2):
        lateral_prop_y[-(i + 2) : -(i), :] = (
            pic_pitch_y + Ey_pos_bias[-(i + 2) : -(i), :] + 6.4 * um
        ) * (i // 2)
    # If the number of rows is odd, set the first row to continue the pattern

    if num_rows % 2 == 1:
        lateral_prop_y[0, :] = (pic_pitch_y + Ey_pos_bias[0, :] + 6.4 * um) * (
            num_rows // 2
        )
    interleave_no = interleave_number * 2
    interleave_length = torch.tensor(
        [[0, 0, 3.8 * mm, 3.8 * mm, 16 * mm, 16 * mm, 24 * mm, 24 * mm]]
    )
    interleave = torch.zeros(pic_pixel_x, pic_pixel_y).T
    for i in range(pic_pixel_y):
        interleave[i :: interleave_no * 2] = interleave_length[
            0, int(torch.remainder(torch.tensor(i), interleave_no))
        ] * (1 + 0.00 * torch.rand(1))
    total_delay = lateral_prop_x + interleave + lateral_prop_y
    total_phase = 2 * math.pi * total_delay

    # Mask generation

    radius = torch.tensor([grt_aperture / 2.0] * pic_pixel_x * pic_pixel_y)
    phases = total_phase.flatten()

    # transfer to GPU

    large_grid = large_grid.to(device)
    emitter_pos = emitter_pos.to(device)
    radius = radius.to(device)
    phases = phases.to(device)

    amplitude_profile, phase_profile = restrict_amplitude_and_phase_batch(
        large_grid, emitter_pos, radius, phases, dx1 / 2, device, 100
    )

    # back to cpu

    amplitude_profile = torch.Tensor.cpu(amplitude_profile)
    phase_profile = torch.Tensor.cpu(phase_profile)

    amplitude_mask = torch.tensor(amplitude_profile.reshape(len(x1), len(y1)).T)
    phase_mask = torch.tensor(phase_profile.reshape(len(x1), len(y1)).T)

    # phase_mask_temp_exp = torch.exp(1j * phase_mask)
    # final_mask_temp = amplitude_mask * phase_mask_temp_exp
    # # amplitude_mask = torch.abs(final_mask)
    # phase_mask_temp = torch.angle(final_mask_temp)

    # loop through lines

    metric_size = int(sampling_npix)
    metric_window = slice(int(M / 2 - metric_size / 2), int(M / 2 + metric_size / 2))

    # I2_lcos_plane = {color: torch.zeros(metric_size,metric_size) for color in colors}

    I2_lcos_plane_fs = {color: torch.zeros(M, M) for color in colors}
    I2_lcos_plane_fs_cropped = {
        color: torch.zeros(metric_size, metric_size) for color in colors
    }
    # u2_lcos_plane = {color: {} for color in colors}

    for c, color in enumerate(wavelengths):
        for w, wavelength in enumerate(tqdm(wavelengths[color])):
            phase_mask_exp = torch.exp(1j * phase_mask * neff[color][w] / (wavelength))
            # phase_mask_exp = torch.exp(1j * phase_mask * neff0[color] / (wavelength))

            u1 = amplitude_mask * phase_mask_exp
            u2, L2, x2 = propForward(
                u1,
                L1,
                x1,
                wavelength,
                n_glass[color][w] * z,
                dx1,
                X1,
                Y1,
                FX1,
                FY1,
                "RS",
            )

            I2_lcos_plane_fs[color] += S[color][w] * (abs(u2) ** 2)
        I2_lcos_plane_fs[color] = I2_lcos_plane_fs[color] / torch.max(
            torch.max(I2_lcos_plane_fs[color])
        )
        I2_lcos_plane_fs_cropped[color] = I2_lcos_plane_fs[color][
            metric_window, metric_window
        ]

        speckle_contrast[color] = torch.std(I2_lcos_plane_fs_cropped[color])/torch.mean(I2_lcos_plane_fs_cropped[color])

    for color in speckle_contrast:
        speckle_contrasts[color].append(speckle_contrast[color])
    emitter_positions.append(
        emitter_pos.clone()
    )  # Assuming emitter_pos is defined in your loop
    phase_values.append(phases.clone())  # Assuming phases is defined in your loop
# Calculate median speckle contrast


contrasts_tensor = torch.tensor(speckle_contrasts["g"])
median_speckle_contrast_g = torch.median(contrasts_tensor).item()
# Find the closest iteration for green based on median speckle contrast


differences = [
    abs(contrast - median_speckle_contrast_g) for contrast in speckle_contrasts["g"]
]
min_index = differences.index(min(differences))
closest_iteration = min_index
# Retrieve the emitter positions and phase values for the closest iteration for green


emitter_pos_closest = emitter_positions[closest_iteration]
phases_closest = phase_values[closest_iteration]

""":py"""
data_to_plot = [speckle_contrasts[color] for color in speckle_contrasts]
color_labels = ['red', 'green', 'blue']  # Full names for the colors
# Create a box plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(data_to_plot, patch_artist=True)
# Adding colors to each box
for patch, color in zip(box['boxes'], ['red', 'green', 'blue']):
    patch.set_facecolor(color)
# Set custom x-axis labels
plt.xticks(ticks=[1, 2, 3], labels=color_labels)
plt.title('Distribution of Speckle Contrasts for Different Colors')
plt.ylabel('Speckle Contrast')
plt.xlabel('Color')
plt.grid(True)
plt.show()

""":py"""
amplitude_profile, phase_profile = restrict_amplitude_and_phase_batch(
    large_grid, emitter_pos_closest, radius, phases_closest, dx1 / 2, device, 100
)

# back to cpu

amplitude_profile = torch.Tensor.cpu(amplitude_profile)
phase_profile = torch.Tensor.cpu(phase_profile)

amplitude_mask = torch.tensor(amplitude_profile.reshape(len(x1), len(y1)).T)
phase_mask = torch.tensor(phase_profile.reshape(len(x1), len(y1)).T)

# loop through lines
metric_size = int(sampling_npix)
metric_window = slice(int(M / 2 - metric_size / 2), int(M / 2 + metric_size / 2))

I2_lcos_plane_fs = {color: torch.zeros(M, M) for color in colors}
I2_lcos_plane_fs_cropped = {
    color: torch.zeros(metric_size, metric_size) for color in colors
}

for c, color in enumerate(wavelengths):
    for w, wavelength in enumerate(tqdm(wavelengths[color])):
        phase_mask_exp = torch.exp(1j * phase_mask * neff[color][w] / (wavelength))
        # phase_mask_exp = torch.exp(1j * phase_mask * neff0[color] / (wavelength))

        u1 = amplitude_mask * phase_mask_exp
        u2, L2, x2 = propForward(
            u1,
            L1,
            x1,
            wavelength,
            n_glass[color][w] * z,
            dx1,
            X1,
            Y1,
            FX1,
            FY1,
            "RS",
        )

        I2_lcos_plane_fs[color] += S[color][w] * (abs(u2) ** 2)
    I2_lcos_plane_fs[color] = I2_lcos_plane_fs[color] / torch.max(
        torch.max(I2_lcos_plane_fs[color])
    )
    I2_lcos_plane_fs_cropped[color] = I2_lcos_plane_fs[color][
        metric_window, metric_window
    ]

    speckle_contrast[color] = {
        color: torch.std(I2_lcos_plane_fs_cropped[color])
        / torch.mean(I2_lcos_plane_fs_cropped[color])
    }

""":py"""
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow((amplitude_profile.reshape(len(x1), len(y1)).T), cmap="binary")
plt.grid(False)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow((phase_profile.reshape(len(x1), len(y1)).T / (2 * math.pi)))
plt.colorbar()
plt.grid(False)

# # print((total_phase / (2 * math.pi)) * 1e6)
# df = pd.DataFrame(total_delay.numpy())
# df.to_csv(f"emitter_map.csv", index=False)

""":md
### Colormap for RGB plots
"""

""":py"""
def wavelength_to_rgb(wavelength):
    gamma = 0.80
    intensity_max = 255
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0
    R *= intensity_max
    G *= intensity_max
    B *= intensity_max
    return (R / 255), (G / 255), (B / 255)


# Test the function
# print(wavelength_to_rgb(638))  # Should print RGB values for a shade of red
def create_colormap(wavelength, color_name):
    colors = [(0.0, 0.0, 0.0), wavelength_to_rgb(wavelength)]  # Dark color to the color
    return plt.cm.colors.LinearSegmentedColormap.from_list(
        f"{color_name}_colormap", colors, N=256, gamma=1 / 2.2
    )

# Create colormaps
cmap_red = create_colormap(638, 'red')
cmap_green = create_colormap(520, 'green')
cmap_blue = create_colormap(450, 'blue')

""":md
### Plot RGB speckle images
"""

""":py"""
def plot_image(ax, image, title, extent, cmap):
    ax.imshow(
        image,
        aspect="auto",
        interpolation="none",
        extent=extent,
        origin="lower",
        cmap=cmap,
    )
    ax.grid(False)
    ax.set_title(title)


def setup_figure_and_axes(num_plots, extent, labels):
    fig, axs = plt.subplots(1, num_plots, figsize=(15 * num_plots, 15))
    plt.rcParams.update({"font.size": 15})
    plt.rc("axes", labelsize=15)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    fig.text(0.5, 0.04, labels["x"], ha="center", va="center")
    fig.text(0.06, 0.5, labels["y"], ha="center", va="center", rotation="vertical")
    return fig, axs


def plot_lcos_planes(
    x2, I2_lcos_plane_fs_cropped, I2_lcos_plane_fs, cmap_red, cmap_green, cmap_blue
):
    # Calculate extent for metric window

    extent_metric = [
        torch.min(x2[metric_window]),
        torch.max(x2[metric_window]),
        torch.min(x2[metric_window]),
        torch.max(x2[metric_window]),
    ]
    # Calculate extent for full plane

    extent_full = [
        torch.min(x2),
        torch.max(x2),
        torch.min(x2),
        torch.max(x2),
    ]

    # Plot LCoS plane image

    fig, axs = setup_figure_and_axes(3, extent_metric, {"x": "x [mm]", "y": "y [mm]"})
    plot_image(
        axs[0],
        np.array(I2_lcos_plane_fs_cropped["r"], dtype=float),
        "Red",
        extent_metric,
        cmap_red,
    )
    plot_image(
        axs[1],
        np.array(I2_lcos_plane_fs_cropped["g"], dtype=float),
        "Green",
        extent_metric,
        cmap_green,
    )
    plot_image(
        axs[2],
        np.array(I2_lcos_plane_fs_cropped["b"], dtype=float),
        "Blue",
        extent_metric,
        cmap_blue,
    )
    plt.show()

    # Plot LCoS full-plane image

    fig, axs = setup_figure_and_axes(3, extent_full, {"x": "x [mm]", "y": "y [mm]"})
    plot_image(
        axs[0],
        np.array(I2_lcos_plane_fs["r"], dtype=float),
        "Red",
        extent_full,
        cmap_red,
    )
    plot_image(
        axs[1],
        np.array(I2_lcos_plane_fs["g"], dtype=float),
        "Green",
        extent_full,
        cmap_green,
    )
    plot_image(
        axs[2],
        np.array(I2_lcos_plane_fs["b"], dtype=float),
        "Blue",
        extent_full,
        cmap_blue,
    )
    plt.show()


# Assuming other necessary variables like 'x2', 'I2_lcos_plane', 'I2_lcos_plane_fs', 'cmap_red', 'cmap_green', 'cmap_blue' are already defined

plot_lcos_planes(
    x2, I2_lcos_plane_fs_cropped, I2_lcos_plane_fs, cmap_red, cmap_green, cmap_blue
)

contrast1 = {
    color: torch.std(I2_lcos_plane_fs_cropped[color])
    / torch.mean(I2_lcos_plane_fs_cropped[color])
    for color in colors
}
for color in colors:
    print(f"Speckle contrast without averaging {color.upper()}: {contrast1[color]}")

""":py"""
import oculus.research.orcoptics.storage_async as storage_async

base_path = f"tree/epic/ePIC FL4/simulation_no{simulation_no}/"

for color in ["r", "g", "b"]:
    df = pd.DataFrame(I2_lcos_plane_fs[color].numpy())
    file_name = f"I2_lcos_plane_fs_{color}.csv"
    df.to_csv(file_name, index=False)
    path = f"{base_path}{file_name}"
    storage_async.manifold_upload_file(path=path, localfile=file_name, overwrite=True)
df = pd.DataFrame(phase_mask.numpy())
file_name = f"phase_mask.csv"
df.to_csv(file_name, index=False)
path = f"{base_path}{file_name}"
storage_async.manifold_upload_file(path=path, localfile=file_name, overwrite=True)

# print((total_phase / (2 * math.pi)) * 1e6)

df = pd.DataFrame(amplitude_mask.numpy())
file_name = f"amplitude_mask.csv"
df.to_csv(file_name, index=False)
path = f"{base_path}{file_name}"
storage_async.manifold_upload_file(path=path, localfile=file_name, overwrite=True)

""":md
#### Sample Notebook for sotring and accessing files in frl_optics_orcoptics Manifold bucket

Folder hierarchy: You can check the following link for the overall folder hierarchy:

https://www.internalfb.com/manifold/explorer/frl_optics_orcoptics/tree

Please use arvr_optics bento kernel
"""

""":py"""
import h5py
import numpy as np
import oculus.research.orcoptics.storage_async as storage_async
from matplotlib import pyplot as plt
import os

help(storage_async.manifold_upload_file)
help(storage_async.manifold_download_as_file)

""":md
#### Store the simulated speckle intensity

Data is stored into h5 file and uploaded to Manifold (frl_optics_orcoptics)

Use the following format for data contained in the file:

h5file = { <p>
<pre>
    "<b>lens_F</b>": F-number of the lens stack,
    "<b>lens_res</b>: lens resolving power in um,
    "<b>projector_FoV</b>: FoV of the projector,
    "<b>LCoS_pix_size</b>: LCoS pixel size in um,
    "<b>LCoS_size</b>: LCoS size in um,
    "<b>um2deg</b>: um2deg,
    "<b>sampling_res</b>": pixel size of the complex field at the LCoS plane = lens_res / sampling_factor in um,
    "<b>sampling_npix</b>": Number of pixels in the files, 
    "<b>sampling_img_size</b>": Size on the LCoS in um,  
    "<b>x</b>": Sampling points along x,  
    "<b>y</b>": Sampling points along y 
    "<b>fields</b>":
        "<b>color</b>": 
            "<b>id</b>": intensity of the field
    "<b>Spectrum</b>":
        "<b>color</b>": wavelengths
    "<b>lambda0</b>": center wavelengths
    "<b>fwhm</b>": full-width at half maximum of laser linewidths
</pre>
</p>
}

Intensity fields stored into group named "fields" that is the sum of all complex fields simulated for all wavelengths, and time sampling (if any) at each color channel. Each color channel (r,g,or b) includes dubgroup of wavelengths, which is further split to complex fields at different time instances of dynamic despeckle.

"""

""":py"""
id = (
    "0p" + str(int(randomness * 100)) + "rand_z" + str(int(z / um)) + "um"
)  # ID for the simulation
file_name = "speckle_field_data_{}.h5".format(id)
base_path = (
    "tree/epic/speckle/ePIC_FL4/"
    + str(int(pic_pitch_x / um))
    + "p_x_"
    + str(int(pic_pitch_y / um))
    + "p_y_"
    + "sc_"
    + "/"
)
path = base_path+file_name

with h5py.File(file_name, "w") as h5:
    h5["lens_F"] = lens_F
    h5["lens_res"] = lens_res
    h5["projector_FoV"] = projector_FoV
    h5["LCoS_pix_size"] = LCoS_pix_size
    h5["LCoS_size"] = LCoS_size
    h5["um2deg"] = um2deg
    h5["sampling_res"] = sampling_res
    h5["sampling_npix"] = sampling_npix
    h5["sampling_img_size"] = sampling_img_size
    h5["x"] = x1[metric_window]
    h5["y"] = y1[metric_window]
    h5["lambda0"] = np.array(list(lambda0.values()), dtype=np.float64)
    h5["fwhm"] = np.array(list(fwhm.values()), dtype=np.float64)

    fields = h5.create_group("fields")
    spectrum = h5.create_group("spectrum")
    for c, color in enumerate(wavelengths):
        # color_set = h5.create_group(str(color))
        # for w, wavelength in enumerate(wavelengths[color]):
        #     wavelength_set = h5.create_group(str(wavelength))
        #     for sample in range(
        #         sample_count
        #     ):  # Assume 3 time samples for dynamic despeckling per wavelength
                # randomfield = np.exp(1j * 2 * np.pi * rng.random(xg.shape))
                # wavelength_set[str(sample)] = randomfield
            # color_set[str(wavelength)] = wavelength_set
        # fields[str(color)] = color_set
        fields[str(color)] = speckle_field_intensity_cropped[color]
        spectrum[str(color)] = wavelengths[color]
    print(fields.keys())
    # print(fields["g"].keys())
## Save file to Manifold (folder structure created automatically if not existing)
storage_async.manifold_upload_file(path=path, localfile=file_name, overwrite=True)

""":py"""
import copy
import os
import time

import h5py
import numpy as np
import oculus.research.orcoptics.storage_async as storage_async
import pandas as pd
from matplotlib import pyplot as plt

from oculus.research.ar_display_modeling.power_models.color_tools import (
    apply_white_pointXYZ,
    colorMatchFcn,
)

from oculus.research.ar_display_modeling.power_models.uniformity_metrics import (
    gen_params,
    rgb_to_xyz_dets,
    wg_synthetic_correction,
)
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import j1
from skimage.color import xyz2rgb

""":md
#### S-CIELAB based uniformity calculation based on speckle data stored in Manifold

Notebook example for data storing and retrieval example is given in https://www.internalfb.com/intern/anp/view/?id=5250168. The dataset should be stored in path "tree/epic/speckle/SUBFOLDER/SIM_NAME.h5". Here SUBFOLDER should be used to group simulations based on design/configuration and SIM_NAME is file name for the simulation data. All the relevant simulation parameters are assumed to be stored in the datafile. See the Notebook for further details. 

The following function loads the dataset and processes the complex speckle fields on LCoS, and forms final RGB speckle image. The image is then analyzed for uniformity metrics using S-CIELAB based approach.
"""

""":py"""
%local-changes

""":py"""
def speckle_metrics(
    manifold_path, show_data=False, plot_results=False, plot_fields=False
):
    # plot_fields - If True will plot individual speckle images for each color/wavelenght/sample

    ## Download simulation data from Manifold -> stored locally as speckle_data.h5 file
    if os.path.exists("speckle_data.h5"):
        os.remove("speckle_data.h5")
    storage_async.manifold_download_as_file(
        path=manifold_path, local_path="speckle_data.h5"
    )

    # Projector parameters
    lens_F = 1.6
    lens_res = 2.0 * 1.22 * 0.52 * lens_F
    projector_FoV = 52  # deg
    LCoS_pix_size = 2
    LCoS_size = 2560 * LCoS_pix_size  # um
    um2deg = projector_FoV / LCoS_size

    # Calculations
    run_data = {}
    system_params = {}
    use_wl = []

    ## Open downloaded speckle_data.h5 file and process
    with h5py.File("speckle_data.h5", "r") as h5:

        field_data = h5["fields"]
        sampling_res = h5["sampling_res"][()]
        sampling_img_size = h5["sampling_img_size"][()]
        sampling_npix = h5["sampling_npix"][()]
        sampling_ppd = sampling_npix / (um2deg * sampling_img_size)
        x = h5["x"][()]
        y = h5["y"][()]
        xg, yg = np.meshgrid(x, y)

        # Lens resolution for green
        lens_res = 2.0 * 1.22 * 0.52 * lens_F
        chrom_metric_ppd = np.floor(1 / (um2deg * lens_res / 2))

        system_params = {
            "lens_F": lens_F,
            "projector_FoV": projector_FoV,
            "LCoS_pix_size": LCoS_pix_size,
            "LCoS_size": LCoS_size,
            "lens_res": lens_res,
            "sampling_res": sampling_res,
            "sampling_img_size": sampling_img_size,
            "sampling_npix": sampling_npix,
            "sampling_ppd": sampling_ppd,
        }

        if show_data:
            print(f"Sampling resolution (um)\t{sampling_res}")
            print(f"Sampling resolution (amin)\t{um2deg * sampling_res*60}")
            print(f"Sampling resolution (ppd)\t{sampling_ppd}")
            print(f"Number of pixels\t\t{sampling_npix}")
            print(f"Image size (um)\t\t\t{sampling_img_size}")
            print(f"Image size (deg)\t\t{um2deg * sampling_img_size}")

            print(f"lens_F#\t\t\t\t{lens_F}")
            print(f"lens_resolving power\t\t{lens_res} um")
            print(f"lens_resolving power\t\t{lens_res*um2deg} deg")
            print(f"LCoS pixel size\t\t\t{LCoS_pix_size}")
            print(f"LCoS size\t\t\t{LCoS_size}")
            print(f"chrom_metric_ppd\t\t{chrom_metric_ppd}")
        lamb, xFcn, yFcn, zFcn = colorMatchFcn("1931_full")
        XYZ_mat = np.zeros((3, len(field_data.values())))
        for c, color in enumerate(field_data.keys()):

            if plot_fields:
                fig = plt.figure()
                fig.suptitle("Color: " + color, fontsize=16)
                fig.tight_layout()
                figId = 0
            # for s, sample in enumerate(sample_data[color][wavelength].keys()):

            #     # Account for diffraction limited projector blur and calculate speckle intensity map
            #     pupilfield = circ(xg_a, yg_a) * np.fft.fftshift(
            #         np.fft.fft2(sample_data[color][wavelength][sample][()])
            #     )
            #     speckle_image = np.abs(np.fft.ifft2(pupilfield)) ** 2
            #     speckle_image_intensity += speckle_image * weights[f]

            #     if plot_fields:
            #         figId += 1
            #         ax = fig.add_subplot(numWaves, numSamples, figId)
            #         ax.set_title(
            #             f"Wavelength {wavelength}, sample {sample}",
            #             fontdict={"fontsize": 12},
            #         )
            #         ax.imshow(
            #             speckle_image,
            #             extent=(
            #                 0,
            #                 np.max(sampling_img_size),
            #                 0,
            #                 np.max(sampling_img_size),
            #             ),
            #         )

            # speckle_image_intensity /= np.max(speckle_image_intensity)
            speckle_image_intensity = field_data[color][()]
            run_data[f"center_{color}"] = copy.deepcopy(speckle_image_intensity)
            run_data[f"input_{color}"] = np.ones_like(speckle_image_intensity)

            # # fit Gaussian spectrum
            # if len(spectrum) > 1:
            #     weights /= np.max(weights)
            #     xd = np.linspace(np.min(spectrum) - 4, np.max(spectrum) + 4, 50)
            #     yd = interpolate.interp1d(
            #         spectrum, weights, fill_value=0, bounds_error=False, kind="nearest"
            #     )(xd)
            #     parameters, covariance = curve_fit(
            #         Gauss, xd, yd, p0=[np.mean(spectrum), 0.1]
            #     )
            #     fit_y = Gauss(xd, parameters[0], parameters[1])
            #     use_wl.append((np.round(parameters[0]), parameters[1]))
            # else:
            #     use_wl.append((np.round(wl), 1))
        use_wl.extend([(lambda0[color], fwhm[color]) for color in colors])
    params = gen_params()
    params["use_wl"] = np.matrix(use_wl)
    params["zoneA_params"] = (5, "circle", 0, 0)
    params["zoneB_params"] = (10, "circle", 0, 0)
    params["zoneC_params"] = (0, "corner", 0, 0)
    params["CIE_CMF_ver"] = "1931_full"

    params["uc_method"] = "default"
    params["input_det_name"] = "input"
    params["vFoV"] = um2deg * sampling_img_size
    params["hFoV"] = um2deg * sampling_img_size
    params["chrom_metric_ppd"] = chrom_metric_ppd
    params["RegionDetails"] = {"Region0": ["center"]}

    params["bypass_UC"] = True
    params["uniformity"] = {}
    params["filter_params"] = {}

    # params["use_SUC"] = False
    # params["SUC_pupil"] = "center"
    # params["SUC_corrMap"] = np.ones_like(xg)
    # params["RGBscale"] = F / np.max(F)

    start = time.time()
    res_df, results_dict, image_dict = wg_synthetic_correction(
        run_or_raw_data=run_data,
        eyebox_location_names=["center"],
        params=params,
        compute_results_dict=True,
        compute_image_dict=True,
        compute_monochrome_metrics=False,
    )

    print(f"Metric calculation duration: {time.time() - start}")

    # dE00 values
    data = {
        "dE_00_mean": res_df.loc[0, "dE_00_Global_FFOV_mean"],
        "dE_00_max": res_df.loc[0, "dE_00_Global_FFOV_max"],
        "dE_00_p95": res_df.loc[0, "dE_00_Global_FFOV_p95"],
        "dE_mean": res_df.loc[0, "dE_Global_FFOV_mean"],
        "dE_max": res_df.loc[0, "dE_Global_FFOV_max"],
        "dE_p95": res_df.loc[0, "dE_Global_FFOV_p95"],
    }

    # Calculate std_dev/mean for white balanced brightness
    xyz_det = rgb_to_xyz_dets(
        [run_data[f"center_r"], run_data[f"center_g"], run_data[f"center_b"]], params
    )
    RGBscale, rXYZ_WB, gXYZ_WB, bXYZ_WB = apply_white_pointXYZ(
        xyz_det[0], xyz_det[1], xyz_det[2], np.ones(xg.shape) == 0
    )
    XYZ_WB = rXYZ_WB + gXYZ_WB + bXYZ_WB
    data["std/mean_L"] = np.round(
        100 * np.std(XYZ_WB[:, :, 1]) / np.mean(XYZ_WB[:, :, 1])
    )

    # std_dev/mean for raw rgb intensities
    for c, cc in enumerate(params["use_wl"]):
        data[f"std/mean_{colors[c]}"] = np.round(
            100
            * np.std(run_data[f"center_{colors[c]}"])
            / np.mean(run_data[f"center_{colors[c]}"])
        )
    # Calculate std_dev/mean for S-CIELAB filtered and white balanced brightness
    XYZ_SCIE = image_dict["center"]["XYZ"]
    data["std/mean_L_scie"] = np.round(
        100 * np.std(XYZ_SCIE[:, :, 1]) / np.mean(XYZ_SCIE[:, :, 1])
    )

    # Display metrics
    if plot_results:
        plot_imges(params, image_dict)
    return (system_params, data, params, image_dict["center"])


## Function for plotting results
def plot_imges(params, image_dict):

    extent = [
        -params["hFoV"] / 2,
        params["hFoV"] / 2,
        -params["vFoV"] / 2,
        params["vFoV"] / 2,
    ]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(image_dict["center"]["RGB_balanced"], extent=extent)
    ax[0, 0].set_xlabel("deg")
    ax[0, 0].set_ylabel("deg")
    ax[0, 0].set_title("RGB balanced", fontsize=12)

    ax[0, 1].imshow(xyz2rgb(image_dict["center"]["XYZ"]), extent=extent)
    ax[0, 1].set_xlabel("deg")
    ax[0, 1].set_ylabel("deg")
    ax[0, 1].set_title("RGB SCIELAB", fontsize=12)

    ax[1, 0].imshow(image_dict["center"]["dE_Global"], extent=extent)
    ax[1, 0].set_xlabel("deg")
    ax[1, 0].set_ylabel("deg")
    ax[1, 0].set_title("dE", fontsize=12)

    ax[1, 1].imshow(image_dict["center"]["dL_Global"], extent=extent)
    ax[1, 1].set_xlabel("deg")
    ax[1, 1].set_ylabel("deg")
    ax[1, 1].set_title("dL", fontsize=12)

    fig.tight_layout()


def Airy(x, y, F, wave):
    r = np.sqrt(x**2 + y**2)
    arg = np.pi * r / (wave * F)
    arg[arg == 0] = 1e-9
    val = j1(arg) / arg
    return val


def circ(x, y):
    return np.where(np.sqrt(x**2 + y**2) <= 0.5, 1, 0)


# Gaussian function for spectral fit
def Gauss(x, center, sigma):
    y = np.exp(-((center - x) ** 2) / (2 * sigma**2))  # FWHM definition
    return y

""":md
#### Process ePIC simulation data for static baseline designs
"""

""":py"""
# """:py"""
# base_paths = [
#     "tree/epic/speckle/stationary3/20p_x_20p_y_si/",
# ]
# files = [
#     "speckle_field_data_0p0rand_z700um.h5",
# ]
base_paths = [base_path]
files = [file_name]

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.set_option("display.width", 9999)

metrics = []
params = []
rgb_balanced = {}
dE00_imgs = {}
for p, pname in enumerate(base_paths):
    for f, fname in enumerate(files):
        prms, data, DPA_dict, image_dict = speckle_metrics(
            pname + fname, show_data=True, plot_results=True
        )
        data["file"] = pname + fname
        metrics.append(data)
        prms["file"] = pname + fname
        params.append(prms)
        dE00_imgs[pname + fname] = image_dict["dE_00_Global"]
        rgb_balanced[pname + fname] = image_dict["RGB_balanced"]
FoV = [
    -DPA_dict["hFoV"] / 2,
    DPA_dict["hFoV"] / 2,
    -DPA_dict["vFoV"] / 2,
    DPA_dict["vFoV"] / 2,
]

df_metric = pd.DataFrame(data=metrics)
df_params = pd.DataFrame(data=params)
index = df_metric.loc[:, "file"]
print(df_metric.drop(labels="file", axis=1).set_index(index))
print()
print(df_params.drop(labels="file", axis=1).set_index(index))

""":py"""
base_paths = [base_path]
files = [file_name]

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = None
pd.set_option("display.width", 9999)

metrics = []
params = []
rgb_balanced = {}
dE00_imgs = {}
for p, pname in enumerate(base_paths):
    for f, fname in enumerate(files):
        print(f"Process file {pname}{fname}")
        prms, data, DPA_dict, image_dict = speckle_metrics(
            pname + fname, show_data=True, plot_results=True
        )
        data["file"] = pname + fname
        metrics.append(data)
        prms["file"] = pname + fname
        params.append(prms)
        dE00_imgs[pname + fname] = image_dict["dE_00_Global"]
        rgb_balanced[pname + fname] = image_dict["RGB_balanced"]
FoV = [
    -DPA_dict["hFoV"] / 2,
    DPA_dict["hFoV"] / 2,
    -DPA_dict["vFoV"] / 2,
    DPA_dict["vFoV"] / 2,
]

df_metric = pd.DataFrame(data=metrics)
df_params = pd.DataFrame(data=params)
index = df_metric.loc[:, "file"]
print(df_metric.drop(labels="file", axis=1).set_index(index))
print()
print(df_params.drop(labels="file", axis=1).set_index(index))

""":py"""
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%m/%d/%Y %H:%M:%S")
print("Simulation finished at =", current_time)
