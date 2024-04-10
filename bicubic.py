import torch

BC = (
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0),
    (2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1),
    (0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1),
    (-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0),
    (9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2),
    (-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2),
    (2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0),
    (-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1),
    (4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1),
)


def bicubic_kernels(Z, d1, d2):
    dZ1 = torch.zeros_like(Z)
    dZ2 = torch.zeros_like(Z)
    dZ12 = torch.zeros_like(Z)

    # First derivatives on first axis
    # df/dx = (f(x+1, y) - f(x-1, y)) / 2
    dZ1[1:-1] = (Z[:-2] - Z[2:]) / (2 * d1)
    dZ1[0] = (Z[0] - Z[1]) / d1
    dZ1[-1] = (Z[-2] - Z[-1]) / d1
    # First derivatives on second axis
    # df/dy = (f(x,y+1) - f(x,y-1)) / 2
    dZ2[:, 1:-1] = (Z[:, :-2] - Z[:, 2:]) / (2 * d2)
    dZ2[:, 0] = (Z[:, 0] - Z[:, 1]) / d2
    dZ2[:, -1] = (Z[:, -2] - Z[:, -1]) / d2

    # Second derivatives across both axes
    # d2f/dxdy = (f(x-h, y-k) - f(x-h, y+k) - f(x+h, y-k) + f(x+h, y+k)) / (4hk)
    dZ12[1:-1, 1:-1] = (Z[:-2, :-2] - Z[:-2, 2:] - Z[2:, :-2] + Z[2:, 2:]) / (4 * d1 * d2)
    return dZ1, dZ2, dZ12


def interp_bicubic(
    x,
    y,
    Z,
    dZ1=None,
    dZ2=None,
    dZ12=None,
    get_Y=True,
    get_dY=False,
    get_ddY=False,
):
    """
    Compute bicubic interpolation of a 2D grid at arbitrary locations.

    Parameters
    ----------
    x : torch.Tensor
        x-coordinates of the points to interpolate. Must be a 0D or 1D tensor.
        It should be in corner pixel units, meaning that x,y = 0,0 is the
        position of the [0,0] pixel in the grid.
    y : torch.Tensor
        y-coordinates of the points to interpolate. Must be a 0D or 1D tensor.
        It should be in corner pixel units, meaning that x,y = 0,0 is the
        position of the [0,0] pixel in the grid.
    Z : torch.Tensor
        2D grid of values to interpolate. The first axis corresponds to the
        y-axis and the second axis to the x-axis. The values in Z correspond to
        pixel corner values, so Z[0,0] is the value at the bottom left corner of
        the grid. The grid should be at least 2x2 which would be the 4 corners
        of a single pixel.
    dZ1 : torch.Tensor or None
        First derivative of Z along the x-axis. If None, it will be estimated
        using central differences.
    dZ2 : torch.Tensor or None
        First derivative of Z along the y-axis. If None, it will be estimated
        using central differences.
    dZ12 : torch.Tensor or None
        Second derivative of Z along both axes. If None, it will be estimated
        using central differences.
    get_Y : bool
        Whether to return the interpolated values. This will add the estimated Y values to the return tuple
    get_dY : bool
        Whether to return the interpolated first derivatives. This will add dY1 and dY2 to the return tuple
    get_ddY : bool
        Whether to return the interpolated second derivatives. This will add dY12, dY11, and dY22 to the return tuple
    """

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (received {Z.ndim}D tensor)")
    if x.ndim > 1:
        raise ValueError(f"x must be 0 or 1D (received {x.ndim}D tensor)")
    if y.ndim > 1:
        raise ValueError(f"y must be 0 or 1D (received {y.ndim}D tensor)")

    # Convert coordinates to pixel indices
    h, w = Z.shape
    x = 0.5 * ((x + 1) * w - 1)
    x = x.clamp(-0.5, w - 0.5)
    y = 0.5 * ((y + 1) * h - 1)
    y = y.clamp(-0.5, h - 0.5)
    d1 = 2.0 / w
    d2 = 2.0 / h

    # Compute bicubic kernels if not provided
    if dZ1 is None or dZ2 is None or dZ12 is None:
        _dZ1, _dZ2, _dZ12 = bicubic_kernels(Z, d1, d2)
    if dZ1 is None:
        dZ1 = _dZ1
    if dZ2 is None:
        dZ2 = _dZ2
    if dZ12 is None:
        dZ12 = _dZ12

    # Extract pixel values
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = x0.clamp(0, w - 2)
    x1 = x1.clamp(1, w - 1)
    y0 = y0.clamp(0, h - 2)
    y1 = y1.clamp(1, h - 1)

    # Build interpolation vector
    v = torch.zeros((len(x), 16), dtype=Z.dtype, device=Z.device)
    v[:, 0] = Z[y0, x0]
    v[:, 1] = Z[y0, x1]
    v[:, 2] = Z[y1, x1]
    v[:, 3] = Z[y1, x0]
    v[:, 4] = dZ1[y0, x0] * d1
    v[:, 5] = dZ1[y0, x1] * d1
    v[:, 6] = dZ1[y1, x1] * d1
    v[:, 7] = dZ1[y1, x0] * d1
    v[:, 8] = dZ2[y0, x0] * d2
    v[:, 9] = dZ2[y0, x1] * d2
    v[:, 10] = dZ2[y1, x1] * d2
    v[:, 11] = dZ2[y1, x0] * d2
    v[:, 12] = dZ12[y0, x0] * d1 * d2
    v[:, 13] = dZ12[y0, x1] * d1 * d2
    v[:, 14] = dZ12[y1, x1] * d1 * d2
    v[:, 15] = dZ12[y1, x0] * d1 * d2

    # Compute interpolation coefficients
    c = (torch.tensor(BC, dtype=v.dtype, device=v.device) @ v.unsqueeze(-1)).reshape(-1, 4, 4)

    # Compute interpolated values
    return_interp = []
    t = torch.where((x < 0), (x % 1) - 1, torch.where(x >= w - 1, x % 1 + 1, x % 1))
    u = torch.where((y < 0), (y % 1) - 1, torch.where(y >= h - 1, y % 1 + 1, y % 1))
    if get_Y:
        Y = torch.zeros_like(x)
        for i in range(4):
            for j in range(4):
                Y += c[:, i, j] * t**i * u**j
        return_interp.append(Y)
    if get_dY:
        dY1 = torch.zeros_like(x)
        dY2 = torch.zeros_like(x)
        for i in range(4):
            for j in range(4):
                if i > 0:
                    dY1 += i * c[:, i, j] * t ** (i - 1) * u**j
                if j > 0:
                    dY2 += j * c[:, i, j] * t**i * u ** (j - 1)
        return_interp.append(dY1)
        return_interp.append(dY2)
    if get_ddY:
        dY12 = torch.zeros_like(x)
        dY11 = torch.zeros_like(x)
        dY22 = torch.zeros_like(x)
        for i in range(4):
            for j in range(4):
                if i > 0 and j > 0:
                    dY12 += i * j * c[:, i, j] * t ** (i - 1) * u ** (j - 1)
                if i > 1:
                    dY11 += i * (i - 1) * c[:, i, j] * t ** (i - 2) * u**j
                if j > 1:
                    dY22 += j * (j - 1) * c[:, i, j] * t**i * u ** (j - 2)
        return_interp.append(dY12)
        return_interp.append(dY11)
        return_interp.append(dY22)
    return tuple(return_interp)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a 2D grid of random values
    np.random.seed(0)
    Z = np.random.randn(10, 10)

    # Interpolate at random locations
    x = np.linspace(-1, 1, 101)
    y = np.linspace(-1, 1, 101)
    x, y = np.meshgrid(x, y)
    Y, dY1, dY2, dY12, dY11, dY22 = interp_bicubic(
        torch.tensor(x.flatten()),
        torch.tensor(y.flatten()),
        torch.tensor(Z),
        get_Y=True,
        get_dY=True,
        get_ddY=True,
    )
    Y = Y.numpy().reshape(101, 101)
    dY1 = dY1.numpy().reshape(101, 101)
    dY2 = dY2.numpy().reshape(101, 101)
    dY12 = dY12.numpy().reshape(101, 101)
    dY11 = dY11.numpy().reshape(101, 101)
    dY22 = dY22.numpy().reshape(101, 101)

    # Plot the results
    print("Z")
    plt.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("Z.png")
    plt.close()
    print("Y")
    plt.imshow(Y, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("Y_est.png", dpi=200)
    plt.close()
    print("dY1")
    plt.imshow(dY1, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("dY1_est.png")
    plt.close()
    print("dY2")
    plt.imshow(dY2, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("dY2_est.png")
    plt.close()
    print("dY12")
    plt.imshow(dY12, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("dY12_est.png")
    plt.close()
    print("dY11")
    plt.imshow(dY11, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("dY11_est.png")
    plt.close()
    print("dY22")
    plt.imshow(dY22, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    plt.colorbar()
    plt.savefig("dY22_est.png")
    plt.close()
