import torch

def rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula implemented with PyTorch.
    axis:   shape (3,)
    theta:  scalar tensor (angle in radians)
    returns: (3,3) rotation matrix
    """
    axis = axis / torch.norm(axis)
    ux, uy, uz = axis.unbind(0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    C = 1 - c

    R = torch.stack([
        torch.stack([    c + ux*ux*C, ux*uy*C - uz*s, ux*uz*C + uy*s ]),
        torch.stack([ uy*ux*C + uz*s,     c + uy*uy*C, uy*uz*C - ux*s ]),
        torch.stack([ uz*ux*C - uy*s, uz*uy*C + ux*s,     c + uz*uz*C ])
    ], dim=0)
    return R

def rotate_point(p: torch.Tensor,
                 pivot: torch.Tensor,
                 axis: torch.Tensor,
                 theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate point(s) p around the line through pivot with direction axis.
    p:     (..., 3)
    pivot: (3,)
    axis:  (3,)
    theta: scalar
    returns: (..., 3)
    """
    q = p - pivot
    R = rotation_matrix(axis, theta)        # (3,3)
    # for possibly batched p, use @ with proper dims
    q_rot = q @ R.T                         # (...,3) @ (3,3) -> (...,3)
    return q_rot + pivot

def compute_rotation_angles(x0: torch.Tensor,
                            x1: torch.Tensor,
                            r: torch.Tensor,
                            axis= None) -> (torch.Tensor, torch.Tensor):
    """
    Solve for angles θ such that rotating x1 about axis (x0×x1) around pivot x0
    yields a point at distance r from the origin.
    x0, x1: (3,)
    r:      scalar (target radius)
    returns: tuple of two scalars (θ1, θ2)
    """

    # axis direction
    if axis is None:
        axis = torch.cross(x0, x1)
    norm_axis = torch.norm(axis)
    if norm_axis < 1e-8:
        raise ValueError("Axis ill-defined (x0 × x1 ≃ 0).")
    u = axis / norm_axis

    # compute coefficients for D0 + D1 cosθ + D2 sinθ = r^2
    a = x0
    q = x1 - x0

    D0 = a.dot(a) + q.dot(q) + 2 * (a.dot(u)) * (u.dot(q))
    D1 = 2 * (a.dot(q) - (a.dot(u)) * (u.dot(q)))
    D2 = 2 * a.dot(torch.cross(u, q))

    K  = r*r - D0
    Rm = torch.hypot(D1, D2)

    if torch.abs(K) > Rm + 1e-6:
        raise ValueError("No real solution for requested r.")

    phi   = torch.atan2(D2, D1)
    alpha = torch.acos(K / Rm)

    return phi + alpha, phi - alpha

