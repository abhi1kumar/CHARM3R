import math
import numpy as np
import torch

def project_3d_points_in_4D_format(p2, points_4d, pad_ones= False):
    """
        Projects 3d points appened with ones to 2d using projection matrix
        :param p2:       np array or torch Tensor 4 x 4
        :param points:   np array or torch Tensor 4 x N
        :return: coord2d np array or torch Tensor 4 x N
    """
    N = points_4d.shape[1]
    z_eps = 1e-2

    if type(points_4d) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points_4d, np.ones((1, N))))

        coord2d = np.matmul(p2, points_4d)
        ind = np.where(np.abs(coord2d[2]) > z_eps)

    elif type(points_4d) == torch.Tensor:
        if pad_ones:
            points_4d = torch.cat([points_4d, torch.ones((1, N), dtype= points_4d.dtype, device= points_4d.device)], dim= 0)

        coord2d = torch.matmul(p2, points_4d)
        ind = torch.abs(coord2d[2]) > z_eps

    coord2d[:2, ind] /= coord2d[2, ind]

    return coord2d

def backproject_2d_pixels_in_4D_format(p2_inv, points, pad_ones= False):
    """
        Projects 2d points with x and y in pixels and appened with ones to 3D using inverse of projection matrix
        :param p2_inv:   np array or torch Tensor 4 x 4
        :param points:   np array or torch Tensor 4 x N or 3 x N
        :param pad_ones: whether to pad_ones or not. 3 X N shaped points need to be padded
        :return: coord2d np array or torch Tensor 4 x N
    """
    N = points.shape[1]

    if type(points) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points, np.ones((1, N))))
        else:
            points_4d = points

        points_4d[0] = np.multiply(points_4d[0], points_4d[2])
        points_4d[1] = np.multiply(points_4d[1], points_4d[2])
        output       = np.matmul(p2_inv, points_4d)

    elif type(points) == torch.Tensor:
        if pad_ones:
            points_4d = torch.cat([points, torch.ones((1, N), dtype= points.dtype, device= points.device)], dim= 0)
        else:
            points_4d = points

        # Making a new variable and then multiplying fixes the one of the variables needed for gradient computation has been modified by an inplace operation
        points_t    = torch.ones_like(points_4d)
        points_t[0] = points_4d[0] * points_4d[2]
        points_t[1] = points_4d[1] * points_4d[2]
        points_t[2] = points_4d[2]
        output      = torch.matmul(p2_inv, points_t)

    return output

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices
    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        # corners_2D = np.zeros([3, corners_3d.shape[1]])
        # for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = corners_2D[:2].T#(corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d

def convertRot2Alpha(ry3d, z3d, x3d):

    if type(z3d) == torch.Tensor:
        alpha = ry3d - torch.atan2(-z3d, x3d) - 0.5 * math.pi
        while torch.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while torch.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2

    elif type(z3d) == np.ndarray:
        alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * math.pi
        while np.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while np.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2

    else:
        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        while alpha > math.pi: alpha -= math.pi * 2
        while alpha <= (-math.pi): alpha += math.pi * 2

    return alpha

def snap_to_pi(ry3d):

    if type(ry3d) == torch.Tensor:
        while (ry3d > (math.pi)).any(): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while (ry3d <= (-math.pi)).any(): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    elif type(ry3d) == np.ndarray:
        while np.any(ry3d > (math.pi)): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while np.any(ry3d <= (-math.pi)): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    else:

        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d <= (-math.pi): ry3d += math.pi * 2

    return ry3d
