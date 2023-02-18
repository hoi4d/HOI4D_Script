import torch
import numpy as np
import cv2

def projKps(self, kps):
        '''
        # Example: Go from camera coordinate system (3D coordinates) to pixel coordinate system (pixel coordinates)
        ans = pcd.transpose()
        ans = np.dot(camMat, ans)
        ans = ans.transpose()
        ans1 = (ans[:, 0] / ans[:, 2]).reshape(-1, 1)
        ans2 = (ans[:, 1] / ans[:, 2]).reshape(-1, 1)
        print(np.concatenate((ans1, ans2), axis=1))
        '''
        kps = kps.permute(0, 2, 1)
        kps = torch.einsum('ik, bkj -> bij', self.camMat, kps)
        kps = kps.permute(0, 2, 1)
        ret1 = kps[..., 0] / kps[..., 2]
        ret2 = kps[..., 1] / kps[..., 2]
        ret3 = kps[..., 2]
        ret = torch.stack([ret1, ret2, ret3], dim=-1)
        return ret

def showHandJoints(imgInOrg, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = 3

    for joint_num in range(gtIn.shape[0]):

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            if PYTHON_VERSION == 3:
                joint_color = list(
                    map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num %
                                  4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(
                gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
        else:
            if PYTHON_VERSION == 3:
                joint_color = list(
                    map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num %
                                  4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(
                gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

    for limb_num in range(len(limbs)):

        x1 = gtIn[limbs[limb_num][0], 1]
        y1 = gtIn[limbs[limb_num][0], 0]
        x2 = gtIn[limbs[limb_num][1], 1]
        y2 = gtIn[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(
                    map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num %
                                 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

    if filename is not None:
        cv2.imwrite(filename, imgIn)
    else:
        cv2.imshow('kp+rgb',imgIn)
        cv2.waitKey(0)


    return imgIn