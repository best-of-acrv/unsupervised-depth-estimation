# Code adapted from https://github.com/mrharicot/monodepth, with the following licence

# Copyright © Niantic, Inc. 2018. Patent Pending.
# All rights reserved.

# ================================================================================

# This Software is licensed under the terms of the UCLB ACP-A Licence which allows
# for non-commercial use only. For any other use of the software not covered by
# the terms of this licence, please contact info@uclb.com

# ================================================================================

# UCLB ACP-A Licence

# This Agreement is made by and between the Licensor and the Licensee as defined
# and identified below. 

# 1.  Definitions.
#     In this Agreement (“the Agreement”) the following words shall have the
#     following meanings:

#     "Authors" shall mean C. Godard, O. Mac Aodha, G. Brostow
#     "Licensee" Shall mean the person or organisation agreeing to use the
#     Software in accordance with these terms and conditions.
#     "Licensor" shall mean UCL Business PLC whose registered office is at The
#     Network Building, 97 Tottenham Court Road, London W1T 4TP. UCL Business is a
#     the technology transfer arm of University College London (UCL).
#     "Owner" shall mean Niantic Inc., a company organised and existing under the
#     laws of Delaware, whose principal place of business is at 2 Bryant Street,
#     #220, San Francisco, 94105. Owner is a third party beneficiary of this
#     Agreement and may enforce its terms as if it were a party to this Agreement.
#     "Software" shall mean the MonoDepth Software in source code or object code
#     form and any accompanying documentation. 

# 2.  License.
#     2.1 The Licensor has all necessary rights to grant a licence under: (i)
#     copyright and rights in the nature of copyright subsisting in the Software;
#     and (ii) patent rights resulting from a patent application filed by the
#     Licensor in the United Kingdom in connection with the Software. The Licensor
#     grants the Licensee for the duration of this Agreement, a free of charge,
#     non-sublicenseable, non-exclusive, non-transferable copyright and patent
#     licence (in consequence of said patent application) to use the Software for
#     non-commercial purpose only, including teaching and research at educational
#     institutions and research at not-for-profit research institutions in
#     accordance with the provisions of this Agreement. Non-commercial use
#     expressly excludes any profit-making or commercial activities, including
#     without limitation sale, licence, manufacture or development of commercial
#     products, use in commercially-sponsored research, provision of consulting
#     service, use for or on behalf of any commercial entity, and use in research
#     where a commercial party obtains rights to research results or any other
#     benefit. Any use of the Software for any purpose other than non-commercial
#     research shall automatically terminate this Licence. 
    
#     2.2 The Licensee is permitted to make modifications to the Software provided
#     that distribution of such modifications is in accordance with Clause 3.

#     2.3 Except as expressly permitted by this Agreement and save to the extent
#     and in the circumstances expressly required to be permitted by law, the
#     Licensee is not permitted to rent, lease, sell, offer to sell or loan the
#     Software or its associated documentation.

# 3.  Redistribution and modifications
#     3.1 The Licensee may reproduce and distribute copies of the Software only to
#     this same GitHub repository with or without modifications, in source format
#     only and provided that any and every distribution is accompanied by an
#     unmodified copy of this Licence and that the following copyright notice is
#     always displayed in an obvious manner: Copyright © Niantic, Inc. 2018. All
#     rights reserved. 
    
#     3.2 In the case where the Software has been modified, any distribution must
#     include prominent notices indicating which files have been changed.
    
#     3.3 The Licensee shall cause any work that it distributes or publishes, that
#     in whole or in part contains or is derived from the Software or any part
#     thereof (“Work based on the Software”), to be licensed as a whole at no
#     charge to all third parties under the terms of this Licence.

# 4.  Duration.
#     This Agreement is effective until the Licensee terminates it by destroying
#     the Software and its documentation together with all copies. It will also
#     terminate automatically if the Licensee fails to abide by its terms. Upon
#     automatic termination the Licensee agrees to destroy all copies of the
#     Software and its documentation.

# 5.  Disclaimer of Warranties.
#     The Software is provided as is. To the maximum extent permitted by law,
#     Licensor provides no warranties or conditions of any kind, either express or
#     implied, including without limitation, any warranties or condition of title,
#     non-infringement or fitness for a particular purpose. 

# 6.  Limitation of Liability.
#     In no event shall the Licensor and/or Authors be liable for any direct,
#     indirect, incidental, special, exemplary or consequential damages (including
#     but not limited to, procurement of substitute goods or services; loss of
#     use, data or profits; or business interruption) however caused and on any
#     theory of liability, whether in contract, strict liability, or tort
#     (including negligence or otherwise) arising in any way out of the use of
#     this Software, even if advised of the possibility of such damage.

# 7.  Indemnity.
#     The Licensee shall indemnify the Licensor and/or Authors against all third
#     party claims that may be asserted against or suffered by the Licensor and/or
#     Authors and which relate to use of the Software by the Licensee or the
#     Recipient.

# 8.  Intellectual Property.
#     8.1 As between the Licensee and Licensor,copyright and all other
#     intellectual property rights subsisting in or in connection with the
#     Software and supporting information shall remain at all times the property
#     of the Licensor but Licensee acknowledges and agrees that Owner is the owner
#     of all right, title and interest in and to the Software. The Licensee shall
#     acquire no rights in any such material except as expressly provided in this
#     Agreement.

#     8.2 No permission is granted to use the trademarks or product names of the
#     Licensor or Owner except as required for reasonable and customary use in
#     describing the origin of the Software and for the purposes of abiding by the
#     terms of Clause 3.1.

#     8.3 The Licensee shall promptly notify the Licensor, in sufficient detail,
#     all improvements and new uses of the Software (“Improvements”). The Licensor
#     and its affiliates shall have a non-exclusive, fully paid-up, royalty-free,
#     irrevocable and perpetual licence under the Improvements for non-commercial
#     academic research and teaching purposes.  

#     8.4 The Licensee grants an exclusive first option to the Owner to be
#     exercised by the Owner within three (3) years of the date of notification of
#     the Improvements under Clause 8.3 to use any Improvements for commercial
#     purposes on terms to be negotiated and agreed by Licensee and Owner in good
#     faith within a period of six (6) months from the date of exercise of the
#     said option (including without limitation any royalty share in net income
#     from such commercialization payable to the Licensee, as the case may be).

# 9.  Acknowledgements.
#     The Licensee shall acknowledge the Authors and use of the Software in the
#     publication of any work that uses, or results that are achieved through, the
#     use of the Software. The following citation shall be included in the
#     acknowledgement: “Unsupervised Monocular Depth Estimation with Left-Right
#     Consistency, by C. Godard, O Mac Aodha, G. Brostow, CVPR 2017.”

# 10. Governing Law.
#     This Agreement shall be governed by, construed and interpreted in accordance
#     with English law and the parties submit to the exclusive jurisdiction of the
#     English courts.

# 11. Termination.
#     Upon termination of this Agreement, the licenses granted hereunder will
#     terminate and Sections 5, 6, 7, 8, 9, 10 and 11 shall survive any
#     termination of this Agreement.

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.misc import imread, imresize
from collections import Counter
from scipy.interpolate import LinearNDInterpolator as LinearNDInterpolator
import extract_depth 

def read_calib_file(path):
        # taken from https://github.com/hunse/kitti
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data

def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + '/calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam==2:
        focal_length = P2_rect[0,0]
    elif cam==3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline

def get_depth(calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False, inv_depth=False):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + '/calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + '/calib_velo_to_cam.txt')
    
    velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3,3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3,4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth = lin_interp(im_shape, velo_pts_im)

    focal_len, baseline = get_focal_length_baseline(calib_dir, cam)

    if inv_depth:
        focal_len, baseline = get_focal_length_baseline(calib_dir, cam)
        inv_depth = np.zeros(depth.shape)
        val_idx = depth>0
        inv_depth[val_idx] = (baseline * focal_len) / depth[val_idx]
        return inv_depth
    else:
        return depth