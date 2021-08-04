from smallUnet import smallUnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.ndimage as ndi
import cv2
import numpy as np


class trained_NN(object):

    def __init__(self, path, cuda=True): 
        model = smallUnet()
        self.cuda = cuda
        if cuda: 
            model.load_state_dict(torch.load(path))
            model.cuda()

        else:
            print('loading without cuda')
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
           
        model.eval()
        self.model =  model 

    def find_skyrms(self, image, tilt_dir, thresh=0.5): 
    
        ## apply rotation 
        dimy, dimx = image.shape
        imagerot = ndi.rotate(image, 180-tilt_dir)
        image2 = center_pad_pwr2(imagerot)
        ndimy, ndimx = image2.shape
        
        # Convert to 4D tensor (required, even if it is a single image)
        image4d = image2[None, None, ...]
        # Convert to pytorch format and move to GPU
        if self.cuda: 
            image4d_ = torch.from_numpy(image4d).float().cuda()
        else:
            image4d_ = torch.from_numpy(image4d).float()

        # make a prediction
        prediction = self.model.forward(image4d_)
        prediction = F.softmax(prediction, dim=1).cpu().detach().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        # get coordinates

        # prediction = ndi.rotate(prediction, theta-180)
        # prediction[0] = ndi.rotate(prediction[0,:,:,:], theta-180, axes=(0,1))
        prediction2 = ndi.rotate(prediction[0,:,:,:], tilt_dir-180, axes=(0,1))
        
        prediction2 = center_crop_im(prediction2, (dimy,dimx), dim_order_in="channels_last")[:,:,::-1]
        self.prediction = prediction2
        coordinates_e = Finder(self.prediction[None,...], threshold=thresh).get_all_coordinates()
        centers = coordinates_e[0][:,:2]
        return centers

    def rng_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def center_pad_pwr2(image): 
    dimy, dimx = np.shape(image)
    final_dim = int(2**np.ceil(np.log2(max(dimy, dimx))))
    padl = int(np.floor((final_dim-dimx)/2))
    padr = int(np.ceil((final_dim-dimx)/2))
    padt = int(np.floor((final_dim-dimy)/2))
    padb = int(np.ceil((final_dim-dimy)/2))
    return np.pad(image, ((padt, padb), (padl, padr)))

def center_crop_im(image, shape, dim_order_in="channels_last"):
    if image.ndim == 2: 
        dimy, dimx = image.shape
    elif image.ndim == 3: 
        if dim_order_in == "channels_last":
            dimy, dimx, dimz = image.shape
        elif dim_order_in == "channels_first":
            dimz, dimy, dimx = image.shape

    dyf, dxf = shape
    cropl = int(np.floor((dimx-dxf)/2))
    cropr = int(np.ceil((dimx-dxf)/2))
    cropt = int(np.floor((dimy-dyf)/2))
    cropb = int(np.ceil((dimy-dyf)/2))
    if dim_order_in == "channels_last":
        return image[cropt:-cropb, cropl:-cropr]
    elif dim_order_in == "channels_first":
        return image[:,cropt:-cropb, cropl:-cropr]

class Finder:
    '''
    Transforms pixel data from NN output into coordinate data
    '''
    def __init__(self, nn_output,
                 threshold=0.5, dist_edge=5,
                 dim_order='channel_last'):
        
        if nn_output.shape[-1] == 1: # Add background class for 1-channel data
            nn_output_b = 1 - nn_output
            nn_output = np.concatenate(
                (nn_output[:, :, :, None], nn_output_b[:, :, :, None]), axis=3)
        if dim_order == 'channel_first':  # make channel dim the last dim
            nn_output = np.transpose(nn_output, (0, 2, 3, 1))
        elif dim_order == 'channel_last':
            pass
        else:
            raise NotImplementedError(
                'For dim_order, use "channel_first" (e.g. pytorch)',
                 'or "channel_last" (e.g. tensorflow)')
        self.nn_output = nn_output
        self.threshold = threshold
        self.dist_edge = dist_edge
                       
    def get_all_coordinates(self):
        '''Extract all object center coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)'''
        
        def find_com(image_data):
            '''Find objects via center of mass methods'''
            labels, nlabels = ndi.label(image_data)
            coordinates = np.array(
                ndi.center_of_mass(
                    image_data, labels, np.arange(nlabels) + 1))
            coordinates = coordinates.reshape(coordinates.shape[0], 2)
            return coordinates

        d_coord = {}
        for i, decoded_img in enumerate(self.nn_output):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class backgrpund is always the last one
            for ch in range(decoded_img.shape[2]-1):
                _, decoded_img_c = cv2.threshold(
                    decoded_img[:, :, ch], self.threshold, 1, cv2.THRESH_BINARY)

                dilated_img_c = ndi.binary_dilation(decoded_img_c, iterations=2)
                coord = find_com(dilated_img_c)
                coord_ch = self.rem_edge_coord(coord)
                category_ch = np.zeros((coord_ch.shape[0], 1)) + ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis=1)
        return d_coord

    def rem_edge_coord(self, coordinates):
        '''Remove coordinates at the image edges'''

        def coord_edges(coordinates, w, h):
            return [coordinates[0] > w - self.dist_edge,
                    coordinates[0] < self.dist_edge,
                    coordinates[1] > h - self.dist_edge,
                    coordinates[1] < self.dist_edge]

        w, h = self.nn_output.shape[1:3]
        coord_to_rem = [
                        idx for idx, c in enumerate(coordinates) 
                        if any(coord_edges(c, w, h))
                        ]
        coord_to_rem = np.array(coord_to_rem, dtype=int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates


