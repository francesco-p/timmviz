import streamlit as st
import numpy as np
import timm
import  streamlit_toggle as tog
from utils import model_size
import os
from PIL import Image
import urllib
import torchvision.transforms.functional as transform
import cv2
import pickle as pkl
from collections import defaultdict
import cmapy

def to_255(x):
    return (x - x.min()) / (x.max() - x.min()) * 255

@st.cache_data
def prepare_feature_maps(_model, _img, fname):
    """Prepare feature maps for visualization
    Args:
        _model (torch.nn.Module): model 
        _img (torch.Tensor): image
        fname (str): filename to save feature maps
    Returns:
        None
    """

    model = _model
    img = _img

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Move model to gpu and eval mode
    model.to('cuda')
    model.eval()

    # Extract features
    features = model(img.unsqueeze(0).to('cuda'))

    # Moves images and model to cpu
    img = img.cpu()
    model = model.cpu()
    all_layers_features = [x.cpu() for x in features]
    
    # Save feature maps + image
    with open(f'{fname}.pkl', 'wb') as f:
        pkl.dump(all_layers_features+[img], f)

    #return all_layers_features, img
    return [x.shape[1] for x in all_layers_features]


@st.cache_data
def create_plots(fname, num_fmaps):
    """Create feature map plots
    Args:
        fname (str): filename of feature maps
        num_fmaps (list): numbers of feature maps to plot for each layer
    Returns:
        None
    """
    prefix = fname.split('/')[0]

    # Load feature maps
    with open(f'{fname}.pkl', 'rb') as f:
        restore = pkl.load(f)

    # Extract features and image
    all_layers_features = restore[:-1]
    img = restore[-1]
    
    # normalize image to [0,255]
    h,w = img.shape[-2], img.shape[-1]
    img = img.squeeze(0).permute(1,2,0).numpy()
    img = to_255(img).astype(np.uint8)

    # Save plots
    fnames = []
    for i, features_layer_i in enumerate(all_layers_features):
        features_layer_i = features_layer_i.numpy()
        for j, feat_j in enumerate(features_layer_i[0][:num_fmaps[i]]):
            feat_j = cv2.resize(feat_j, (w,h))
            # normalize feature map to [0,255]
            feat_j = to_255(feat_j).astype(np.uint8)
            #feat_j = cv2.applyColorMap(feat_j, cv2.COLORMAP_JET)
            feat_j = cv2.applyColorMap(feat_j, cmapy.cmap('coolwarm'))
            heatmap = cv2.addWeighted(img, 0.35, feat_j, 0.65, 0)
            fname = f'{prefix}/{i:03}_{j:03}.jpg'
            cv2.imwrite(fname, heatmap)
            fnames.append(fname)
    return fnames


@st.cache_data
def load_model(model_name, pretrained):
    """Load model
    Args:
        model_name (str): model name
        pretrained (bool): load pretrained weights
    Returns:
        model (torch.nn.Module): model
        model_is_supported (bool): model is supported
    """
    try:
        return timm.create_model(model_name, pretrained=pretrained, features_only=True), True
    except Exception as e:
        if isinstance(e, urllib.error.URLError):
            st.session_state.logs.append("Model weights can't be downloaded. Chcek your connection!")
        else:
            st.session_state.logs.append(f"Model not supported yet. It must implement features_only")
        return None, False


###############################################################################
st.set_page_config(layout="wide")

# Global variables
if 'pretrained' not in st.session_state:
    st.session_state.pretrained = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'img_file_buffer' not in st.session_state:
    st.session_state.img_file_buffer = []
###############################################################################


with st.sidebar:
    st.markdown("# TimmViz")

    model_name = st.sidebar.selectbox('Select your timm model', timm.list_models())
    
    model, model_is_supported = load_model(model_name, st.session_state.pretrained)

    if model_is_supported:
        st.table(model_size(model))



    _ = tog.st_toggle_switch(label="Pretrained", 
                        key="pretrained", 
                        default_value=False, 
                        label_after = False, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )



if not model_is_supported:
    st.markdown(f'# {st.session_state.logs[-1]}')

else:

    col1, col2 = st.columns(2)
    with col1:
        #show_filters(tensor, selected_module)
        # File uploader
        st.markdown(f"## Upload picture")
        img_file_buffer = st.file_uploader('')
        if img_file_buffer:
            st.session_state.img_file_buffer = img_file_buffer

    with col2:
        if st.session_state.img_file_buffer != []:
            st.image(st.session_state.img_file_buffer, width=300, caption='Uploaded Image')  


    # Activation map visualization
    if st.session_state.img_file_buffer != []:
        
        tensor_image = transform.to_tensor(Image.open(st.session_state.img_file_buffer))

        prefix = f'fmaps/{model_name}/pretrained_{st.session_state.pretrained}'
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        
        fname=f'{prefix}/{st.session_state.img_file_buffer.name.split(".")[0]}'
        
        try:
            len_fmaps = prepare_feature_maps(model, tensor_image, fname)

            # slider to select number of feature maps
            with st.sidebar:
                num_fmaps = []
                for i in range(len(len_fmaps)):
                    number = st.number_input(f'Number of activation maps in layer {i+1}', 1, len_fmaps[i], 10)
                    num_fmaps.append(number)

            image_files = create_plots(fname, num_fmaps)

            with st.container():
                st.write("## Activation maps")
                img_dict = defaultdict(list)

                for i, image_path in enumerate(image_files):
                    layer = image_path.split('/')[-1].split('_')[0]
                    img_dict[layer].append(open(image_path, 'rb').read())
                    #feature = image_path.split('/')[-1].split('_')[1].split('.')[0]
                    #image_captions.append(f'{int(layer)}.{int(feature)}')

                for k in img_dict.keys():
                    st.write(f'### Layer {int(k)+1}')
                    images = img_dict[k]
                    image_captions = [f'{i}' for i in range(len(images))]
                    st.image(images,width=200,caption=image_captions)

        except Exception as e:
            st.session_state.logs.append("The forward failed for some reason. Check your console!")
            st.markdown(f"# {st.session_state.logs[-1]}")
            print(e)
            st.stop()