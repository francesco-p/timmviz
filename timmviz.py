import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
import torchshow as ts
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
import  streamlit_toggle as tog
from utils import process_attrs, show_hist, model_size
from torchvision import datasets, transforms
from einops import rearrange
import os
from PIL import Image
import torch.nn.functional as F
import builtins
import urllib
import seaborn as sns
import torchvision.transforms.functional as transform

sns.set()

# deep lab → senza la parte di (decoder) e viceversa
# saliency detection (unsupervised)
# metodi visualizzazione 

@st.cache_data
def prepare_feature_maps(_model, model_name, _img, img_name, pretrained):
    model = _model
    model.to('cuda')
    model.eval()
    # img = Image.open('castoro.jpg')
    # #img = img.resize((224, 224))
    # img = transforms.ToTensor()(img)
    img = _img.unsqueeze(0)
    img = img.to('cuda')

    features = model(img)
    
    # Moves images and model to cpu
    img = img.cpu()
    model = model.cpu()
    all_layers_features = [x.cpu() for x in features]

    # Save feature maps
    fnames = []
    for i, features_layer_i in enumerate(all_layers_features):
        for j, feature_j in enumerate(features_layer_i[0][:5]):
            feature_j = feature_j.unsqueeze(0).unsqueeze(0)
            h,w = img.shape[-2], img.shape[-1]
            fname = f'fmaps/{model_name}/pretrained_{pretrained}/{i:03}_{j:03}.jpg'
            feature_j = F.interpolate(feature_j, size=[h,w], mode='bicubic')
            #feature_j = feature_j.expand(1,3, -1, -1)
            print(feature_j.dtype, img.dtype)
            print(feature_j.shape, img.shape)
            print(feature_j.min(), feature_j.max(), img.min(), img.max())
            ts.overlay([img[0], feature_j[0]], cmap='coolwarm', save_as=fname, alpha=[1, 0.8])
            #ts.save(feature_j, path=fname, cmap='coolwarm')
            fnames.append(fname)
    return fnames


@st.cache_data
def load_model(model_name, pretrained):
    try:
        return timm.create_model(model_name, pretrained=pretrained, features_only=True), True
    except Exception as e:
        if isinstance(e, urllib.error.URLError):
            st.session_state.logs.append("Model weights can't be downloaded. Chcek your connection!")
        else:
            st.session_state.logs.append(f"Model not supported yet. It must implement features_only")
        return None, False


def show_filters(tensor, selected_module):
    if len(tensor.shape) == 4 and (tensor.shape[-1] == tensor.shape[-2] and tensor.shape[-1] == 1):
        ts.save(tensor.squeeze(-1).squeeze(-1), 'imgs/tensor.png')
        st.image('imgs/tensor.png', width=300, caption=selected_module)
    elif  len(tensor.shape) == 4 and tensor.shape[0] > 100:
        tensor = tensor[:12, :32, :, :] 
        #tensor = rearrange(tensor, 'f c h w -> c h (w f)')
        ts.save(tensor, 'imgs/tensor.png')
        st.image('imgs/tensor.png', width=300, caption=selected_module + ' (first 10 features, first 32 channels stacked)')
    else:
        
        # ch = st.slider('Next Channels:', min_value=1, max_value=10, value=1)
        # start = (ch-1)*3
        # end = ch*3
        # ts.save(tensor[:64, :end, :, :], 'imgs/tensor.png')
        ts.save(tensor, 'imgs/tensor.png')
        st.image('imgs/tensor.png', width=300, caption=selected_module)


st.set_page_config(layout="wide")




###############################################################################
###############################################################################
###############################################################################
# Global variables
if 'pretrained' not in st.session_state:
    st.session_state.pretrained = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
###############################################################################


with st.sidebar:
    st.markdown("# TimmViz")

    model_name = st.sidebar.selectbox('Select your timm model', timm.list_models())
    
    model, model_is_supported = load_model(model_name, st.session_state.pretrained)

    if model_is_supported:
        st.table(model_size(model))


    
    tog.st_toggle_switch(label="Pretrained", 
                        key="pretrained", 
                        default_value=False, 
                        label_after = False, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )
    


if not model_is_supported:
    st.markdown('# Something wrong :/')
    st.markdown(f'`More Logs : {st.session_state.logs[-1]}`')
else:
    # disable gradient and create df
    df = pd.DataFrame(columns=['module', 'shape'])
    for name, param in model.named_parameters():
        param.requires_grad = False
        df = df.append({'module': name, 'shape': param.shape}, ignore_index=True)


    with st.sidebar:
        st.markdown("## Parameters")
        gb = GridOptionsBuilder.from_dataframe(df[["module", "shape"]])
        # configure selection
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()
        data = AgGrid(df,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True,
                    allow_unsafe_jscode=True,
                    height=500,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)


    # Default option is the first layer.
    if len(data["selected_rows"]) != 0:
        selected_row = data["selected_rows"][0]
    else:
        selected_row = df.iloc[0]
    selected_module = selected_row["module"]
    selected_shape = selected_row["shape"]

    # Get the selected tensor
    attrs = selected_module.split(".")
    tensor = process_attrs(model, attrs)

    # adds batch size of 1 if no bsize is present
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)




    col1, col2 = st.columns(2)
    with col1:
        #show_filters(tensor, selected_module)
        # File uploader
        st.markdown(f"## Upload picture")
        img_file_buffer = st.file_uploader('upload your image')
 
        show_filters(tensor, selected_module)



    with col2:
        st.markdown(f"### .{selected_module} {tuple(selected_shape)}")
        #nbins = st.slider('Histogram bins:', min_value=5, max_value=200, value=100)
        show_hist(tensor, nbins=100)
        st.image('imgs/hist.png', width=400, caption=f'Histogram of all the module weights')

                         
    # Feature map visualization
    if img_file_buffer is not None:
        PIL_image = Image.open(img_file_buffer)
        tensor_image = transform.to_tensor(PIL_image)
        image_files = prepare_feature_maps(model, model_name, tensor_image, img_file_buffer.name, pretrained=st.session_state.pretrained)
        
        print(st.session_state.pretrained)
        with st.container():
            st.write("## Feature maps")
            images = []
            image_captions = []
            for i, image_path in enumerate(image_files):
                images.append(open(image_path, 'rb').read())
                layer = image_path.split('/')[-1].split('_')[0]
                feature = image_path.split('/')[-1].split('_')[1].split('.')[0]
                image_captions.append(f'{int(layer)}.{int(feature)}')
            st.image(images,width=200,caption=image_captions)

