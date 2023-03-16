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

@st.cache_data
def prepare_feature_maps(_model, model_name, pretrained):
    model = _model
    model.to('cuda')
    model.eval()
    img = Image.open('io.jpg')
    #img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = img.to('cuda')

    features = model(img)
    
    # Moves images and model to cpu
    img = img.cpu()
    model = model.cpu()
    all_layers_features = [x.cpu() for x in features]

    # Save feature maps
    fnames = []
    for i, features_layer_i in enumerate(all_layers_features):
        for j, feature_j in enumerate(features_layer_i[0][:8]):
            feature_j = feature_j.unsqueeze(0).unsqueeze(0)
            h,w = img.shape[-2], img.shape[-1]
            fname = f'fmaps/{model_name}/pretrained_{pretrained}/{i:03}_{j:03}.jpg'
            feature_j = F.interpolate(feature_j, size=[h,w], mode='bicubic')
            #feature_j = feature_j.expand(1,3, -1, -1)
            print(feature_j.shape, img.shape)
            print(feature_j.min(), feature_j.max(), img.min(), img.max())
            ts.overlay([img[0], feature_j[0]], cmap='coolwarm', save_as=fname, alpha=[1, 0.8])
            #ts.save(feature_j, path=fname, cmap='coolwarm')
            fnames.append(fname)
    return fnames


@st.cache_data
def load_model(model_name, pretrained):
    return timm.create_model(model_name, pretrained=pretrained, features_only=True)

st.set_page_config(layout="wide")


if 'pretrained' not in st.session_state:
    st.session_state.pretrained = False



with st.sidebar:
    st.markdown("# TimmViz")
    model_name = st.sidebar.selectbox('Select your timm model',timm.list_models())

    model = load_model(model_name, st.session_state.pretrained)
    
    

    tog.st_toggle_switch(label="Pretrained", 
                        key="pretrained", 
                        default_value=False, 
                        label_after = False, 
                        inactive_color = '#D3D3D3', 
                        active_color="#11567f", 
                        track_color="#29B5E8"
                        )


size_dataframe = model_size(model)

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



if len(data["selected_rows"]) != 0:
    selected_row = data["selected_rows"][0]
else:
    selected_row = df.iloc[0]
selected_module = selected_row["module"]
selected_shape = selected_row["shape"]

attrs = selected_module.split(".")
tensor = process_attrs(model, attrs)
if len(tensor.shape) == 1:
    tensor = tensor.unsqueeze(0)


st.markdown(f"## Summary {model_name}")
st.table(size_dataframe)

st.markdown(f"# .{selected_module}")
col1, col2 = st.columns(2)
with col1:
    st.markdown("## Histogram")
    # Show histogram of tensor
    show_hist(tensor, nbins=100)
    st.image('imgs/hist.png', caption=selected_module)



with col2:
    st.markdown(f'## shape: {selected_shape}')
    if len(tensor.shape) == 4 and tensor.shape[0] > 100:
        tensor = tensor[:12, :32, :, :]
        #tensor = rearrange(tensor, 'f c h w -> c h (w f)')
        st.write(tensor.shape)
        ts.save(tensor, 'imgs/tensor.png')
        st.image('imgs/tensor.png', caption=selected_module + ' (first 10 features, first 32 channels stacked)')
    else:
        ts.save(tensor, 'imgs/tensor.png')
        st.image('imgs/tensor.png', caption=selected_module)

# Feature map visualization

image_files = prepare_feature_maps(model,model_name, pretrained=st.session_state.pretrained)
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
