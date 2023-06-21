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


def compute_fmaps_torchshow(img, all_layers_features, pretrained):
   # Save feature maps
    fnames = []
    for i, features_layer_i in enumerate(all_layers_features):
        for j, feature_j in enumerate(features_layer_i[0][:35]):
            feature_j = feature_j.unsqueeze(0).unsqueeze(0)
            h,w = img.shape[-2], img.shape[-1]
            fname = f'fmaps/{model_name}/pretrained_{pretrained}/{i:03}_{j:03}.jpg'
            feature_j = F.interpolate(feature_j, size=[h,w], mode='bicubic')
            #feature_j = feature_j.expand(1,3, -1, -1)
            print(feature_j.dtype, img.dtype)
            print(feature_j.shape, img.shape)
            print(feature_j.min(), feature_j.max(), img.min(), img.max())
            print(img[0].dtype, type(img[0]))
            ts.overlay([img[0], feature_j[0]], cmap='coolwarm', save_as=fname, alpha=[1, 0.8])
            #ts.save(feature_j, path=fname, cmap='coolwarm')
            fnames.append(fname)
    return fnames
