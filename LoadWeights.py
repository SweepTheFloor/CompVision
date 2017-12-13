def preload_encoder_weights(model_ft):
    
    import torchvision.models as models
    vgg16 = models.vgg16_bn(pretrained=True)
    vgg_state_dict = vgg16.state_dict()

    model_ft_dict = model_ft.state_dict()

    #In for loop ignore ignore classifiers from vgg and only keep features
    vgg_num_classifiers = 6       # The last six learning parameters of vgg model are classifiers
    vgg_num_features = len(vgg_state_dict.items())-6
    count = 0 
    for a,b in zip(vgg_state_dict.items(), model_ft_dict.items()):  # it zips until the shortest dict: vgg in this case

        if count <vgg_num_features :# ignore the last 6 classifiers
            ka,va = a
            kb,vb = b
            #print(ka, '\n', kb , '\n\n')
            model_ft_dict[kb] = va
            count+= 1

    # load the new state dict        
    model_ft.load_state_dict(model_ft_dict)

    
# ##----------- DEMO --------------------------
# %run 'segnet_model.ipynb'
# #from segnet_model import network
# model_ft = network()
# preload_encoder_weights(model_ft)
