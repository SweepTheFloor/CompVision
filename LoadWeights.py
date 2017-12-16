def load_vgg16_cnn_encoder_weights(model_ft):
    
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

def save_weights_as_pickleFile(model_ft):
    import pickle
    with open("segnet_state_dict.pkl", "wb") as output_file:
         pickle.dump(model_ft.state_dict(), output_file)
    
def load_weights_from_pickleFile(model_ft):
    import pickle
    with open("segnet_state_dict.pkl", "rb") as output_file:
         my_state_dict = pickle.load(output_file)

    # assign dictionary parameters to  
    model_ft.load_state_dict(my_state_dict)
    
    # must evaluate to initialize batchnorms and dropouts correctly
    model_ft.eval()
    
# ##----------- DEMO --------------------------
# %run 'segnet_model.ipynb'
# #from segnet_model import network
# model_ft = network()
# load_vgg16_cnn_encoder_weights(model_ft)
# save_weights_as_pickleFile(model_ft)

# another_model_ft = network()
# load_weights_from_pickleFile(another_model_ft)

# print(another_model_ft.state_dict())


