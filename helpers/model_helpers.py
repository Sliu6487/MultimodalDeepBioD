import copy

import torch


def copy_freeze_parameters(model1, model2, model3,
                           emb_section=-1, emb_layer=-1):
    """
    copy weight and freeze all layers before embedding layer
    """
    combined_model = copy.deepcopy(model3)
    combined_model.chemception.load_state_dict(model1.state_dict())
    if emb_section == -1:
        children = list(combined_model.chemception.children())
    else:
        children = list(combined_model.chemception.children())[:emb_section + 1]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False

    combined_model.mlp_decpt.load_state_dict(model2.state_dict())
    if emb_layer == -1:
        layers = combined_model.mlp_decpt.hidden
    else:
        layers = combined_model.mlp_decpt.hidden[:emb_layer + 1]
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

    return combined_model


def check_frozen_layers(trained_models: dict):
    trained_model1 = trained_models['model_trained1']
    trained_model2 = trained_models['model_trained2']
    trained_model3 = trained_models['model_trained3']

    n_wrong_param = 0
    for param1, param3 in zip(trained_model1.parameters(),
                              trained_model3.chemception.parameters()):
        if ~torch.all(torch.eq(param1, param3)):
            print('problem in madel1 parameters')
            n_wrong_param += 1
    if n_wrong_param == 0:
        print('No problem in madel1 parameters.')
    else:
        print(f'{n_wrong_param} parameters are wrong')

    n_wrong_param = 0
    for param2, param3 in zip(trained_model2.parameters(),
                              trained_model3.mlp_decpt.parameters()):
        if ~torch.all(torch.eq(param2, param3)):
            print('problem in madel1 parameters')
            n_wrong_param += 1
    if n_wrong_param == 0:
        print('No problem in madel2 parameters.')
    else:
        print(f'{n_wrong_param} parameters are wrong')
