import copy

import torch


def freeze_layers(fusion_model,
                  freeze_mlp_layers_to=-1) -> None:
    # todo: make this cleaner
    # MLP
    emb_mlp_layer = fusion_model.emb_mlp_layer

    if emb_mlp_layer == -1:
        layers = fusion_model.mlp_decpt.hidden
    else:
        layers = fusion_model.mlp_decpt.hidden[:emb_mlp_layer + 1]

    if len(layers) < -freeze_mlp_layers_to:
        freeze_mlp_layers_to = len(layers)

    if freeze_mlp_layers_to == -1:
        for param in fusion_model.mlp_decpt.hidden.parameters():
            param.requires_grad = False
    else:
        frz_indx = emb_mlp_layer + freeze_mlp_layers_to + 1
        # freeze before embedding layer based on freeze_mlp_layers_to
        for layer in fusion_model.mlp_decpt.hidden[:frz_indx + 1]:
            for param in layer.parameters():
                param.requires_grad = False

        # # freeze after embedding layer, not necessary but easier for checking
        if emb_mlp_layer < -1:
            for layer in fusion_model.mlp_decpt.hidden[emb_mlp_layer + 1:]:
                for param in layer.parameters():
                    param.requires_grad = False

    fusion_model.fusion_dict['trainable_mlp_emb_layers'] = -freeze_mlp_layers_to - 1
    fusion_model.fusion_dict['trainable_mlp_all_layers'] = 0

    for layer in fusion_model.mlp_decpt.hidden:
        for param in layer.parameters():
            if param.requires_grad:
                fusion_model.fusion_dict['trainable_mlp_all_layers'] += 1
                break

    a = fusion_model.fusion_dict['trainable_mlp_all_layers']
    b = fusion_model.fusion_dict['trainable_mlp_emb_layers']
    assert a == b

    # Chemception
    for param in fusion_model.chemception.parameters():
        param.requires_grad = False

    return None


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
