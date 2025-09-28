import torch.nn as nn
from .ses_conv import SESConv_H_H_1x1, SESConv_H_H, SESConv_Z2_H

def replace_single_layer(network, child, child_name, scales, homos, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= "constant"):
    """
    Replace single layer by SES (Scale Equivariant Steerable) counterpart
    :param network:    network object to replace
    :param child:      network child object
    :param child_name: name
    :param scales:     sesn scales
    :param homos:      homographies
    :param first_conv: handles first conv specially
    :return:
    """
    new_layer = None

    if first_conv:
        # Replace the first conv by SESConv_Z2_H
        in_channels    = child.in_channels
        out_channels   = child.out_channels
        effective_size = child.kernel_size[0]
        kernel_size    = child.kernel_size[0] + 2
        stride         = child.stride[0]
        padding        = kernel_size//2

        if not replace_first_conv_by_1x1_h:
            new_layer      = SESConv_Z2_H(in_channels= in_channels, out_channels= out_channels,
                                      kernel_size= kernel_size, effective_size= effective_size,
                                      scales= scales, stride= stride, padding= padding, bias= False,
                                      homos= homos,
                                      padding_mode= sesn_padding_mode)

        else:
            # Replace by SESConv_H_H_1x1
            scale_size     = 1
            new_layer      = SESConv_H_H_1x1(in_channels= in_channels, out_channels= out_channels,
                                          scale_size= scale_size, stride= stride, bias= False)


    elif isinstance(child, nn.Conv2d):
        in_channels    = child.in_channels
        out_channels   = child.out_channels
        effective_size = child.kernel_size[0]
        stride         = child.stride[0] if len(child.stride)> 1 else child.stride

        if effective_size > 1:
            # Replace by SESConv_H_H
            kernel_size    = child.kernel_size[0] + 2
            padding        = kernel_size//2

            new_layer      = SESConv_H_H(in_channels= in_channels, out_channels= out_channels,
                                         scale_size= 1, kernel_size= kernel_size,
                                         effective_size= effective_size, scales= scales,
                                         homos = homos, homos_size= 1,
                                         stride= stride, padding= padding,
                                         bias= False, padding_mode= sesn_padding_mode)

        else:
            # Replace by SESConv_H_H_1x1
            scale_size     = 1

            new_layer      = SESConv_H_H_1x1(in_channels= in_channels, out_channels= out_channels,
                                          scale_size= scale_size, stride= stride, bias= False)

    elif isinstance(child, nn.BatchNorm2d):
        # Replace by BatchNorm3d
        num_features        = child.num_features
        eps                 = child.eps
        momentum            = child.momentum
        affine              = child.affine
        track_running_stats = child.track_running_stats
        new_layer           = nn.BatchNorm3d(num_features= num_features, eps= eps, momentum= momentum,
                                             affine= affine, track_running_stats= track_running_stats)

    elif isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d):
        kernel_size    = (1, child.kernel_size, child.kernel_size)
        stride         = (1, child.stride, child.stride)
        padding        = (0, child.padding, child.padding)
        ceil_mode      = child.ceil_mode
        if isinstance(child, nn.MaxPool2d):
            # Replace by MaxPool3d
            dilation       = child.dilation
            return_indices = child.return_indices
            new_layer      = nn.MaxPool3d(kernel_size= kernel_size, stride= stride, padding= padding,
                                      dilation= dilation, return_indices= return_indices, ceil_mode= ceil_mode)
        else:
            # Replace by AvgPool3d
            new_layer      = nn.AvgPool3d(kernel_size= kernel_size, stride= stride, padding= padding,
                                      ceil_mode= ceil_mode)

    if new_layer is not None:
        # Replace the layer
        setattr(network, child_name, new_layer)
        return True
    else:
        return False

def get_child_names_of_network(network):
    network_child_names = []
    for child_name, child in network.named_children():
        network_child_names.append(child_name)

    return network_child_names