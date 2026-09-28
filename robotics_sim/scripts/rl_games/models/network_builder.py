import copy

from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder

from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import  SquashedNormal
from rl_games.common.layers.recurrent import  GRUWithDones, LSTMWithDones
from rl_games.common.layers.value import  TwoHotEncodedValue, DefaultValue
from rl_games.algos_torch.spatial_softmax import SpatialSoftArgmax


class ResNetInferenceModule(nn.Module):
    def __init__(self, res_model, res_inference_fn):
        super(ResNetInferenceModule, self).__init__()
        self.res_model = res_model
        self.res_inference_fn = res_inference_fn

    def forward(self, x):
        # ResNet 모델과 reset 함수로 추론
        return self.res_inference_fn(self.res_model, x)


class A2CIntegralResnetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            # 이부분 수정 필요
            cnn_output_size = self._calc_input_size(input_shape, self.actor_cnn)
            
            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                # default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]
                cnn_output_size = 1000
                # self.actor_cnn = self._build_conv(**cnn_args)
            cnn_output_size = 1000

            mlp_input_size = cnn_output_size
            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    if self.rnn_concat_input:
                        rnn_in_size += cnn_output_size

                    out_size = self.rnn_units
                    if self.rnn_concat_output:
                        out_size += cnn_output_size
                else:
                    rnn_in_size = cnn_output_size

                    mlp_input_size = self.rnn_units
                    if self.rnn_concat_output:
                        mlp_input_size += cnn_output_size

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : mlp_input_size,
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def _preproc_res(self, obs_batch):
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
            return obs_batch

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            # self.res_model.eval()

            # if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                # if self.permute_input and len(obs.shape) == 4:
                #     obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                out = self.actor_cnn(obs)
                # out = self._preproc_res(out)
                a_out = c_out = out

                a_out = a_out.contiguous().view(a_out.size(0), -1)
                c_out = c_out.contiguous().view(c_out.size(0), -1)

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    a_cnn_out = a_out
                    c_cnn_out = c_out
                    if not self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_cnn_out)
                        c_out = self.critic_mlp(c_cnn_out)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_cnn_out], dim=1)
                            c_out = torch.cat([c_out, c_cnn_out], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.rnn_concat_output:
                        a_out = torch.cat([a_out, a_cnn_out], dim=1)
                        c_out = torch.cat([c_out, c_cnn_out], dim=1)

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)
                # out = self._preproc_res(out)

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    cnn_out = out
                    if not self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, cnn_out], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.rnn_concat_output:
                        out = torch.cat([out, cnn_out], dim=1)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)
                self.rnn_concat_output = params['rnn'].get('concat_output', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False

        def _prepare_resnet_model(self, model_name: str) -> dict:
            """Prepare the ResNet model for inference.

            Args:
                model_name: The name of the ResNet model to prepare.
                model_device: The device to store and infer the model on.

            Returns:
                A dictionary containing the model and inference functions.
            """
            from torchvision import models

            def _load_model() -> torch.nn.Module:
                """Load the ResNet model."""
                # map the model name to the weights
                resnet_weights = {
                    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                    "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                    "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                    "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
                }

                # load the model
                model = getattr(models, model_name)(weights=resnet_weights[model_name])
                return model

            def _inference(model, images: torch.Tensor) -> torch.Tensor:
                """Inference the ResNet model.

                Args:
                    model: The ResNet model.
                    images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

                Returns:
                    The extracted features tensor. Shape is (num_envs, feature_dim).
                """
                # move the image to the model device
                image_proc = images
                # permute the image to (num_envs, channel, height, width)
                image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0

                device = next(model.parameters()).device
                # normalize the image
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                image_proc = (image_proc - mean) / std

                # forward the image through the model
                return model(image_proc)

            # return the model, preprocess and inference functions
            return {"model": _load_model, "inference": _inference}

    def build(self, name, **kwargs):
        net = A2CIntegralResnetBuilder.Network(self.params, **kwargs)
        return net
