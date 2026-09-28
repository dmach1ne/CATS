import copy

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import model_builder
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.algos_torch import central_value

import torch
from torch import nn
from torch import optim

from .network_builder import A2CIntegralResnetBuilder

import numpy as np


class ResNetInferenceModule(nn.Module):
    def __init__(self, res_model, res_inference_fn):
        super(ResNetInferenceModule, self).__init__()
        self.res_model = res_model
        self.res_inference_fn = res_inference_fn
        self.res_model.eval()

    def forward(self, x):
        # ResNet 모델과 reset 함수로 추론
        return self.res_inference_fn(self.res_model, x)


class A2CIRAgent(A2CAgent):
    def __init__(self, base_name, params):
        # super().__init__(base_name, params)

        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        obs_shape = (1000,)
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

        #resnet
        model_config = self._prepare_resnet_model("resnet18")
        self.res_model = model_config["model"]()
        self.res_reset_fn = model_config.get("reset")
        self.res_inference_fn = model_config["inference"]
        self.preproc_cnn = ResNetInferenceModule(self.res_model, self.res_inference_fn)
        self.preproc_cnn.to(self.ppo_device)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        builder.network_builder.network_factory.register_builder('integral_resnet_actor_critic', lambda **kwargs: A2CIntegralResnetBuilder())
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def _preproc_obs(self, obs_batch):
        feature = self.preproc_cnn(obs_batch).detach()
        if type(feature) is dict:
            feature = copy.copy(feature)
            for k,v in feature.items():
                if v.dtype == torch.uint8:
                    feature[k] = v.float() / 255.0
                else:
                    feature[k] = v
        else:
            if feature.dtype == torch.uint8:
                feature = feature.float() / 255.0
        return feature
    
    # def env_step(self, actions):
    #     actions = self.preprocess_actions(actions)
    #     obs, rewards, dones, infos = self.vec_env.step(actions)

    #     print(f"obs_min : {obs.amin(dim=(0, 1, 2))}, obs_max : {obs.amax(dim=(0, 1, 2))}")

    #     if self.is_tensor_obses:
    #         if self.value_size == 1:
    #             rewards = rewards.unsqueeze(1)
    #         return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
    #     else:
    #         if self.value_size == 1:
    #             rewards = np.expand_dims(rewards, axis=1)
    #         return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

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
                model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
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
                # print(f"img : {image_proc[60,34:50,34:50,:]}")
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
