from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.algos_torch import model_builder
from rl_games.common.player import BasePlayer
import torch

from .network_builder import A2CIntegralResnetBuilder


class PpoPlayerContinuousIr(PpoPlayerContinuous):
    def __init__(self, params):
        print("플레이어 init")
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        self.obs_shape = (1000,)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        print(f"network model : {self.model}")

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        # 빌더의 network 빌더에 integral_resnet_actorcritic 추가
        builder.network_builder.network_factory.register_builder('integral_resnet_actor_critic', lambda **kwargs: A2CIntegralResnetBuilder())
        self.config['network'] = builder.load(params)
