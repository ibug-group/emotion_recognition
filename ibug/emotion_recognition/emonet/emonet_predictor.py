import os
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, Dict
from .emonet import EmoNet


__all__ = ['EmoNetPredictor']


class EmoNetPredictor(object):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None,
                 config: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = EmoNetPredictor.get_model()
        if config is None:
            config = EmoNetPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = EmoNet(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(
                1, self.config.num_input_channels, self.config.input_size, self.config.input_size).to(self.device))

    @staticmethod
    def create_config(use_jit: bool = True) -> SimpleNamespace:
        return SimpleNamespace(use_jit=use_jit)

    @staticmethod
    def get_model(name: str = 'emonet248') -> SimpleNamespace:
        name = name.lower()
        if name == 'emonet248':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet248.pth'),
                                   config=SimpleNamespace(num_input_channels=768, input_size=64, n_blocks=4, n_reg=2,
                                                          emotion_labels=('neutral', 'happy', 'sad', 'surprise',
                                                                          'fear', 'disgust', 'anger', 'contempt')))
        elif name == 'emonet245':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet245.pth'),
                                   config=SimpleNamespace(num_input_channels=768, input_size=64, n_blocks=4, n_reg=2,
                                                          emotion_labels=('neutral', 'happy', 'sad',
                                                                          'surprise', 'anger')))
        if name == 'emonet248_alt':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet248_alt.pth'),
                                   config=SimpleNamespace(num_input_channels=768, input_size=64, n_blocks=4, n_reg=2,
                                                          emotion_labels=('neutral', 'happy', 'sad', 'surprise',
                                                                          'fear', 'disgust', 'anger', 'contempt')))
        elif name == 'emonet245_alt':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet245_alt.pth'),
                                   config=SimpleNamespace(num_input_channels=768, input_size=64, n_blocks=4, n_reg=2,
                                                          emotion_labels=('neutral', 'happy', 'sad',
                                                                          'surprise', 'anger')))
        else:
            raise ValueError("name must be set to either emonet248 or emonet245")

    @torch.no_grad()
    def __call__(self, fan_features: torch.Tensor) -> Dict:
        if fan_features.numel() > 0:
            results = self.net(fan_features.to(self.device)).cpu().numpy()
            return {'emotion': np.argmax(results[:, :-2], axis=1).astype(int),
                    'valence': results[:, -2].copy(),
                    'arousal': results[:, -1].copy(),
                    'raw_results': results}
        else:
            return {'emotion': np.empty(shape=(0,), dtype=int),
                    'valence': np.empty(shape=(0,), dtype=np.float32),
                    'arousal': np.empty(shape=(0,), dtype=np.float32),
                    'raw_results': np.empty(shape=(0, len(self.config.emotion_labels) + self.config.n_reg),
                                            dtype=np.float32)}
