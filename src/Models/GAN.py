from torch import nn
import torch
from .generator import Generator
from .discriminator import Discriminator
import json
from typing import Tuple

class Pix2Pix(nn.Module):
    def __init__(self, in_features: int = 3, kernel_size: Tuple[int, int] = (3, 3), maxpool_size: Tuple[int, int] = (2, 2), stride: Tuple[int, int] = (1, 1), padding: int = 1, path: str = None):
        super(Pix2Pix, self).__init__()
        try:
            self.generator = Generator(in_features, kernel_size, maxpool_size, stride, padding)
            self.discriminator = Discriminator(in_features, kernel_size, maxpool_size, stride, padding)
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.g = [in_features, kernel_size, maxpool_size, stride, padding]
            try:
                if path:
                    try:
                        with open(path + '/config.json', 'r') as file:
                            config = json.load(file)
                        for stored_values, check_values in zip(config, self.g):
                            assert(stored_values == check_values, f"Stored values {stored_values} do not match the values {check_values}")
                        self.generator.load_state_dict(torch.load(path + '/generator.pth', weights_only=True))
                        self.discriminator.load_state_dict(torch.load(path + '/discriminator.pth', weights_only=True))
                        self.generator_optimizer.load_state_dict(torch.load(path + '/generator_optimizer.pth', weights_only=True))
                        self.discriminator_optimizer.load_state_dict(torch.load(path + '/discriminator_optimizer.pth', weights_only=True))
                    except Exception as e: # If the file does not exist or is corrupt
                        print(f"Error: {e}")

            except Exception as e: # If the path does not exist
                print(f"Error: {e}")

        except Exception as e:
            print(f"Error: {e}")
        
    def generate(self, x):
        x =  self.generator(x)
        x = torch.clamp(x, -1, 1)
        return x
    
    def save(self, path: str):
        try:
            with open(path + '/config.json', 'w') as file:
                json.dump(self.g, file)
            torch.save(self.generator.state_dict(), path + '/generator.pth')
            torch.save(self.discriminator.state_dict(), path + '/discriminator.pth')
            torch.save(self.generator_optimizer.state_dict(), path + '/generator_optimizer.pth')
            torch.save(self.discriminator_optimizer.state_dict(), path + '/discriminator_optimizer.pth')
        except Exception as e:
            print(f"Error: {e}")
    

    
