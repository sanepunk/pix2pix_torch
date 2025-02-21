import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        residual = self.block_1(x)
        x = self.block_2(residual)
        x = self.block_3(torch.concat([x, residual], dim = 1))
        return x


class DownSample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, max_pooling_size, stride, padding):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
            ResidualBlock(out_features, out_features, stride, kernel_size, padding)
        )

        self.max_pooling = nn.MaxPool2d(kernel_size=max_pooling_size)

    def forward(self, x):
        x = self.block(x)
        print("downsample main", x.shape)
        return self.max_pooling(x), x


class UpSample(nn.Module):
    def __init__(self, deep_channels, in_features, out_features, kernel_size, stride, padding):
        super(UpSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features + deep_channels, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
            ResidualBlock(out_features, out_features, stride, kernel_size, padding)
        )
    
    def forward(self, x, skip_connection):
        x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        print("upsample main", x.shape, skip_connection.shape)
        x = torch.concat([x, skip_connection], dim=1)
        return self.block(x)
    
def generator_loss(critic: nn.Module, generator: nn.Module, input_image: torch.Tensor):
    fake = generator(input_image)
    fake_pred = critic(fake)
    return -torch.mean(fake_pred)

def critic_loss(critic: nn.Module, generator: nn.Module, input_image: torch.Tensor, real_image: torch.Tensor, gp_weight: int):
    fake = generator(input_image)
    fake_score = critic(fake)
    real_score = critic(real_image)
    wassserstein_loss = torch.mean(fake_score) - torch.mean(real_score)
    epsilon = torch.rand(real_image.size(0), 1, 1, 1, device=real_image.device)
    interpolated = epsilon * real_image + (1 - epsilon) * fake
    interpolated.detach().requires_grad_()
    score = critic(interpolated)
    gradients = torch.autograd.grad(outputs=score, inputs=interpolated, grad_outputs=torch.ones_like(score), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return wassserstein_loss + gradient_penalty

def train_step(critic: nn.Module, generator: nn.Module, critic_optimizer: torch.optim, generator_optimizer: torch.optim, input_image: torch.Tensor, real_output_image: torch.Tensor, gp_weight: int):
    critic_optimizer.zero_grad()
    critic_loss_value = critic_loss(critic, generator, input_image, real_output_image, gp_weight)
    critic_loss_value.backward()
    critic_optimizer.step()

    generator_optimizer.zero_grad()
    generator_loss_value = generator_loss(critic, generator, input_image)
    generator_loss_value.backward()
    generator_optimizer.step()

    return critic_loss_value, generator_loss_value