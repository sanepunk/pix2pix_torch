from src import Pix2Pix, train_step
import torch

x = torch.randn(1, 3, 256, 256)
y = torch.randn(1, 3, 256, 256)

model = Pix2Pix()

for i in range(100):
    critic_loss, generator_loss = train_step(model.discriminator, model.generator, model.discriminator_optimizer, model.generator_optimizer, x, y, gp_weight=10)
    print(f"Critic Loss: {critic_loss}, Generator Loss: {generator_loss}")