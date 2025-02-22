from src import Pix2Pix, train_step
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
torch.cuda.manual_seed(0)
x = torch.randn(1, 3, 256, 256).to(device)
y = torch.randn(1, 3, 256, 256).to(device)

model = Pix2Pix().to(device)

for i in range(100):
    critic_loss, generator_loss = train_step(model.discriminator, model.generator, model.discriminator_optimizer, model.generator_optimizer, x, y, gp_weight=10)
    print(f"Critic Loss: {critic_loss}, Generator Loss: {generator_loss}")