import wandb
run = wandb.Api().run("krrishbhasin-bits-pilani/hw1-imitation/4fwy13e5")
run.history(keys=["train/loss"], samples=10000).to_csv("mse_loss.csv")