"""PPO math shared by the combat experiment and focused correctness tests."""
import torch


def advantages(rewards, dones, values, bootstrap, gamma=.99, lam=.95):
    result=torch.zeros_like(rewards)
    carry=torch.zeros_like(bootstrap)
    following=bootstrap
    for t in reversed(range(len(rewards))):
        alive=~dones[t]
        delta=rewards[t]+gamma*alive*following-values[t]
        carry=delta+gamma*lam*alive*carry
        result[t]=carry
        following=values[t]
    return result,result+values


def clipped_actor_loss(logp, old_logp, advantage, clip=.2):
    ratio=(logp-old_logp).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
