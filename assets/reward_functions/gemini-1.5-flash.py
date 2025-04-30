@torch.jit.script
def compute_reward(
    # List of variables...
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    device = obj_base_pos.device

    #Revised scaling based on analysis.  Focus on balancing components.
    pos_scale = 2.0
    orn_scale = 5.0
    contact_scale = 1.0
    sparse_reward_scale = 20.0 #Increased to make sparse reward more impactful.

    #Temperature parameters adjusted for better sensitivity.
    pos_temp = 0.5
    orn_temp = 1.0
    contact_temp = 1.0 #Added temperature for contact reward.

    #Reward shaping with improved scaling and temperature parameters.
    pos_reward = torch.exp(-torch.norm(active_pos, p=2, dim=-1)**2 * pos_scale / pos_temp)
    orn_reward = torch.exp(-torch.norm(active_quat[..., :3], p=2, dim=-1)**2 * orn_scale / orn_temp)
    contact_reward = torch.tanh(n_good_contacts / n_tips * contact_scale / contact_temp) #Normalized by n_tips

    #Sparse reward adjusted for increased scale.
    dist_to_goal = torch.norm(active_pos, dim=-1)
    sparse_reward = torch.where(dist_to_goal < 0.05, success_bonus * sparse_reward_scale, torch.zeros_like(success_bonus))

    #Combine rewards with adjusted weights
    total_reward = 2 * pos_reward + 3 * orn_reward + contact_reward + sparse_reward

    #Early reset penalty reduced to avoid dominance.
    early_reset_penalty = torch.where(n_good_contacts == 0, early_reset_penalty_value * 0.001, torch.zeros_like(early_reset_penalty_value))
    total_reward -= early_reset_penalty

    reward_components = {
        "pos_reward": pos_reward,
        "orn_reward": orn_reward,
        "contact_reward": contact_reward,
        "sparse_reward": sparse_reward,
        "early_reset_penalty": early_reset_penalty,
    }

    return total_reward, reward_components