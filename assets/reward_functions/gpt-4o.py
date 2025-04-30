@torch.jit.script
def compute_reward(
    # List of variables...
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Initialize temperatures for reward components
    temperature_pos = 0.8  # Increased temperature
    temperature_orn = 3.0  # Further increased temperature
    temperature_contact = 0.5  
    temperature_good_contacts = 0.7  
    temperature_success = 25.0  # Steady weight for success bonus

    # Distance to goal position reward
    dist_to_goal = torch.norm(active_pos, p=2, dim=-1)
    normalized_dist = torch.exp(-temperature_pos * dist_to_goal)  # Increased temperature
    reward_goal_pos = 0.5 * normalized_dist  # Adjusted weight

    # Orientation alignment to goal reward
    quat_diff = torch.norm(active_quat, p=2, dim=-1)
    normalized_orn = torch.exp(-temperature_orn * quat_diff)  # Increased temperature
    reward_goal_orn = 0.3 * normalized_orn  # Adjusted weight

    # Fingertip contact reward
    contact_reward = torch.sum(tip_object_contacts, dim=-1).float()
    reward_fingertip_contact = torch.log(1.0 + temperature_contact * contact_reward)  # Scaled component

    # Reward for maximizing good contacts
    reward_good_contacts = torch.log(1.0 + temperature_good_contacts * n_good_contacts.float())  # Scaled component

    # Composite reward
    total_reward = reward_goal_pos + reward_goal_orn + reward_fingertip_contact + reward_good_contacts

    # Add success bonuses and penalties
    total_reward += temperature_success * success_bonus  # Constant steady weight for success bonus
    total_reward -= early_reset_penalty_value  # Keep the penalty constant
    
    reward_dict = {
        'reward_goal_pos': reward_goal_pos,
        'reward_goal_orn': reward_goal_orn,
        'reward_fingertip_contact': reward_fingertip_contact,
        'reward_good_contacts': reward_good_contacts,
        'success_bonus': temperature_success * success_bonus,  # Include scaled success bonus
        'early_reset_penalty': -early_reset_penalty_value,
    }

    return total_reward, reward_dict