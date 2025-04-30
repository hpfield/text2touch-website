@torch.jit.script
def compute_reward(
    obj_base_pos: torch.Tensor,
    goal_base_pos: torch.Tensor,
    obj_base_orn: torch.Tensor,
    goal_base_orn: torch.Tensor,
    n_good_contacts: torch.Tensor,
    n_non_tip_contacts: torch.Tensor,
    success_bonus: torch.Tensor,
    tip_object_contacts: torch.Tensor,
    obj_kp_positions_centered: torch.Tensor,
    goal_kp_positions_centered: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Temperature parameters
    kp_temp = torch.tensor(5.0, device=obj_base_pos.device)
    orn_temp = torch.tensor(3.5, device=obj_base_pos.device)
    contact_temp = torch.tensor(1.2, device=obj_base_pos.device)  # Increased from 0.8
    non_tip_temp = torch.tensor(0.015, device=obj_base_pos.device)  # Reduced penalty
    success_temp = torch.tensor(35.0, device=obj_base_pos.device)  # Increased from 30
    
    # Keypoint alignment
    kp_dist = torch.norm(obj_kp_positions_centered - goal_kp_positions_centered, dim=-1).mean(dim=1)
    kp_reward = torch.exp(-kp_temp * kp_dist)

    # Orientation alignment using geodesic distance
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(goal_base_orn))
    rot_angle = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], dim=1), max=1.0))
    orn_reward = torch.exp(-orn_temp * rot_angle)

    # Contact quality with increased emphasis
    active_tips = tip_object_contacts.sum(dim=1)
    contact_quality = n_good_contacts.float() / (active_tips + 1e-6)
    contact_reward = contact_temp * torch.tanh(6.0 * contact_quality)  # Increased tanh scaling

    # Adjusted non-tip penalty
    non_tip_penalty = -non_tip_temp * torch.pow(n_non_tip_contacts.float(), 1.2)  # Less severe for few contacts

    # Enhanced success bonus
    scaled_success = success_temp * success_bonus + 0.2 * (kp_reward + orn_reward)

    total_reward = kp_reward + orn_reward + contact_reward + non_tip_penalty + scaled_success

    reward_components = {
        "kp_reward": kp_reward,
        "orn_reward": orn_reward,
        "contact_reward": contact_reward,
        "non_tip_penalty": non_tip_penalty,
        "scaled_success": scaled_success
    }

    return total_reward, reward_components