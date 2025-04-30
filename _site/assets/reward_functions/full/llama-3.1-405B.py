@torch.jit.script
def compute_reward(
    tip_object_contacts: torch.Tensor,
    obj_pos_handframe: torch.Tensor,
    obj_orn_handframe: torch.Tensor,
    goal_pos_handframe: torch.Tensor,
    goal_orn_handframe: torch.Tensor,
    n_good_contacts: torch.Tensor,
    success_bonus: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pos_temperature = 2.0
    orien_temperature = 2.0
    contact_temperature = 1.0
    reward_object_position = torch.exp(-torch.norm(obj_pos_handframe - goal_pos_handframe, p=2, dim=-1) / pos_temperature)
    reward_object_orientation = torch.exp(-torch.norm(obj_orn_handframe - goal_orn_handframe, p=2, dim=-1) / orien_temperature)
    reward_contact_quality = n_good_contacts.float() / contact_temperature
    reward_success = 10.0 * success_bonus

    total_reward = (
        0.2 * reward_object_position
        + 0.2 * reward_object_orientation
        + 0.1 * reward_contact_quality
        + 0.5 * reward_success
    )
    individual_rewards: Dict[str, torch.Tensor] = {
        "object_position": reward_object_position,
        "object_orientation": reward_object_orientation,
        "contact_quality": reward_contact_quality,
        "success": reward_success,
    }
    return total_reward, individual_rewards