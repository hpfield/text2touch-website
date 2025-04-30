@torch.jit.script
def compute_reward(
    success_bonus: torch.Tensor, 
    early_reset_penalty_value: torch.Tensor, 
    contact_pose_range_sim: torch.Tensor,
    base_hand_pos: torch.Tensor,
    base_hand_orn: torch.Tensor,
    kp_dist: float,
    n_keypoints: int,
    obj_kp_positions: torch.Tensor,
    goal_kp_positions: torch.Tensor,
    kp_basis_vecs: torch.Tensor,
    fingertip_pos_handframe: torch.Tensor,
    fingertip_orn_handframe: torch.Tensor,
    thumb_tip_name_idx: int,
    index_tip_name_idx: int,
    middle_tip_name_idx: int,
    pinky_tip_name_idx: int,
    n_tips: int,
    contact_positions: torch.Tensor,
    contact_positions_worldframe: torch.Tensor,
    contact_positions_tcpframe: torch.Tensor,
    sim_contact_pose_limits: torch.Tensor,
    contact_threshold_limit: float,
    obj_indices: torch.Tensor,
    goal_indices: torch.Tensor,
    default_obj_pos_handframe: torch.Tensor,
    prev_obj_orn: torch.Tensor,
    goal_displacement_tensor: torch.Tensor,
    root_state_tensor: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    rigid_body_tensor: torch.Tensor,
    current_force_apply_axis: torch.Tensor,
    obj_force_vector: torch.Tensor,
    pivot_axel_worldframe: torch.Tensor,
    pivot_axel_objframe: torch.Tensor,
    goal_base_pos: torch.Tensor,
    goal_base_orn: torch.Tensor,
    net_tip_contact_forces: torch.Tensor,
    net_tip_contact_force_mags: torch.Tensor,
    tip_object_contacts: torch.Tensor,
    n_tip_contacts: torch.Tensor,
    n_non_tip_contacts: torch.Tensor,
    thumb_tip_contacts: torch.Tensor,
    index_tip_contacts: torch.Tensor,
    middle_tip_contacts: torch.Tensor,
    pinky_tip_contacts: torch.Tensor,
    fingertip_pos: torch.Tensor,
    fingertip_orn: torch.Tensor,
    fingertip_linvel: torch.Tensor,
    fingertip_angvel: torch.Tensor,
    tip_contact_force_pose: torch.Tensor,
    tip_contact_force_pose_low_dim: torch.Tensor,
    tip_contact_force_pose_bins: torch.Tensor,
    n_good_contacts: torch.Tensor,
    hand_joint_pos: torch.Tensor,
    hand_joint_vel: torch.Tensor,
    obj_base_pos: torch.Tensor,
    obj_base_orn: torch.Tensor,
    obj_pos_handframe: torch.Tensor,
    obj_orn_handframe: torch.Tensor,
    obj_displacement_tensor: torch.Tensor,
    obj_pos_centered: torch.Tensor,
    delta_obj_orn: torch.Tensor,
    obj_base_linvel: torch.Tensor,
    obj_base_angvel: torch.Tensor,
    obj_linvel_handframe: torch.Tensor,
    obj_angvel_handframe: torch.Tensor,
    goal_pos_centered: torch.Tensor,
    goal_pos_handframe: torch.Tensor,
    goal_orn_handframe: torch.Tensor,
    active_pos: torch.Tensor,
    active_quat: torch.Tensor,
    obj_kp_positions_centered: torch.Tensor,
    goal_kp_positions_centered: torch.Tensor,
    active_kp: torch.Tensor,
    obj_force_vector_handframe: torch.Tensor,
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