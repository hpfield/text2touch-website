@torch.jit.script
def compute_reward(
    # termination penalty and success bonus
    success_bonus: torch.Tensor, # To be scaled and added to the final reward.
    early_reset_penalty_value: torch.Tensor, # To be scaled and subtracted from the final reward.

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