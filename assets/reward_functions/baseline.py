@torch.jit.script
def compute_reward(
    # List of variables...
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # ROTATION REWARD
    # cosine distance between obj and goal orientation
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(goal_base_orn))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    rot_rew = (1.0 / (torch.abs(rot_dist) + rot_eps))

    # delta rotation reward
    rot_quat_diff = quat_mul(obj_base_orn, quat_conjugate(prev_obj_orn))
    rpy_diff = torch.stack(get_euler_xyz(rot_quat_diff), dim=1)
    rpy_diff = torch.where(rpy_diff > torch.pi, rpy_diff - 2*torch.pi, rpy_diff)
    delta_rot = torch.sum(rpy_diff * target_pivot_axel, dim=1) 
    # delta_rot = torch.sum(rpy_diff * current_pivot_axel, dim=1) 
    delta_rot_rew = torch.clamp(delta_rot, min=delta_rot_clip_min, max=delta_rot_clip_max)
    # print(delta_rot_rew[0])
    # print('rotated_angle: ', delta_rot[0])

    # KEYPOINT REWARD
    # distance between obj and goal keypoints
    kp_deltas = torch.norm(obj_kps - goal_kps, p=2, dim=-1)
    mean_kp_dist = kp_deltas.mean(dim=-1)
    kp_rew = lgsk_kernel(kp_deltas, scale=kp_lgsk_scale, eps=kp_lgsk_eps).mean(dim=-1)
    # print('key point reward: ', kp_rew[0])

    # ANGVEL REWARD
    # bound and scale rewards such that they are in similar ranges
    obj_angvel_about_axis = torch.sum(obj_angvel * target_pivot_axel, dim=1)
    av_rew = torch.clamp(obj_angvel_about_axis, min=av_clip_min, max=av_clip_max)

    # HAND SMOOTHNESS
    # Penalty for deviating from the original grasp pose by too much
    hand_pose_penalty = -torch.norm(current_joint_pos - init_joint_pos, p=2, dim=-1)
    # Penalty for high torque
    torque_penalty = -torch.norm(current_joint_eff, p=2, dim=-1)
    # Penalty for high work
    work_penalty = -torch.sum(torch.abs(current_joint_eff * current_joint_vel), dim=-1)
    # angular velocity penalty masked for over the desired av
    obj_angvel_mag = torch.norm(obj_angvel, p=2, dim=-1)
    av_penalty = (obj_angvel_mag > desired_max_av) * -torch.sqrt((obj_angvel_mag - desired_max_av)**2) + \
        (obj_angvel_mag  < desired_min_av) * -torch.sqrt((desired_min_av - obj_angvel_mag)**2) 

    # OBJECT SMOOTHNESS
    # distance between obj and goal COM
    com_dist_rew = -torch.norm(obj_base_pos - goal_base_pos, p=2, dim=-1)
    # Penalty for object linear velocity
    obj_linvel_penalty = -torch.norm(obj_linvel, p=2, dim=-1)
    # Penalty for axis deviation
    cos_dist = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axis_cos_dist = -(1.0 - cos_dist)
    axis_deviat_angle = torch.arccos(cos_dist)
    # print(cos_dist)

    # PRECISION GRASP
    # Penalise tip to obj distance masked for non-contacted tips
    total_finger_tip_obj_dist = -torch.sum((tip_object_contacts == 0)*finger_tip_obj_dist, dim=-1)
    # print(total_finger_tip_obj_dist)

    # Penalise contact pose if not in normal direction: maximum contact penalty if tip is not in contact
    contact_normal_penalty = torch.abs(contact_pose).sum(-1)
    contact_normal_penalty = -torch.where(tip_object_contacts == 0, torch.pi, contact_normal_penalty).sum(-1)

    # Good contact normal reward, award envs for normal contact and tip contact >=2
    contact_normal_rew = torch.abs(contact_pose).sum(-1)
    contact_normal_rew = torch.where(tip_object_contacts== 0, 0.0, torch.pi - contact_normal_rew).sum(-1)
    contact_normal_rew = torch.where(n_tip_contacts >= 2, contact_normal_rew/(n_tip_contacts * torch.pi), 0.0)

    # Penalise if total tip contact force is below 2x obj mass
    total_tip_contact_force = torch.sum(contact_force_mags, dim=-1)
    tip_force_penalty = total_tip_contact_force - obj_masses.squeeze() * 2.0 * 10
    tip_force_penalty = torch.where(tip_force_penalty > 0, torch.zeros_like(tip_force_penalty), tip_force_penalty)

    # Reward curriculum: zero penalties below rotation
    lamda_good_contact *= lambda_reward_curriculum
    lamda_bad_contact *= lambda_reward_curriculum
    lambda_pose_penalty *= lambda_reward_curriculum
    lambda_work_penalty *= lambda_reward_curriculum
    lambda_torque_penalty *= lambda_reward_curriculum
    lambda_com_dist *= lambda_reward_curriculum
    lambda_linvel_penalty *= lambda_reward_curriculum
    lambda_av_penalty *= lambda_reward_curriculum
    lambda_contact_normal_penalty *= lambda_reward_curriculum
    lambda_contact_normal_rew *= lambda_reward_curriculum
    lambda_tip_force_penalty *= lambda_reward_curriculum

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = \
        lambda_rot * rot_rew + \
        lambda_delta_rot * delta_rot_rew + \
        lambda_kp * kp_rew + \
        lambda_av * av_rew + \
        lambda_pose_penalty * hand_pose_penalty + \
        lambda_torque_penalty * torque_penalty + \
        lambda_work_penalty * work_penalty + \
        lambda_av_penalty * av_penalty + \
        lambda_com_dist * com_dist_rew + \
        lambda_linvel_penalty * obj_linvel_penalty + \
        lambda_axis_cos_dist * axis_cos_dist + \
        lambda_tip_obj_dist* total_finger_tip_obj_dist + \
        lambda_contact_normal_penalty * contact_normal_penalty + \
        lambda_contact_normal_rew * contact_normal_rew + \
        lambda_tip_force_penalty * tip_force_penalty
    
    # add reward for contacting with tips
    # total_reward = torch.where(n_tip_contacts >= 2, total_reward + lamda_good_contact, total_reward)
    total_reward = torch.where(n_tip_contacts >= 2, total_reward + (n_tip_contacts - 1) * lamda_good_contact, total_reward)    # Alternative good contact reward hybrid of dense and binary
    # total_reward += n_tip_contacts * lamda_good_contact         # good contact dense
    # total_reward = torch.where(n_good_contacts >= 2, total_reward + (n_good_contacts - 1) * lamda_good_contact, total_reward)    # Alternative good contact reward hybrid of dense and binary
    
    # add penalty for contacting with links other than the tips
    total_reward = torch.where(n_non_tip_contacts > 0, total_reward - lamda_bad_contact, total_reward)
    # total_reward = torch.where(n_non_tip_contacts > 0, total_reward - (lamda_bad_contact * n_non_tip_contacts), total_reward)     # Bad contact dense

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    total_reward = torch.where(mean_kp_dist <= success_tolerance, total_reward + reach_goal_bonus, total_reward)

    # Fall or deviation penalty: distance to the goal or target axis is larger than a threashold or if no tip is in contact
    early_reset_cond = torch.logical_or(mean_kp_dist >= fall_reset_dist, axis_deviat_angle >= axis_deviat_reset_dist)
    early_reset_cond = torch.logical_or(early_reset_cond, n_tip_contacts == 0)
    total_reward = torch.where(early_reset_cond, total_reward - early_reset_penalty, total_reward)

    # Debug: first env check termination conditions
    # if (mean_kp_dist >= fall_reset_dist)[0]:
    #     print('fallen')
    # if (axis_deviat_angle >= axis_deviat_reset_dist)[0]:
    #     print('deviated')

    # zero reward when less than 2 tips in contact
    if require_contact:
        rew_cond_1 = n_tip_contacts < 1
        rew_cond_2 = axis_deviat_angle >= 0.5
        total_reward = torch.where(torch.logical_or(rew_cond_1, rew_cond_2), torch.zeros_like(rew_buf), total_reward)
    
    # zero reward if more than 2 bad/non-tip contacts
    if require_n_bad_contacts:
        total_reward = torch.where(n_non_tip_contacts > 2, torch.zeros_like(rew_buf), total_reward)

    # total_reward = torch.where(n_tip_contacts < 1, torch.zeros_like(rew_buf), total_reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(mean_kp_dist <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(early_reset_cond, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # find average consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    # Find the number of rotations over all the envs
    rotation_counts = rotation_counts + delta_rot/3.14

    info: Dict[str, torch.Tensor] = {

        'successes': successes,
        'successes_cons': cons_successes,
        'rotation_counts': rotation_counts,
        'angvel_mag': obj_angvel_mag,

        'total_finger_obj_dist': total_finger_tip_obj_dist,
        'num_tip_contacts': n_tip_contacts,
        'num_non_tip_contacts': n_non_tip_contacts,
        'num_good_contacts': n_good_contacts,
        'contact_normal_penalty': contact_normal_penalty, 
        'contact_normal_reward': contact_normal_rew, 
        'tip_force_penalty': tip_force_penalty,

        'reward_rot': rot_rew,
        'reward_delta_rot': delta_rot,
        'reward_keypoint': kp_rew,
        'reward_angvel': av_rew,
        'reward_total': total_reward,

        'penalty_hand_pose': hand_pose_penalty,
        'penalty_hand_torque': torque_penalty,
        'penalty_hand_work': work_penalty,
        'penalty_angvel': av_penalty,

        'reward_com_dist': com_dist_rew,
        'penalty_obj_linvel': obj_linvel_penalty,
        'penalty_axis_cos_dist': axis_cos_dist,
    }

    return total_reward, resets, goal_resets, successes, cons_successes, rotation_counts, info