@torch.jit.script
def compute_reward(
    # termination bonus and early reset penalty: shape (num_envs,)
    success_bonus: torch.Tensor,
    early_reset_penalty_value: torch.Tensor,
    
    # Object-to-goal errors in position and relative orientation
    active_pos: torch.Tensor,       # Shape: (num_envs, 3)
    active_quat: torch.Tensor,      # Shape: (num_envs, 4)  (relative orientation quaternion)
    
    # Fingertip contact information
    n_good_contacts: torch.Tensor,  # Shape: (num_envs,) number of "good" contacts
    n_tips: int                     # total number of fingertips (integer)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    ANALYSIS & IMPROVEMENTS BASED ON POLICY FEEDBACK:
    
    1. Position Reward (pos_reward):
       - Observed values were in a narrow range (0.37 - 0.70).
       - To better differentiate even small improvements, we decrease the temperature.
       - New formulation uses an exponential decay with a lower pos_temp so that errors are penalized more.
       
    2. Orientation Reward (orn_reward):
       - Previous orientation reward using exp(-error^2/orn_temp) yielded nearly constant values.
       - New formulation linearly maps the absolute alignment of the quaternion's scalar part,
         i.e. using w = |q_w|. When w is low (<0.5) it outputs 0 reward and linearly scales to 1 when w reaches 1.
         This provides a wider dynamic range.
       
    3. Contact Reward (contact_reward):
       - Prior contact reward squared the ratio; its values spanned a small range.
       - We now use a cubic transformation on the ratio of good contacts to better reward robust contacts,
         so that fewer contacts result in very low rewards and only high-quality contact yields a strong bonus.
    
    4. Success Bonus and Early Reset Penalty:
       - We keep the scaling for these sparse reward components but re-balance the overall sum.
    
    The total reward is a weighted sum of these components.
    """
    
    # Temperature/scaling parameters for exponential/linear transforms:
    pos_temp: float = 0.0002   # lower temperature for position reward -> sharper decay of reward with error.
    # For orientation, instead of exponential decay we use a linear mapping:
    # when |q_w| (the scalar part) is below 0.5, reward is 0; above that, it scales linearly to 1.
    
    # --- 1. Positional Reward ---
    # Compute L2-norm of the object-goal positional error.
    pos_error: torch.Tensor = torch.norm(active_pos, p=2, dim=-1)  # shape: (num_envs,)
    # Exponential fall-off: perfect alignment (error=0) gives reward 1, larger errors drop quickly.
    pos_reward: torch.Tensor = torch.exp(- (pos_error * pos_error) / pos_temp)
    
    # --- 2. Orientation Reward ---
    # Use the absolute value of the scalar part of the quaternion as a proxy for alignment.
    # A perfect alignment has |q_w| = 1, while lower values indicate misalignment.
    q_w: torch.Tensor = torch.abs(active_quat[:, 3])  # shape: (num_envs,)
    # Linearly map from [0.5, 1.0] -> [0, 1]. Values below 0.5 yield 0.
    orn_reward: torch.Tensor = torch.clamp((q_w - 0.5) / 0.5, min=0.0, max=1.0)
    
    # --- 3. Contact Reward ---
    # Compute the ratio of good contacts to total fingertips.
    good_contact_ratio: torch.Tensor = n_good_contacts.to(torch.float32) / float(n_tips)
    # Apply a cubic transformation so that near-complete contact is rewarded substantially.
    contact_reward: torch.Tensor = good_contact_ratio * good_contact_ratio * good_contact_ratio
    
    # --- 4. Weighting factors ---
    # We re-scale each reward component so that no one term dominates excessively.
    weight_pos: float = 3.0
    weight_orn: float = 2.5
    weight_contact: float = 2.0
    weight_success: float = 50.0
    weight_penalty: float = 1.0
    
    # --- Total Reward ---
    total_reward: torch.Tensor = (
        weight_pos * pos_reward +
        weight_orn * orn_reward +
        weight_contact * contact_reward +
        weight_success * success_bonus -
        weight_penalty * early_reset_penalty_value
    )
    
    # Build a dictionary of individual reward components for monitoring and diagnostics.
    reward_components: Dict[str, torch.Tensor] = {
        "pos_reward": pos_reward,
        "orn_reward": orn_reward,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus * weight_success,
        "early_reset_penalty": early_reset_penalty_value,
    }
    
    return total_reward, reward_components