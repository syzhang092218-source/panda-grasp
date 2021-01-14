from .policy import expert_policy, drag_policy, knock_over_policy, slow_policy, detour1_policy, detour2_policy, near_optimal_policy


POLICY = {
    'expert': expert_policy,
    'drag': drag_policy,
    'knock_over': knock_over_policy,
    'slow': slow_policy,
    'detour1': detour1_policy,
    'detour2': detour2_policy,
    'near_optimal': near_optimal_policy
}
