from .policy import expert_policy, drag_policy, knock_over_policy, slow_policy, detour_policy


POLICY = {
    'expert': expert_policy,
    'drag': drag_policy,
    'knock_over': knock_over_policy,
    'slow': slow_policy,
    'detour': detour_policy
}
