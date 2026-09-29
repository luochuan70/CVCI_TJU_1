from .common import CategoryRule
from .configs import get_config

def build_rule():
    return CategoryRule(get_config(19))
