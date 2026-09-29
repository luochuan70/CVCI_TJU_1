"""Ordinary 220 category rules.

Each ``class_XX`` module is an explicit category entry point.  The shared
implementation only supplies bounded control primitives; category parameters
are selected after the radar classifier has locked a class.
"""

from .router import OrdinaryRuleRouter

__all__ = ["OrdinaryRuleRouter"]

