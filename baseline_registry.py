import collections
import imp
from typing import Optional, Type

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
    
class Registry(metaclass=Singleton):
    mapping = collections.defaultdict(dict)

    @classmethod
    def _register_impl(cls, _type, to_register, name, assert_type=None):
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)
        
    @classmethod
    def _get_impl(cls, _type, name):
        return cls.mapping[_type].get(name, None)
    
registry = Registry()

class BaselineRegistry(Registry):
    
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from common.base_trainer import BaseRLTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseRLTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)


    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL policy with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.rl.ppo.policy import Policy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyPolicy(Policy):
                pass


            # or

            @baseline_registry.register_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass

        """
        from rl.ppo.policy import Policy

        return cls._register_impl(
            "policy", to_register, name, assert_type=Policy
        )

    @classmethod
    def get_policy(cls, name: str):
        r"""Get the RL policy with :p:`name`."""
        return cls._get_impl("policy", name)

    @classmethod
    def register_perception_net(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a Observation Transformer with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.common.obs_transformers import ObservationTransformer
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyObsTransformer(ObservationTransformer):
                pass


            # or

            @baseline_registry.register_policy(name="MyTransformer")
            class MyObsTransformer(ObservationTransformer):
                pass

        """
        from rl.ppo.perception_nets import (
            Net,
        )

        return cls._register_impl(
            "perception_net",
            to_register,
            name,
            assert_type=Net,
        )

    @classmethod
    def get_perception_net(cls, name: str):
        r"""Get the Observation Transformer with :p:`name`."""
        return cls._get_impl("perception_net", name)

    @classmethod
    def register_augmentation_method(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        from augmentation.methods.random_base import Augmentation_Base

        return cls._register_impl(
            "augmentation_method",
            to_register,
            name,
            assert_type=Augmentation_Base,
        )
    
    @classmethod
    def get_augmentation_method(cls, name:str):
                
        return cls._get_impl("augmentation_method", name)
    
baseline_registry = BaselineRegistry()
        
        