"""ROS 2 node for trust-aware hand-over.

Topics (``std_msgs/String`` carrying JSON):

* subscribes ``/handover/events``: ``{"outcome": "taken"}`` or ``{"outcome": "dropped"}`` after a
  hand-over, ``{"answer": "yes"}`` or ``{"answer": "no"}`` after the robot asked, ``{"event": "waited"}``
  after a wait, and ``{"event": "reset"}`` for a new person;
* publishes ``/handover/decision``: ``{"action": ..., "p_trust": ..., "costs": {...}}`` once at start
  and after every event.

The robot's motion controller subscribes to the decisions and reports what happened. The logic lives
in :class:`HandoverController`, which needs no ROS and is tested on its own; :func:`main` wraps it in
an ``rclpy`` node (ROS 2 Humble or later). Run it with
``ros2 run``-style: ``python -m quantum_handover.ros2_node --policy quantum-like``.
"""
from __future__ import annotations

import argparse
import json

from .policies import make_policy

__all__ = ['HandoverController', 'main']


class HandoverController:
    """Turns hand-over events into the next decision.

    Parameters
    ----------
    policy : str, optional
        A name from :data:`quantum_handover.policies.POLICIES`.
    **costs
        Cost settings passed to :func:`quantum_handover.policies.make_policy`.
    """

    def __init__(self, policy='quantum-like', **costs):
        self.name, self.costs = policy, costs
        self.policy = make_policy(policy, **costs)

    def decision(self):
        """The current decision as a JSON string."""
        d = self.policy.decide()
        return json.dumps({'action': d['action'], 'p_trust': round(float(d['p_trust']), 4),
                           'costs': {k: round(float(v), 4) for k, v in d.get('costs', {}).items()}})

    def handle(self, message):
        """Apply one JSON event and return the next decision (JSON).

        Parameters
        ----------
        message : str
            Event as described in the module docstring.

        Returns
        -------
        str

        Raises
        ------
        ValueError
            For an event the controller does not know.
        """
        ev = json.loads(message)
        if ev.get('outcome') in ('taken', 'dropped'):
            self.policy.observe(int(ev['outcome'] == 'taken'))
        elif ev.get('answer') in ('yes', 'no'):
            self.policy.answer(ev['answer'] == 'yes')
        elif ev.get('event') == 'waited':
            self.policy.wait()
        elif ev.get('event') == 'reset':
            self.policy = make_policy(self.name, **self.costs)
        else:
            raise ValueError('unknown hand-over event: %s' % message)
        return self.decision()


def main(argv=None):
    """Run the ROS 2 node (needs rclpy and std_msgs from a ROS 2 installation)."""
    ap = argparse.ArgumentParser(description='Trust-aware hand-over ROS 2 node')
    ap.add_argument('--policy', default='quantum-like')
    args, ros_args = ap.parse_known_args(argv)
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class HandoverNode(Node):
        def __init__(self):
            super().__init__('trust_handover')
            self.controller = HandoverController(args.policy)
            self.pub = self.create_publisher(String, '/handover/decision', 10)
            self.create_subscription(String, '/handover/events', self.on_event, 10)
            self.publish(self.controller.decision())

        def publish(self, text):
            msg = String()
            msg.data = text
            self.pub.publish(msg)

        def on_event(self, msg):
            try:
                self.publish(self.controller.handle(msg.data))
            except (ValueError, json.JSONDecodeError) as exc:
                self.get_logger().warning(str(exc))

    rclpy.init(args=ros_args)
    node = HandoverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
