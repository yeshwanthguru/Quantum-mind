"""ROS 2 node that serves qlcog's human-model ensemble to a robot.

Topics (std_msgs/String carrying JSON, so no custom interfaces are needed):
  subscribes  ~/answers   {"order": "AB", "answers": [1, 0]}          one person's two answers
  subscribes  ~/query     {"order": "AB", "ask_cost": 1, "error_cost": 3}
  publishes   ~/decision  the reply of HumanModelService.query(): prediction, uncertainty,
                          ensemble weights and {"action": "ask" | "act", "reason": ...}
Parameters: refit_every (int, 20), min_answers (int, 20), ask_cost (float, 1.0), error_cost (float, 5.0),
max_disagreement_bits (float, 0.05) - defaults for queries that omit them.

Run:  ros2 run qlcog_ros human_model_node
      ros2 topic pub --once /human_model/answers std_msgs/String "{data: '{\\"order\\": \\"AB\\", \\"answers\\": [1, 0]}'}"
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from qlcog.applications.robotics import HumanModelService


class HumanModelNode(Node):
    def __init__(self):
        super().__init__('human_model')
        for name, default in (('refit_every', 20), ('min_answers', 20), ('ask_cost', 1.0), ('error_cost', 5.0),
                              ('max_disagreement_bits', 0.05)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value                        # noqa: E731
        self.service = HumanModelService(refit_every=int(p('refit_every')), min_answers=int(p('min_answers')))
        self.defaults = {k: float(p(k)) for k in ('ask_cost', 'error_cost', 'max_disagreement_bits')}
        self.pub = self.create_publisher(String, '~/decision', 10)
        self.create_subscription(String, '~/answers', self.on_answer, 10)
        self.create_subscription(String, '~/query', self.on_query, 10)

    def on_answer(self, msg):
        try:
            info = self.service.add_answer(json.loads(msg.data))
            if info['refitted']:
                self.get_logger().info('refitted human models on %d answers' % info['n_answers'])
        except (ValueError, json.JSONDecodeError) as e:
            self.get_logger().warning('ignored answer message: %s' % e)

    def on_query(self, msg):
        try:
            req = {**self.defaults, **(json.loads(msg.data) if msg.data else {})}
            reply = self.service.query(req)
        except (ValueError, json.JSONDecodeError) as e:
            reply = {'action': 'ask', 'reason': 'bad query: %s' % e}
        out = String(); out.data = json.dumps(reply); self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HumanModelNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
