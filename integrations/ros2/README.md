# ROS 2 integration (`qlcog_ros`)

A ROS 2 (`ament_python`) package with one node, `human_model_node`, that serves qlcog's
`HumanModelService`: it collects people's answers, refits the human-model ensemble, and answers
queries with a predicted answer distribution, its uncertainty and an ask-or-act decision.

| Topic | Type | Content |
|---|---|---|
| `/human_model/answers` (subscribed) | `std_msgs/String` | JSON `{"order": "AB", "answers": [1, 0]}` (1 = yes, in the order asked) |
| `/human_model/query` (subscribed) | `std_msgs/String` | JSON `{"order": "AB", "ask_cost": 1, "error_cost": 3}` (fields optional) |
| `/human_model/decision` (published) | `std_msgs/String` | JSON reply: `prediction`, `p_first_yes`, `uncertainty`, `weights`, `action`, `reason` |

Only standard messages are used, so no interface package has to be built.

```bash
# in a ROS 2 workspace (Humble or later), with qlcog installed in the same Python environment
cp -r integrations/ros2/qlcog_ros ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select qlcog_ros && source install/setup.bash
ros2 launch qlcog_ros human_model.launch.py
ros2 topic pub --once /human_model/answers std_msgs/String "{data: '{\"order\": \"AB\", \"answers\": [1, 0]}'}"
ros2 topic pub --once /human_model/query std_msgs/String "{data: '{\"order\": \"AB\"}'}"
ros2 topic echo /human_model/decision
```

**Testing status.** The service logic (`qlcog.applications.robotics.HumanModelService`) is tested in
the package's CI, and the node's callbacks are tested there with stand-in `rclpy` and `std_msgs`
modules (`tests/test_ros2_node.py`). The package has not yet been built and run in a ROS 2
installation; please report problems via the issue tracker.
