from setuptools import setup

setup(
    name='quantum_mind_ros',
    version='1.3.0',
    packages=['quantum_mind_ros'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/quantum_mind_ros']),
                ('share/quantum_mind_ros', ['package.xml']),
                ('share/quantum_mind_ros/launch', ['launch/human_model.launch.py'])],
    install_requires=['setuptools', 'quantum_mind'],
    zip_safe=True,
    maintainer='Yeshwanth Guru',
    maintainer_email='yeshwanth445@gmail.com',
    description="ROS 2 node serving quantum_mind's human-model ensemble",
    license='Apache-2.0',
    entry_points={'console_scripts': ['human_model_node = quantum_mind_ros.human_model_node:main']},
)
