from setuptools import setup

setup(
    name='qlcog_ros',
    version='1.3.0',
    packages=['qlcog_ros'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/qlcog_ros']),
                ('share/qlcog_ros', ['package.xml']),
                ('share/qlcog_ros/launch', ['launch/human_model.launch.py'])],
    install_requires=['setuptools', 'qlcog'],
    zip_safe=True,
    maintainer='Yeshwanth Guru',
    maintainer_email='yeshwanth445@gmail.com',
    description="ROS 2 node serving qlcog's human-model ensemble",
    license='Apache-2.0',
    entry_points={'console_scripts': ['human_model_node = qlcog_ros.human_model_node:main']},
)
