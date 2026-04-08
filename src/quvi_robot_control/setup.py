from setuptools import find_packages, setup

package_name = 'quvi_robot_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QUVI Team',
    maintainer_email='team@quvi.local',
    description='QUVI 로봇팔 + 레일 + 턴테이블 제어 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_control_node = quvi_robot_control.robot_control_node:main',
        ],
    },
)
