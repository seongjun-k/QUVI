from setuptools import find_packages, setup

package_name = 'quvi_inspect'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/inspect_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QUVI Team',
    maintainer_email='rkdsungjun3344@gmail.com',
    description='QUVI 양불 판정 노드',
    license='MIT',
    entry_points={
        'console_scripts': [
            'inspect_node = quvi_inspect.inspect_node:main',
            'stl_renderer = quvi_inspect.stl_renderer:main',
        ],
    },
)
