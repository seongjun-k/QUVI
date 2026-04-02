import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'quvi_hmi'

# static 및 template 파일 수집
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config', ['config/hmi_params.yaml']),
]

# templates
for dirpath, dirnames, filenames in os.walk(package_name + '/templates'):
    install_dir = os.path.join('share', package_name, dirpath.replace(package_name + '/', '', 1))
    files = [os.path.join(dirpath, f) for f in filenames]
    if files:
        data_files.append((install_dir, files))

# static
for dirpath, dirnames, filenames in os.walk(package_name + '/static'):
    install_dir = os.path.join('share', package_name, dirpath.replace(package_name + '/', '', 1))
    files = [os.path.join(dirpath, f) for f in filenames]
    if files:
        data_files.append((install_dir, files))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QUVI Team',
    maintainer_email='rkdsungjun3344@gmail.com',
    description='QUVI HMI Web 대시보드',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hmi_node = quvi_hmi.hmi_node:main',
        ],
    },
)
