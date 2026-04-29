from setuptools import find_packages, setup

package_name = 'manipulation_servers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/servers.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='Pick and place action servers',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'objects_finder_server = manipulation_servers.objects_finder_server:main',
            'pick_server = manipulation_servers.pick_server:main',
            'place_server = manipulation_servers.place_server:main',
        ],
    },
)
