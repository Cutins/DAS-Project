from setuptools import setup
from glob import glob

package_name = 'formation_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('resource/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giulia',
    maintainer_email='giulia.cutini.gc@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'the_agent = formation_control.the_agent:main',
            'visualizer = formation_control.visualizer:main',
            'the_plotter = formation_control.the_plotter:main',
            'the_obstacle = formation_control.the_obstacle:main',
            'visualizer_for_obstacle = formation_control.visualizer_for_obstacle:main',
        ],
    },
)
