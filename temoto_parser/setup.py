from setuptools import setup

package_name = 'temoto_parser'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'src/srl_test',
        'src/TeMotoUMRF',
        ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='slwanna',
    author_email="slwanna@utexas.edu",
    maintainer='Selma',
    maintainer_email="slwanna@utexas.edu",
    keywords=['ROS','ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='TODO: Package description.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'srl_test = src.srl_test:main',
        ],
    },
)
