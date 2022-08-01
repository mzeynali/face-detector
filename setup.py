from setuptools import setup

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]
    
setup(
    name='face_detector',
    version='0.1.0',
    author='Mohammad.Zeynali',
    author_email='mzeynali01@gmail.com',
    packages=['face_detector'],
    description='face detector using mediapipe which extract keypoints, eyes ratio (closeness) and ',
    install_requires=requirements,
    python_requires='>=3.8.0',
)
