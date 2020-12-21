#   Temporal Basis Transformation Network
#   Copyright (C) 2020  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from distutils.core import setup

setup(
    name='temporal_basis_transformation_network',
    version='1.0',
    author='Andreas Stöckel',
    author_email='astoecke@uwaterloo.ca',
    description=
    'TensorFlow/Keras network layer for temporal basis function transformations',
    packages=['temporal_basis_transformation_network'],
    license=
    'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=list(
        map(lambda s: s.strip(),
            open('requirements.txt').readlines())),
)

