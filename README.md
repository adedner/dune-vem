DUNE-VEM
========

[[_TOC_]]

[DUNE-VEM][20] is a [Distributed and Unified Numerics Environment][1]
module which provides implementation of a range of virtual element
spaces. It is based on the interfaces defined in [DUNE-FEM][0].
In addition to the C++ implementation and extensive Python interface
is provided.

If you need help, please ask on our [mailinglist][5]. Bugs can also be submitted
to the DUNE-VEM [bugtracker][6] instead.

Paper
-----

A detailed description of the VEM implementation is given in our paper
[A framework for implementing general virtual element spaces][21].
Please cite this paper, if you find this module useful for your research.

The scripts to reproduce the numerical examples provided in the paper are
provided in the [dune-vem-paper][22] repository. These were based on the
2.9.0 version of DUNE. Versions of these scripts updated to match the current
versions together with some new ones are included in this repository
and are part of the Python package uploaded to PyPi.
To obtain the current versions of these scripts
after installation of the DUNE-VEM Python package you can run
```
python -m dune.vem
```
which places the scripts into the folder ``vem_tutorial``.

Tutorial
--------

The more general [DUNE-FEM tutorial][18] includes a number of further examples showcasing the DUNE-VEM module
and provides an overview of DUNE.

General installation instructions
---------------------------------

**Using pip**

DUNE-VEM can be installed using the Package Index of Python (pip).

```
pip install dune-vem
```

See https://dune-project.org/doc/installation-pip/ for a more detailed
description.

**From source**

For a full explanation of the DUNE installation process please read
the [installation notes][2].

When using the main branch observe the [build status][19]
to make sure you get a working version.

Dependencies
------------

DUNE-VEM requires a recent C++ compiler (e.g. g++ or clang),
cmake, pkg-config (see DUNE [installation][2] for details)
and depends on the following DUNE modules:

* [dune-common][10]

* [dune-geometry][11]

* [dune-grid][12]

* [dune-fem][12]

The following DUNE modules are suggested:

* [dune-istl][13]

* [dune-localfunctions][14]

* [dune-alugrid][8]

* [dune-spgrid][9]

The following software is optional:

* [PETSc][3]

* [SIONlib][16]

* [SuiteSparse][15]

License
-------

The DUNE-VEM library, headers and test programs are free open-source software,
licensed under version 2 or later of the GNU General Public License.

See the file [LICENSE][7] for full copying permissions.


References
----------

A detailed description of DUNE-VEM and the uderlying DUNE-FEM module can be found in

* A. Dedner, A. Hodson. A framework for implementing general virtual * element space.
  https://arxiv.org/abs/2208.08978

* A. Dedner, R. Klöfkorn, M. Nolte, and M. Ohlberger. A Generic Interface for Parallel and Adaptive Scientific Computing:
  Abstraction Principles and the DUNE-FEM Module.
  Computing, 90(3-4):165--196, 2010. http://dx.doi.org/10.1007/s00607-010-0110-3

* A. Dedner, R. Klöfkorn, and M. Nolte. Python Bindings for the DUNE-FEM module.
  Zenodoo, 2020 http://dx.doi.org/10.5281/zenodo.3706994


 [0]: https://www.dune-project.org/modules/dune-fem/
 [1]: https://www.dune-project.org
 [2]: https://www.dune-project.org/doc/installation/
 [3]: http://www.mcs.anl.gov/petsc/
 [4]: http://eigen.tuxfamily.org
 [5]: http://lists.dune-project.org/mailman/listinfo/dune-fem
 [6]: http://gitlab.dune-project.org/dune-fem/dune-fem/issues
 [7]: LICENSE.md
 [8]: http://gitlab.dune-project.org/extensions/dune-alugrid
 [9]: http://gitlab.dune-project.org/extensions/dune-spgrid
 [10]: http://gitlab.dune-project.org/core/dune-common
 [11]: http://gitlab.dune-project.org/core/dune-geometry
 [12]: http://gitlab.dune-project.org/core/dune-grid
 [13]: http://gitlab.dune-project.org/core/dune-istl
 [14]: http://gitlab.dune-project.org/core/dune-localfunctions
 [15]: http://faculty.cse.tamu.edu/davis/suitesparse.html
 [16]: http://www.fz-juelich.de/jsc/sionlib
 [17]: http://icl.cs.utk.edu/papi/software/index.html
 [18]: https://dune-project.org/sphinx/content/sphinx/dune-fem/
 [19]: https://gitlab.dune-project.org/dune-fem/dune-fem/-/pipelines/
 [20]: https://www.dune-project.org/modules/dune-vem/
 [21]: https://arxiv.org/abs/2208.08978
 [22]: https://gitlab.dune-project.org/dune-fem/dune-vem-paper
