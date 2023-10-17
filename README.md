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
The numerical examples provided in there are included in this repository
and are part of the Python package uploaded to PyPi. The tutorial will be
kept up to date with the development of the DUNE code. To results from the
paper are based on the 2.9.0 version.
To obtain the scripts to reproduce the results from the paper you can
either install the package from PyPi
```
pip install dune-vem==2.9.0
```
or check out the 2.9.0 tag of this repository
```
git clone -b 2.9.0 https://gitlab.dune-project.org/dune-fem/dune-vem.git
```
and follow the instruction of installing from source given below.
After installation of the Python package you can run
```
python -m dune.vem
```
which places the scripts into the folder ``vem_tutorial``. The tutorial
consists of the following scripts:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-jlhb{background-color:#ffc702;border-color:#000000;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-jlhb">Script</th>
    <th class="tg-jlhb">Section</th>
    <th class="tg-jlhb">Description</th>
    <th class="tg-jlhb">Parameters</th>
    <th class="tg-jlhb">Output</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">laplace.py</td>
    <td class="tg-0lax">7.1</td>
    <td class="tg-0lax">solves Laplace problem in <br>primal and dual form on a Voronoi grid</td>
    <td class="tg-0lax">None</td>
    <td class="tg-0lax">pngs of solution and grids</td>
  </tr>
  <tr>
    <td class="tg-0lax">mixedSolver.py</td>
    <td class="tg-0lax">7.1<br></td>
    <td class="tg-0lax">helper script with solver for mixed problem</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">testStabilization.py</td>
    <td class="tg-0lax">7.2</td>
    <td class="tg-0lax">Compare stabilized and non-stabilized versions<br>for Laplace and Bilaplace problem<br></td>
    <td class="tg-0lax">-L max-level &gt; 0<br>- l polynomial order &gt;= 1<br>-p problem = [laplace|biharmonic]<br>-s use stabilization = [-1|0|1]<br>     1: standard projections and stabilization<br>     0: standard projection no stabilization<br>    -1: higher order projections, no stabilization<br>         this projects the gradient and the hessian into <br>         P_l and P_{l-1}, respectively.</td>
    <td class="tg-0lax">errors and convergence rates</td>
  </tr>
  <tr>
    <td class="tg-0lax">cylinderUzawa.py</td>
    <td class="tg-0lax">7.3<br></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">None</td>
    <td class="tg-0lax">time series vtk files</td>
  </tr>
  <tr>
    <td class="tg-0lax">uzawa.py</td>
    <td class="tg-0lax">7.3<br></td>
    <td class="tg-0lax">helper script with solver for Stokes problem</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">willmore.py</td>
    <td class="tg-0lax">7.4</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax">None</td>
    <td class="tg-0lax">ime series vtk files</td>
  </tr>
</tbody>
</table>

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

A detailed description of DUNE-FEM can be found in

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
