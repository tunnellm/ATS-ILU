# External Dependencies and Matrix Data

MC64 source and binaries are not distributed. Obtain a compatible implementation
from the [HSL MC64 catalogue page](https://www.hsl.rl.ac.uk/catalogue/hsl_mc64.html)
under its applicable terms; this repository grants no MC64 license.
`c_ilu/include/superlu_ddefs.h` is an interface shim, not an implementation.

SuiteSparse manifests identify matrices contributed to the SuiteSparse Matrix
Collection. Matrix values are downloaded separately; retain upstream notices.
Cite T. A. Davis and Y. Hu, The University of Florida Sparse Matrix Collection,
ACM Transactions on Mathematical Software 38(1), 2011, Article 1.

Julia and Python libraries are external dependencies with their own licenses;
they are not relicensed by the repository's BSD license. Versions are recorded
in `Project.toml`, `Manifest.toml`, and `requirements.txt`.
