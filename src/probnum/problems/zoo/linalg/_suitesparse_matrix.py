"""Sparse matrices from the SuiteSparse Matrix Collection."""

import os
from typing import List, Optional, Tuple, Union

import scipy.io
import scipy.sparse

from ._suitesparse._database import suitesparse_db_instance


def suitesparse_matrix(
    name: Optional[str] = None,
    matid: Optional[int] = None,
    group: Optional[str] = None,
    rows: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
    cols: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
    nnz: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
    dtype: Optional[str] = None,
    isspd: Optional[bool] = None,
    psym: Optional[Union[Tuple[Union[float, None], Union[float, None]], float]] = None,
    nsym: Optional[Union[Tuple[Union[float, Union[float, None]], float], float]] = None,
    is2d3d: Optional[bool] = None,
    kind: Optional[str] = None,
    query_only: bool = True,
    max_results: int = 10,
    location: str = None,
) -> Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]]:
    """Sparse matrix from the SuiteSparse Matrix Collection.

    Download a sparse matrix benchmark from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Any of the matrix properties used to query support partial matches.

    Parameters
    ----------
    matid :
        Unique identifier for the matrix in the database.
    group :
        Group the matrix belongs to.
    name :
        Name of the matrix.
    rows :
        Number of rows or tuple :code:`(min, max)` defining limits.
    cols :
        Number of columns or tuple :code:`(min, max)` defining limits.
    nnz  :
        Number of non-zero elements or tuple :code:`(min, max)` defining limits.
    dtype:
        Datatype of non-zero elements: `real`, `complex` or `binary`.
    is2d3d:
        Does this matrix come from a 2D or 3D discretization?
    isspd :
        Is this matrix symmetric, positive definite?
    psym :
        Degree of symmetry of the matrix pattern or tuple :code:`(min, max)` defining
        limits.
    nsym :
        Degree of numerical symmetry of the matrix or tuple :code:`(min, max)` defining
        limits.
    kind  :
        Problem domain this matrix arises from.
    query_only :
        In :code:`query_only` mode information about the sparse matrices is returned
        without download.
    max_results :
        Maximum number of results to return from the database.
    location :
        Location to download the matrices too.

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1
           (2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.

    Examples
    --------
    Query the SuiteSparse Matrix Collection.

    >>> from probnum.problems.zoo.linalg import suitesparse_matrix
    >>> suitesparse_matrix(group="Oberwolfach", rows=(10, 20))
    [_SuiteSparseMatrix(matid=1438, group='Oberwolfach', name='LF10', rows=18, cols=18, nnz=82, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem'), _SuiteSparseMatrix(matid=1440, group='Oberwolfach', name='LFAT5', rows=14, cols=14, nnz=46, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem')]

    Download a sparse matrix and check its sparsity level.

    >>> import numpy as np
    >>> sparse_mat = suitesparse_matrix(matid=1438, query_only=False)
    >>> np.mean(sparse_mat > 0)
    0.16049382716049382
    """
    # Query the SuiteSparse Matrix collection
    matrices = suitesparse_db_instance.search(
        matid=matid,
        group=group,
        name=name,
        rows=rows,
        cols=cols,
        nnz=nnz,
        dtype=dtype,
        is2d3d=is2d3d,
        isspd=isspd,
        psym=psym,
        nsym=nsym,
        kind=kind,
        limit=max_results,
    )

    # Download Matrices
    if not query_only:
        matrixformat = "MM"
        spmatrices = []

        for matrix in matrices:
            matrix.download(matrixformat, location, extract=True)

            # Read from file
            destpath = matrix.localpath(matrixformat, location, extract=True)[0]
            mat = scipy.io.mmread(source=os.path.join(destpath, matrix.name + ".mtx"))
            spmatrices.append(mat)

        return spmatrices[0] if len(spmatrices) == 1 else spmatrices

    return matrices
