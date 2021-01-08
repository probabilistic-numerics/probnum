"""Sparse matrices from the SuiteSparse Matrix Collection."""

import logging
import os
from typing import Optional, Tuple, Union

import scipy.io
import scipy.sparse

from ._suitesparse._database import suitesparse_db_instance

logger = logging.getLogger(__name__)


def suitesparse_matrix(
    name: Optional[str] = None,
    matid: Optional[int] = None,
    group: Optional[str] = None,
    rows: Optional[Union[Tuple[int], int]] = None,
    cols: Optional[Union[Tuple[int], int]] = None,
    nnz: Optional[Union[Tuple[int], int]] = None,
    dtype: Optional[str] = None,
    isspd: Optional[bool] = None,
    psym: Optional[Union[Tuple[float], float]] = None,
    nsym: Optional[Union[Tuple[float], float]] = None,
    is2d3d: Optional[bool] = None,
    kind: Optional[str] = None,
    query_only: bool = True,
    matrixformat: str = "MM",
    location: str = None,
) -> scipy.sparse.spmatrix:
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
        Number of rows.
    cols :
        Number of columns.
    nnz  :
        Number of non-zero elements.
    dtype:
        Datatype of non-zero elements: `real`, `complex` or `binary`.
    is2d3d:
        Does this matrix come from a 2D or 3D discretization?
    isspd :
        Is this matrix symmetric, positive definite?
    psym :
        Degree of symmetry of the matrix pattern.
    nsym :
        Degree of numerical symmetry of the matrix.
    kind  :
        Information of the problem domain this matrix arises from.
    query_only :
        In :code:`query_only` mode information about the sparse matrices is returned
        without download.
    matrixformat :
        Format to download the matrices in. One of ("MM", "MAT").
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

    >>> suitesparse_matrix(group="Oberwolfach", rowbounds=(10, 20))
    [_SuiteSparseMatrix(matid=1438, group='Oberwolfach', name='LF10', rows=18, cols=18, nonzeros=82, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem'), _SuiteSparseMatrix(matid=1440, group='Oberwolfach', name='LFAT5', rows=14, cols=14, nonzeros=46, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem')]

    Download a sparse matrix and create a linear system from it.

    >>> import numpy as np
    >>> sparse_mat = suitesparse_matrix(matid=1438, query_only=False)
    >>> np.mean(sparse_mat > 0)
    0.16049382716049382
    """
    # Query SuiteSparse Matrix collection
    matrices = _suitesparse_query(name_or_id, **kwargs)

    # Download Matrices
    spmatrices = []
    if len(matrices) > 0:
        logger.info(
            "Found %d %s", len(matrices), "entry" if len(matrices) == 1 else "entries"
        )
        if not query_only:
            for matrix in matrices:
                logger.info(
                    "Downloading %s/%s to %s",
                    matrix.group,
                    matrix.name,
                    matrix.localpath(matrixformat, location, extract=True)[0],
                )
                matrix.download(matrixformat, location, extract=True)

                # Read from file
                destpath = matrix.localpath(matrixformat, location, extract=True)[0]
                if matrixformat == "MM":
                    mat = scipy.io.mmread(
                        source=os.path.join(destpath, matrix.name + ".mtx")
                    )
                elif matrixformat == "MAT":
                    mat = scipy.io.loadmat(file_name=destpath)["Problem"][0][0][2]
                # elif matrixformat == "RB":
                #     mat = scipy.io.hb_read(
                #         path_or_open_file=os.path.join(destpath, matrix.name + ".rb")
                #     )
                else:
                    raise ValueError("Format must be 'MM', 'MAT' or 'RB'")
                spmatrices.append(mat)
            if len(spmatrices) == 1:
                return spmatrices[0]
            else:
                return spmatrices

        # TODO ensure that information passed in _SuiteSparseMatrix is not lost,
        #  maybe by wrapping the sparse matrix into a special LinearOperator which has
        #  properties symmetric, positive_definite, etc.?

    return matrices


def _suitesparse_query(name_or_id: Optional[str] = None, **kwargs):
    """Search for matrices with a given name pattern or numeric ID.

    Optionally, limit search to matrices of a specific data type or with
    the specified range of rows, columns and non-zero values.
    """
    if name_or_id is not None:
        if isinstance(name_or_id, str):
            if "/" in name_or_id:
                group, name = name_or_id.split("/")
                kwargs["group"] = group
                if not name == "" and not name == "*":
                    kwargs["name"] = name
            else:
                kwargs["name"] = name_or_id
        elif isinstance(name_or_id, int):
            kwargs["matid"] = name_or_id
        else:
            raise ValueError("First argument to search must be a string or an integer")

    return suitesparse_db_instance.search(**kwargs)
