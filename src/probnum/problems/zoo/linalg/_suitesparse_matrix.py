"""Sparse matrices from the SuiteSparse Matrix Collection.

This implementation is based on Sudarshan Raghunathan's SSGETPY package
(https://github.com/drdarshan/ssgetpy).
"""

import logging
import os
from typing import Optional, Union

import scipy.io
import scipy.sparse

from ._suitesparse._database import suitesparse_db_instance

logger = logging.getLogger(__name__)


def suitesparse_matrix(
    name_or_id: Optional[Union[int, str]] = None,
    matrixformat: str = "MM",
    location: str = None,
    download: bool = False,
    **kwargs,
) -> scipy.sparse.spmatrix:
    """Sparse matrix from the SuiteSparse Matrix Collection.

    Download a sparse matrix benchmark from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    name_or_id :
        Name or ID of the SuiteSparse matrix.
    matrixformat :
    location :
    download :

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
    >>> sparse_mat = suitesparse_matrix(matid=1438, download=True)
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
        if download:
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
                    mat = scipy.io.loadmat(file_name=destpath)
                elif matrixformat == "RB":
                    mat = scipy.io.hb_read(
                        path_or_open_file=os.path.join(destpath, matrix.name + ".rb")
                    )
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
