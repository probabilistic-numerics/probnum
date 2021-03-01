"""Sparse matrices from the SuiteSparse Matrix Collection."""

import codecs
import csv
import io
import pathlib
import tarfile
import tempfile
from typing import Dict, Optional

import numpy as np
import requests
import scipy.io
import tqdm.auto

import probnum.linops as linops
from probnum.type import DTypeArgType

# URLs and file paths to data
SUITESPARSE_ROOT_URL = "https://sparse.tamu.edu"
SUITESPARSE_INDEX_URL = SUITESPARSE_ROOT_URL + "/files/ssstats.csv"


def suitesparse_matrix(
    name: str,
    group: str,
    download: bool = False,
    path: str = pathlib.Path.cwd(),
) -> "SuiteSparseMatrix":
    """Sparse matrix from the SuiteSparse Matrix Collection.

    Download a sparse matrix benchmark from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    name :
        Name of the matrix.
    group :
        Group of the matrix.
    download :
        Whether to download the matrix. If ``False`` only information about
        the matrix is returned.
    path :
        Filepath to save the downloaded matrix to. Defaults to the current working
        directory.

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1
           (2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.

    Examples
    --------
    >>> ssmat = suitesparse_matrix(name="ash85", group="HB", download=True)
    >>> ssmat
    <85x85 SuiteSparseMatrix with dtype=float64>
    >>> ssmat.trace()
    85.0
    """
    # Get database index
    response = requests.get(SUITESPARSE_INDEX_URL, "r")
    line_gen = response.iter_lines()
    for i in range(2):
        next(line_gen)  # skip lines not part of the matrix table

    # Read index with custom header
    fieldnames = [
        "group",
        "name",
        "nrows",
        "ncols",
        "nnz",
        "real",
        "logical",
        "is2d3d",
        "isspd",
        "psym",
        "nsym",
        "kind",
    ]
    databaseindex_reader = csv.DictReader(
        codecs.iterdecode(line_gen, "utf-8"), fieldnames=fieldnames
    )

    # Query the SuiteSparse Matrix collection
    matrix_attr_dict = None
    matid = 0
    for row in databaseindex_reader:
        matid += 1
        if row["group"] == group and row["name"] == name:
            matrix_attr_dict = row
            matrix_attr_dict["matid"] = f"{matid}"

    if matrix_attr_dict is None:
        raise ValueError(
            f"Could not find matrix {name} in the SuiteSparse database index."
        )

    # Create a SuiteSparseMatrix (and save to file)
    matrix = SuiteSparseMatrix.from_database_entry(matrix_attr_dict)

    if download:
        matrix.download()

    return matrix


class SuiteSparseMatrix(linops.LinearOperator):
    """SuiteSparse Matrix.

    Sparse matrix from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    matid :
        Unique identifier for the matrix in the database.
    group :
        Group this matrix belongs to.
    name :
        Name of this matrix.
    nrows :
        Number of rows.
    ncols :
        Number of columns.
    nnz  :
        Number of non-zero elements.
    dtype:
        Datatype of non-zero elements.
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

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1 (
           2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.
    """

    # pylint: disable="too-many-instance-attributes"
    def __init__(
        self,
        matid: str,
        group: str,
        name: str,
        dtype: DTypeArgType,
        nrows: int,
        ncols: int,
        nnz: int,
        is2d3d: bool,
        isspd: bool,
        psym: float,
        nsym: float,
        kind: str,
    ):
        self.matrix = None
        self.matid = matid
        self.group = group
        self.name = name
        self.nnz = nnz
        self.is2d3d = is2d3d
        self.isspd = isspd
        self.psym = psym
        self.nsym = nsym
        self.kind = kind

        super().__init__(shape=(nrows, ncols), dtype=np.dtype(dtype))

    @classmethod
    def from_database_entry(cls, database_entry: Dict):
        """Create a SuiteSparseMatrix object from an entry of the database index.

        Parameters
        ----------
        database_entry :
            Dictionary representing one entry from the SuiteSparse database index.
        """
        if database_entry["logical"] == "1":
            dtype = np.dtype(np.float_)
        elif database_entry["real"] == "1":
            dtype = np.dtype(np.bool_)
        else:
            dtype = np.dtype(np.complex_)

        return cls(
            matid=database_entry["matid"],
            group=database_entry["group"],
            dtype=dtype,
            nrows=int(database_entry["nrows"]),
            ncols=int(database_entry["ncols"]),
            nnz=int(database_entry["nnz"]),
            is2d3d=bool(database_entry["is2d3d"]),
            isspd=bool(database_entry["isspd"]),
            psym=float(database_entry["psym"]),
            nsym=float(database_entry["nsym"]),
            name=database_entry["name"],
            kind=database_entry["kind"],
        )

    def _matvec(self, x):
        if self.matrix is None:
            raise RuntimeError(
                "Matrix has not been downloaded yet. Call self.download() first."
            )
        else:
            return self.matrix @ x

    def download(self, verbose: bool = False):
        """Download and extract file archive containing the sparse matrix.

        verbose:
            Print additional information.
        """
        url = SUITESPARSE_ROOT_URL + f"/MM/{self.group}/{self.name}.tar.gz"
        response = requests.get(url, stream=True)

        # Write archive to temporary file
        if verbose:
            print("Downloading matrix")

        buffer = io.BytesIO()
        with tqdm.auto.tqdm(
            total=int(response.headers["content-length"]), desc=self.name, unit="B"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=4096):
                buffer.write(chunk)
                pbar.update(4096)

        buffer.seek(0)
        if verbose:
            print("Extracting file archive.")
        tar = tarfile.open(fileobj=buffer, mode="r:gz")
        self.matrix = scipy.io.mmread(tar.extractfile(tar.getmembers()[0]))

    def serialize(self, filename: Optional[str] = None):
        """Save the matrix to file.

        Parameters
        ----------
        filename :
            Filename of the matrix.
        """
        if filename is None:
            filename = pathlib.Path.cwd() / self.name
        raise NotImplementedError

    @classmethod
    def deserialize(self, filename: str):
        """Load matrix from file.

        Parameters
        ----------
        filename :
            Filename of the matrix.
        """
        raise NotImplementedError

    # TODO: tqdm bar in kb or mb
    # TODO: serialization and deserialization
    # TODO: HTML representation for notebooks
    # TODO: efficient methods of linear operators making use of self.matrix
    # TODO: tests


# class SuiteSparseMatrixList(list):
#     """List of SuiteSparseMatrix objects."""
#
#     def _repr_html_(self):
#         """HTML representation."""
#         body = "".join(r.to_html_row() for r in self)
#         return f"<table>{SuiteSparseMatrix.html_header()}<tbody>{body}</tbody></table>"
#
#     def __getitem__(self, expr):
#         result = super().__getitem__(expr)
#         return SuiteSparseMatrixList(result) if isinstance(expr, slice) else result
#
#     def download(self):
#         raise NotImplementedError


# def suitesparse_matrix2(
#     name: Optional[str] = None,
#     matid: Optional[int] = None,
#     group: Optional[str] = None,
#     rows: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
#     cols: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
#     nnz: Optional[Union[Tuple[Union[int, None], Union[int, None]], int]] = None,
#     dtype: Optional[str] = None,
#     isspd: Optional[bool] = None,
#     psym: Optional[Union[Tuple[Union[float, None], Union[float, None]], float]] = None,
#     nsym: Optional[Union[Tuple[Union[float, Union[float, None]], float], float]] = None,
#     is2d3d: Optional[bool] = None,
#     kind: Optional[str] = None,
#     query_only: bool = True,
#     max_results: int = 10,
#     location: str = None,
# ) -> Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]]:
#     """Sparse matrix from the SuiteSparse Matrix Collection.
#
#     Download a sparse matrix benchmark from the `SuiteSparse Matrix Collection
#     <https://sparse.tamu.edu/>`_. [1]_ [2]_
#
#     Any of the matrix properties used to query support partial matches.
#
#     Parameters
#     ----------
#     matid :
#         Unique identifier for the matrix in the database.
#     group :
#         Group the matrix belongs to.
#     name :
#         Name of the matrix.
#     rows :
#         Number of rows or tuple :code:`(min, max)` defining limits.
#     cols :
#         Number of columns or tuple :code:`(min, max)` defining limits.
#     nnz  :
#         Number of non-zero elements or tuple :code:`(min, max)` defining limits.
#     dtype:
#         Datatype of non-zero elements: `real`, `complex` or `binary`.
#     is2d3d:
#         Does this matrix come from a 2D or 3D discretization?
#     isspd :
#         Is this matrix symmetric, positive definite?
#     psym :
#         Degree of symmetry of the matrix pattern or tuple :code:`(min, max)` defining
#         limits.
#     nsym :
#         Degree of numerical symmetry of the matrix or tuple :code:`(min, max)` defining
#         limits.
#     kind  :
#         Problem domain this matrix arises from.
#     query_only :
#         In :code:`query_only` mode information about the sparse matrices is returned
#         without download.
#     max_results :
#         Maximum number of results to return from the database.
#     location :
#         Location to download the matrices too.
#
#     References
#     ----------
#     .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
#            collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1
#            (2011): 1-25.
#     .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
#            interface. *Journal of Open Source Software* 4.35 (2019): 1244.
#
#     Examples
#     --------
#     Query the SuiteSparse Matrix Collection.
#
#     >>> from probnum.problems.zoo.linalg import suitesparse_matrix
#     >>> suitesparse_matrix(group="Oberwolfach", rows=(10, 20))
#     [_SuiteSparseMatrix(matid=1438, group='Oberwolfach', name='LF10', rows=18, cols=18, nnz=82, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem'), _SuiteSparseMatrix(matid=1440, group='Oberwolfach', name='LFAT5', rows=14, cols=14, nnz=46, dtype='real', is2d3d=1, isspd=1, psym=1.0, nsym=1.0, kind='model reduction problem')]
#
#     Download a sparse matrix and check its sparsity level.
#
#     >>> import numpy as np
#     >>> sparse_mat = suitesparse_matrix(matid=1438, query_only=False)
#     >>> np.mean(sparse_mat > 0)
#     0.16049382716049382
#     """
#     # Query the SuiteSparse Matrix collection
#     matrices = suitesparse_db_instance.search(
#         matid=matid,
#         group=group,
#         name=name,
#         rows=rows,
#         cols=cols,
#         nnz=nnz,
#         dtype=dtype,
#         is2d3d=is2d3d,
#         isspd=isspd,
#         psym=psym,
#         nsym=nsym,
#         kind=kind,
#         limit=max_results,
#     )
#
#     # Download Matrices
#     if not query_only:
#         matrixformat = "MM"
#         spmatrices = []
#
#         for matrix in matrices:
#             matrix.download(matrixformat, location, extract=True)
#
#             # Read from file
#             destpath = matrix.localpath(matrixformat, location, extract=True)[0]
#             mat = scipy.io.mmread(source=os.path.join(destpath, matrix.name + ".mtx"))
#             spmatrices.append(mat)
#
#         return spmatrices[0] if len(spmatrices) == 1 else spmatrices
#
#     return matrices
