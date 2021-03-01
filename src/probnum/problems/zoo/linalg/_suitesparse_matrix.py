"""Sparse matrices from the SuiteSparse Matrix Collection."""

import codecs
import csv
import io
import pathlib
import tarfile
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
    <85x85 SuiteSparseMatrix with dtype=bool>
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
            f"Could not find matrix '{name}' in the SuiteSparse database index."
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
    def from_database_entry(cls, database_entry: Dict) -> "SuiteSparseMatrix":
        """Create a SuiteSparseMatrix object from an entry of the database index.

        Parameters
        ----------
        database_entry :
            Dictionary representing one entry from the SuiteSparse database index.
        """
        if bool(int(database_entry["logical"])):
            dtype = np.dtype(np.bool_)
        elif bool(int(database_entry["real"])):
            dtype = np.dtype(np.float_)
        else:
            dtype = np.dtype(np.complex_)

        return cls(
            matid=database_entry["matid"],
            group=database_entry["group"],
            dtype=dtype,
            nrows=int(database_entry["nrows"]),
            ncols=int(database_entry["ncols"]),
            nnz=int(database_entry["nnz"]),
            is2d3d=bool(int(database_entry["is2d3d"])),
            isspd=bool(int(database_entry["isspd"])),
            psym=float(database_entry["psym"]),
            nsym=float(database_entry["nsym"]),
            name=database_entry["name"],
            kind=database_entry["kind"],
        )

    def _check_matrix_downloaded(self):
        if self.matrix is None:
            raise RuntimeError(
                "Matrix has not been downloaded yet. Call self.download() first."
            )

    def _matvec(self, x):
        self._check_matrix_downloaded()
        return self.matrix @ x

    def _matmat(self, x):
        self._check_matrix_downloaded()
        return self.matrix @ x

    def download(self, verbose: bool = False) -> None:
        """Download and extract file archive containing the sparse matrix.

        verbose:
            Print additional information.
        """
        url = SUITESPARSE_ROOT_URL + f"/MM/{self.group}/{self.name}.tar.gz"
        response = requests.get(url, stream=True)

        # Write archive to temporary file
        if verbose:
            print("Downloading compressed matrix.")

        buffer = io.BytesIO()
        with tqdm.auto.tqdm(
            total=int(response.headers["content-length"]),
            desc=self.name,
            unit="B",
            unit_scale=True,
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
            Filename of the matrix. If none given, the matrix will be saved to the
            current working directory.
        """
        if filename is None:
            filename = pathlib.Path.cwd() / self.name
        raise NotImplementedError

    @classmethod
    def deserialize(self, filename: str) -> "SuiteSparseMatrix":
        """Load matrix from file.

        Parameters
        ----------
        filename :
            Filename of the matrix.
        """
        raise NotImplementedError

    # TODO: serialization and deserialization
    # TODO: HTML representation for notebooks
    # TODO: tests

    def todense(self):
        self._check_matrix_downloaded()
        return np.array(self.matrix.todense())

    def rank(self):
        self._check_matrix_downloaded()
        return np.linalg.matrix_rank(self.A)

    def trace(self):
        self._check_matrix_downloaded()
        if self.shape[0] != self.shape[1]:
            raise ValueError("The trace is only defined for square linear operators.")
        else:
            return self.matrix.diagonal().sum()

    # def _render_item_html(self, key, value):
    #     if key == "Group":
    #         return (
    #             f'<a href="{self._attribute_urls["group_info"]}" '
    #             f'target="_blank">{value}</a>'
    #         )
    #     if key == "Name":
    #         return (
    #             f'<a href="{self._attribute_urls["matrix_info"]}" '
    #             f'target="_blank">{value}</a>'
    #         )
    #     if key in ("Pattern Symmetry", "Numerical Symmetry"):
    #         return f"{value:0.2}"
    #     if key in ("2D/3D Discretization", "SPD"):
    #         return "Yes" if value else "No"
    #     if key == "Preview":
    #         return f'<img src="{value}">'
    #
    #     return str(value)
    #
    # def _to_html_row(self):
    #     return (
    #         "<tr>"
    #         + "".join(
    #             f"<td>{self._render_item_html(key, value)}</td>"
    #             for key, value in zip(
    #                 _SuiteSparseMatrix._attribute_names_list,
    #                 list(zip(*self.__dict__.items()))[
    #                     1
    #                 ],  # Attribute values of instance
    #             )
    #         )
    #         + "</tr>"
    #     )
    #
    # @staticmethod
    # def html_header():
    #     """Header of the HTML representation of a SuiteSparseMatrix."""
    #     return (
    #         "<thead>"
    #         + "".join(
    #             f"<th>{attr}</th>" for attr in _SuiteSparseMatrix._attribute_names_list
    #         )
    #         + "</thead>"
    #     )
    #
    # def _repr_html_(self):
    #     """HTML representation."""
    #     return (
    #         f"<table>{_SuiteSparseMatrix.html_header()}"
    #         + f"<tbody>{self._to_html_row()}</tbody></table>"
    #     )


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
