"""Sparse matrices from the SuiteSparse Matrix Collection."""

import codecs
import csv
import io
import tarfile
from typing import Dict, Union

import numpy as np
import scipy.io

import probnum.linops as linops

# URLs and file paths to data
SUITESPARSE_ROOT_URL = "https://sparse.tamu.edu"
SUITESPARSE_INDEX_URL = SUITESPARSE_ROOT_URL + "/files/ssstats.csv"


def suitesparse_matrix(
    name: str,
    group: str,
    verbose: bool = False,
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
    verbose :
        Print additional information.

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1
           (2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.

    Examples
    --------
    >>> ssmat = suitesparse_matrix(name="ash85", group="HB")
    >>> ssmat
    <SuiteSparseMatrix with shape=(85, 85) and dtype=float64>
    >>> ssmat.trace()
    85.0
    """
    # Get database index
    try:
        import requests  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "Cannot query SuiteSparse Matrix collection without optional dependency ",
            "`requests`. Install ProbNum with optional dependencies for the problem zoo via",
            " `pip install probnum[zoo]` or install requests directly: `pip install requests`.",
        ) from err
    response = requests.get(SUITESPARSE_INDEX_URL, "r")
    line_gen = response.iter_lines()
    for _ in range(2):
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
    if verbose:
        print("Querying SuiteSparse Collection.")
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
    return SuiteSparseMatrix.from_database_entry(matrix_attr_dict)


class SuiteSparseMatrix(linops.Matrix):
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
    nnz  :
        Number of non-zero elements.
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

    # pylint: disable="too-many-instance-attributes,too-many-arguments,abstract-method"
    def __init__(
        self,
        matid: str,
        group: str,
        name: str,
        nnz: int,
        is2d3d: bool,
        isspd: bool,
        psym: float,
        nsym: float,
        kind: str,
    ):
        self.matid = matid
        self.group = group
        self.name = name
        self.nnz = nnz
        self.is2d3d = is2d3d
        self.isspd = isspd
        self.psym = psym
        self.nsym = nsym
        self.kind = kind

        super().__init__(A=self._download())

    @classmethod
    def from_database_entry(cls, database_entry: Dict) -> "SuiteSparseMatrix":
        """Create a SuiteSparseMatrix object from an entry of the database index.

        Parameters
        ----------
        database_entry :
            Dictionary representing one entry from the SuiteSparse database index.
        """
        return cls(
            matid=database_entry["matid"],
            group=database_entry["group"],
            nnz=int(database_entry["nnz"]),
            is2d3d=bool(int(database_entry["is2d3d"])),
            isspd=bool(int(database_entry["isspd"])),
            psym=float(database_entry["psym"]),
            nsym=float(database_entry["nsym"]),
            name=database_entry["name"],
            kind=database_entry["kind"],
        )

    def _download(
        self, verbose: bool = False
    ) -> Union[np.ndarray, scipy.sparse.coo_matrix]:
        """Download and extract file archive containing the sparse matrix.

        verbose:
            Print additional information.
        """
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "Cannot query SuiteSparse Matrix collection without optional dependency `requests`. Install ProbNum with optional dependencies for the problem zoo via `pip install probnum[zoo]` or install requests directly: `pip install requests`."
            ) from err

        url = SUITESPARSE_ROOT_URL + f"/MM/{self.group}/{self.name}.tar.gz"
        response = requests.get(url, stream=True)

        # Write archive to temporary file
        if verbose:
            print("Downloading compressed matrix.")

        chunk_size = 4096
        chunk_iter = response.iter_content(chunk_size=chunk_size)
        buffer = io.BytesIO()

        try:
            from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

            with tqdm(
                total=int(response.headers["content-length"]),
                desc=self.name,
                unit="B",
                unit_scale=True,
            ) as pbar:
                for chunk in chunk_iter:
                    buffer.write(chunk)
                    pbar.update(chunk_size)
        except ImportError:
            for chunk in chunk_iter:
                buffer.write(chunk)

        buffer.seek(0)
        if verbose:
            print("Extracting file archive.")
        tar = tarfile.open(fileobj=buffer, mode="r:gz")

        return scipy.io.mmread(tar.extractfile(tar.getmembers()[0]))

    @staticmethod
    def _html_header() -> str:
        """Header of the HTML representation of a SuiteSparseMatrix."""
        return (
            "<thead>"
            + "".join(
                f"<th>{attr}</th>"
                for attr in [
                    "ID",
                    "Group",
                    "Name",
                    "Rows",
                    "Cols",
                    "Nonzeros",
                    "DType",
                    "2D/3D Discretization",
                    "SPD",
                    "Pattern Symmetry",
                    "Numerical Symmetry",
                    "Domain",
                    "Preview",
                ]
            )
            + "</thead>"
        )

    def _to_html_row(self) -> str:
        return (
            "<tr>"
            + "".join(
                f"<td>{str(table_item)}</td>"
                for table_item in [
                    self.matid,
                    (
                        f'<a href="{SUITESPARSE_ROOT_URL + "/" + self.group}"'
                        f' target="_blank">{self.group}</a>'
                    ),
                    (
                        f'<a href="'
                        f'{SUITESPARSE_ROOT_URL + "/" + self.group + "/" + self.name}" '
                        f'target="_blank">{self.name}</a>'
                    ),
                    self.shape[0],
                    self.shape[1],
                    self.nnz,
                    self.dtype,
                    self.is2d3d,
                    self.isspd,
                    f"{self.psym:.2}",
                    f"{self.nsym:.2}",
                    self.kind,
                    f'<img src="{SUITESPARSE_ROOT_URL}/files/{self.group}/{self.name}.png">',
                ]
            )
            + "</tr>"
        )

    def _repr_html_(self) -> str:
        """HTML representation."""
        return (
            f"<table>{self._html_header()}"
            + f"<tbody>{self._to_html_row()}</tbody></table>"
        )
