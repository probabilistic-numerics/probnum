"""Class representing SuiteSparse Matrix Benchmark objects."""

import dataclasses
import gzip
import os
import shutil
import tarfile
import time
from typing import Optional

import requests
import tqdm.auto

from .._suitesparse import SUITESPARSE_DIR, SUITESPARSE_ROOT_URL

__all__ = ["_SuiteSparseMatrixList", "_SuiteSparseMatrix"]


class _SuiteSparseMatrixList(list):
    """List of SuiteSparseMatrix objects."""

    def _repr_html_(self):
        """HTML representation."""
        body = "".join(r.to_html_row() for r in self)
        return f"<table>{_SuiteSparseMatrix.html_header()}<tbody>{body}</tbody></table>"

    def __getitem__(self, expr):
        result = super().__getitem__(expr)
        return _SuiteSparseMatrixList(result) if isinstance(expr, slice) else result

    def download(
        self, matrixformat: str = "MM", destpath: str = None, extract: bool = False
    ):
        """Downloads the list of SuiteSparseMatrices.

        Downloads the matrices to the local machine and unpacks any tar.gz
        files in the process.

        Parameters
        ----------
        matrixformat
        destpath
        extract
        """
        with tqdm.auto.tqdm(total=len(self), desc="Overall progress") as pbar:
            for matrix in self:
                matrix.download(matrixformat, destpath, extract)
                pbar.update(1)


@dataclasses.dataclass
class _SuiteSparseMatrix:
    """SuiteSparse Matrix Benchmark.

    Object representing a sparse matrix from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    matid :
        Unique identifier for the matrix in the database.
    group :
        Group this matrix belongs to.
    name :
        Name of this matrix.
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

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1 (
           2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.
    """

    # pylint: disable="too-many-instance-attributes"
    matid: str
    group: str
    name: str
    rows: int
    cols: int
    nnz: int
    dtype: str
    is2d3d: bool
    isspd: bool
    psym: float
    nsym: float
    kind: str

    _attribute_names_list = [
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
        "Kind",
        "Preview",
    ]

    def _filename(self, matrixformat: str = "MM") -> str:
        if matrixformat in ("MM", "RB"):
            return self.name + ".tar.gz"
        elif matrixformat == "MAT":
            return self.name + ".mat"
        else:
            raise ValueError("Format must be 'MM', 'MAT' or 'RB'")

    def _defaultdestpath(self, matrixformat: str = "MM"):
        return os.path.join(SUITESPARSE_DIR, matrixformat, self.group)

    def localpath(
        self,
        matrixformat: str = "MM",
        destpath: Optional[str] = None,
        extract: bool = False,
    ):
        # destpath is the directory containing the matrix
        destpath = destpath or self._defaultdestpath(matrixformat)

        # localdestpath is the directory containing the unzipped files
        # in the case of MM and RB (if extract is true) or
        # the file itself in the case of MAT (or if extract is False)
        localdest = os.path.join(destpath, self._filename(matrixformat))
        localdestpath = (
            localdest
            if (matrixformat == "MAT" or not extract)
            else os.path.join(destpath, self.name)
        )

        return localdestpath, localdest

    def url(self, matrixformat: str = "MM") -> str:
        """URL for this :class:`SuiteSparseMatrix` instance."""
        fname = self._filename(matrixformat)
        directory = matrixformat.lower() if matrixformat == "MAT" else matrixformat
        return os.path.join(SUITESPARSE_ROOT_URL, directory, self.group, fname)

    @property
    def _attribute_urls(self):
        """URLs associated with this SuiteSparse matrix."""
        _root_url = "https://sparse.tamu.edu"
        return {
            "index": os.path.join(_root_url, "files", "ssstats.csv"),
            "preview": os.path.join(
                SUITESPARSE_ROOT_URL, "files", self.group, f"{self.name}.png"
            ),
            "group_info": os.path.join(SUITESPARSE_ROOT_URL, self.group),
            "matrix_info": os.path.join(SUITESPARSE_ROOT_URL, self.group, self.name),
        }

    def download(
        self,
        matrixformat: str = "MM",
        destpath: Optional[str] = None,
        extract: bool = False,
    ):
        """Download the SuiteSparseMatrix.

        Downloads the matrix to the local machine and unpacks any tar.gz
        files in the process.

        Parameters
        ----------
        matrixformat
        destpath
        extract
        """
        # destpath is the directory containing the matrix
        destpath = destpath or self._defaultdestpath(matrixformat)

        # localdest is matrix file (.MAT or .TAR.GZ)
        # if extract = True, localdestpath is the directory
        # containing the unzipped matrix
        localdestpath, localdest = self.localpath(matrixformat, destpath, extract)

        if not os.access(localdestpath, os.F_OK):
            # Create the destination path if necessary
            os.makedirs(destpath, exist_ok=True)

            response = requests.get(self.url(matrixformat), stream=True)
            content_length = int(response.headers["content-length"])

            with open(localdest, "wb") as outfile, tqdm.auto.tqdm(
                total=content_length, desc=self.name, unit="B"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=4096):
                    outfile.write(chunk)
                    pbar.update(4096)
                    time.sleep(0.1)

            if extract and matrixformat in ("MM", "RB"):
                _SuiteSparseMatrix._extract(localdest)

        return localdestpath, localdest

    @staticmethod
    def _extract(localdest: str):
        """Extract the file archive under the given path."""
        basedir, filename = os.path.split(localdest)
        tarfilename = os.path.join(basedir, ".".join((filename.split(".")[0], "tar")))

        gzfile = gzip.open(localdest, "rb")
        with open(tarfilename, "wb") as outtarfile:
            shutil.copyfileobj(gzfile, outtarfile)
        gzfile.close()

        tarfile.open(tarfilename).extractall(basedir)
        os.unlink(tarfilename)
        os.unlink(localdest)

    def _render_item_html(self, key, value):
        if key == "Group":
            return (
                f'<a href="{self._attribute_urls["group_info"]}" '
                f'target="_blank">{value}</a>'
            )
        if key == "Name":
            return (
                f'<a href="{self._attribute_urls["matrix_info"]}" '
                f'target="_blank">{value}</a>'
            )
        if key in ("Pattern Symmetry", "Numerical Symmetry"):
            return f"{value:0.2}"
        if key in ("2D/3D Discretization", "SPD"):
            return "Yes" if value else "No"
        if key == "Preview":
            return f'<img src="{value}">'

        return str(value)

    def _to_html_row(self):
        return (
            "<tr>"
            + "".join(
                f"<td>{self._render_item_html(key, value)}</td>"
                for key, value in zip(
                    _SuiteSparseMatrix._attribute_names_list,
                    list(zip(*self.__dict__.items()))[
                        1
                    ],  # Attribute values of instance
                )
            )
            + "</tr>"
        )

    @staticmethod
    def html_header():
        """Header of the HTML representation of a SuiteSparseMatrix."""
        return (
            "<thead>"
            + "".join(
                f"<th>{attr}</th>" for attr in _SuiteSparseMatrix._attribute_names_list
            )
            + "</thead>"
        )

    def _repr_html_(self):
        """HTML representation."""
        return (
            f"<table>{_SuiteSparseMatrix.html_header()}"
            + f"<tbody>{self._to_html_row()}</tbody></table>"
        )
