"""Sparse matrices from the SuiteSparse Matrix Collection.

This implementation is based on Sudarshan Raghunathan's SSGETPY package
(https://github.com/drdarshan/ssgetpy).
"""
import dataclasses
import gzip
import os
import shutil
import sys
import tarfile
import time
from typing import Optional

import requests
import scipy.sparse
import tqdm.auto

from probnum.problems.zoo import PROBLEMZOO_DIR

# URLs and file paths to data
SUITESPARSE_ROOT_URL = "https://sparse.tamu.edu"
SUITESPARSE_INDEX_URL = os.path.join(SUITESPARSE_ROOT_URL, "files", "ssstats.csv")
SUITESPARSE_DIR = os.path.join(PROBLEMZOO_DIR, "linalg")
SUITESPARSE_DB = os.path.join(SUITESPARSE_DIR, "index.db")


def suitesparse_matrix(name: str) -> scipy.sparse.spmatrix:
    """Sparse matrix from the SuiteSparse Matrix Collection.

    Download a sparse matrix benchmark from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    name :
        Name of the sparse matrix.

    References
    ----------
    .. [1] Davis, TA and Hu, Y. The University of Florida sparse matrix
           collection. *ACM Transactions on Mathematical Software (TOMS)* 38.1
           (2011): 1-25.
    .. [2] Kolodziej, Scott P., et al. The SuiteSparse matrix collection website
           interface. *Journal of Open Source Software* 4.35 (2019): 1244.

    Examples
    --------
    """
    raise NotImplementedError


class SuiteSparseMatrixList(list):
    """List of SuiteSparseMatrix objects."""

    def _repr_html_(self):
        """HTML representation."""
        body = "".join(r.to_html_row() for r in self)
        return f"<table>{SuiteSparseMatrix.html_header()}<tbody>{body}</tbody></table>"

    def __getitem__(self, expr):
        result = super().__getitem__(expr)
        return SuiteSparseMatrixList(result) if isinstance(expr, slice) else result

    def download(self, format: str = "MM", destpath: str = None, extract: bool = False):
        """Downloads the list of SuiteSparseMatrices.

        Downloads the matrices to the local machine and unpacks any tar.gz
        files in the process.

        Parameters
        ----------
        format
        destpath
        extract
        """
        with tqdm.auto.tqdm(total=len(self), desc="Overall progress") as pbar:
            for matrix in self:
                matrix.download(format, destpath, extract)
                pbar.update(1)


@dataclasses.dataclass
class SuiteSparseMatrix:
    """SuiteSparse Matrix Benchmark.

    Object representing a sparse matrix from the `SuiteSparse Matrix Collection
    <https://sparse.tamu.edu/>`_. [1]_ [2]_

    Parameters
    ----------
    identifier :
        Unique identifier for the matrix in the database.
    group :
        Group this matrix belongs to.
    name :
        Name of this matrix.
    rows :
        Number of rows.
    cols :
        Number of columns.
    nonzeros  :
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

    identifier: str
    group: str
    name: str
    rows: int
    cols: int
    nonzeros: int
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

    def _filename(self, format: str = "MM") -> str:
        if format in ("MM", "RB"):
            return self.name + ".tar.gz"
        elif format == "MAT":
            return self.name + ".mat"
        else:
            raise ValueError("Format must be 'MM', 'MAT' or 'RB'")

    def _defaultdestpath(self, format: str = "MM"):
        return os.path.join(SUITESPARSE_DIR, format, self.group)

    def _localpath(
        self, format: str = "MM", destpath: Optional[str] = None, extract: bool = False
    ):
        # destpath is the directory containing the matrix
        destpath = destpath or self._defaultdestpath(format)

        # localdestpath is the directory containing the unzipped files
        # in the case of MM and RB (if extract is true) or
        # the file itself in the case of MAT (or if extract is False)
        localdest = os.path.join(destpath, self._filename(format))
        localdestpath = (
            localdest
            if (format == "MAT" or not extract)
            else os.path.join(destpath, self.name)
        )

        return localdestpath, localdest

    def url(self, format: str = "MM") -> str:
        """URL for this :class:`SuiteSparseMatrix` instance."""
        fname = self._filename(format)
        directory = format.lower() if format == "MAT" else format
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
        self, format: str = "MM", destpath: Optional[str] = None, extract: bool = False
    ):
        """Download the SuiteSparseMatrix.

        Downloads the matrix to the local machine and unpacks any tar.gz
        files in the process.

        Parameters
        ----------
        format
        destpath
        extract
        """
        # destpath is the directory containing the matrix
        destpath = destpath or self._defaultdestpath(format)

        # localdest is matrix file (.MAT or .TAR.GZ)
        # if extract = True, localdestpath is the directory
        # containing the unzipped matrix
        localdestpath, localdest = self._localpath(format, destpath, extract)

        if not os.access(localdestpath, os.F_OK):
            # Create the destination path if necessary
            os.makedirs(destpath, exist_ok=True)

            response = requests.get(self.url(format), stream=True)
            content_length = int(response.headers["content-length"])

            with open(localdest, "wb") as outfile, tqdm.auto.tqdm(
                total=content_length, desc=self.name, unit="B"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=4096):
                    outfile.write(chunk)
                    pbar.update(4096)
                    time.sleep(0.1)

            if extract and format in ("MM", "RB"):
                self._extract(localdest)

        return localdestpath, localdest

    def _extract(self, localdest: str):
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
                    SuiteSparseMatrix._attribute_names_list,
                    list(zip(*self.__dict__.items()))[
                        1
                    ],  # Attribute values of instance
                )
            )
            + "</tr>"
        )

    @staticmethod
    def html_header():
        return (
            "<thead>"
            + "".join(
                f"<th>{attr}</th>" for attr in SuiteSparseMatrix._attribute_names_list
            )
            + "</thead>"
        )

    def _repr_html_(self):
        """HTML representation."""
        return (
            f"<table>{SuiteSparseMatrix.html_header()}"
            + f"<tbody>{self._to_html_row()}</tbody></table>"
        )
