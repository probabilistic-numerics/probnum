"""Sparse matrices from the SuiteSparse Matrix Collection.

This implementation is based on Sudarshan Raghunathan's SSGETPY package
(https://github.com/drdarshan/ssgetpy).
"""
import contextlib
import csv
import dataclasses
import datetime
import gzip
import logging
import os
import shutil
import tarfile
import time
from typing import Optional, Union

import requests
import scipy.sparse
import tqdm.auto

from probnum.problems.zoo import PROBLEMZOO_DIR

logger = logging.getLogger(__name__)

# URLs and file paths to data
SUITESPARSE_ROOT_URL = "https://sparse.tamu.edu"
SUITESPARSE_INDEX_URL = os.path.join(SUITESPARSE_ROOT_URL, "files", "ssstats.csv")
SUITESPARSE_DIR = os.path.join(PROBLEMZOO_DIR, "linalg")
SUITESPARSE_DB = os.path.join(SUITESPARSE_DIR, "index.db")


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

    See Also
    --------
    suitesparse_query : Query the SuiteSparse Matrix Collection.

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

    >>>

    Download a sparse matrix and create a linear system from it.

    >>>
    """
    # Query SuiteSparse Matrix collection
    matrices = _suitesparse_query(name_or_id, **kwargs)

    # Download Matrices
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

    # pylint: disable="too-many-instance-attributes"
    matid: str
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

    def _filename(self, matrixformat: str = "MM") -> str:
        if matrixformat in ("MM", "RB"):
            return self.name + ".tar.gz"
        elif matrixformat == "MAT":
            return self.name + ".mat"
        else:
            raise ValueError("Format must be 'MM', 'MAT' or 'RB'")

    def _defaultdestpath(self, matrixformat: str = "MM"):
        return os.path.join(SUITESPARSE_DIR, matrixformat, self.group)

    def _localpath(
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
        localdestpath, localdest = self._localpath(matrixformat, destpath, extract)

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


def _from_timestamp(timestamp):
    if hasattr(datetime.datetime, "fromisoformat"):
        return datetime.datetime.fromisoformat(timestamp)
    return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


class _SuiteSparseMatrixDataBase:
    """SuiteSparse matrix database.

    Parameters
    ----------
    database :
        SQLite database file.
    matrix_table :
        Table containing matrices.
    """

    def __init__(self, database: str = SUITESPARSE_DB, matrix_table: str = "MATRICES"):
        import sqlite3

        self.database = database
        self.matrix_table = matrix_table
        self.update_table = "update_table"
        self.conn = sqlite3.connect(self.database)
        self._create_table()

    def _get_nrows(self):
        return int(
            self.conn.execute("SELECT COUNT(*) FROM %s" % self.matrix_table).fetchall()[
                0
            ][0]
        )

    nrows = property(_get_nrows)

    def _get_last_update(self):
        last_update = self.conn.execute(
            "SELECT MAX(update_date) " + f"from {self.update_table}"
        ).fetchall()[0][0]
        return (
            _from_timestamp(last_update)
            if last_update
            else datetime.datetime.utcfromtimestamp(0)
        )

    last_update = property(_get_last_update)

    def _drop_table(self):
        self.conn.execute("DROP TABLE IF EXISTS %s" % self.matrix_table)
        self.conn.execute(f"DROP TABLE IF EXISTS {self.update_table}")
        self.conn.commit()

    def _create_table(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS %s (
                             id INTEGER PRIMARY KEY,
                             matrixgroup TEXT,
                             name TEXT,
                             rows INTEGER,
                             cols INTEGER,
                             nnz INTEGER,
                             dtype TEXT,
                             is2d3d INTEGER,
                             isspd INTEGER,
                             psym REAL,
                             nsym REAL,
                             kind TEXT)"""
            % self.matrix_table
        )

        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.update_table} "
            + "(update_date TIMESTAMP)"
        )
        self.conn.commit()

    def insert(self, values):
        self.conn.executemany(
            "INSERT INTO %s VALUES(?,?,?,?,?,?,?,?,?,?,?,?)" % self.matrix_table,
            values,
        )
        self.conn.execute(
            f"INSERT INTO {self.update_table} " + "VALUES (datetime('now'))"
        )
        self.conn.commit()

    def refresh(self, values):
        self._drop_table()
        self._create_table()
        self.insert(values)

    def dump(self):
        return self.conn.execute("SELECT * from %s" % self.matrix_table).fetchall()

    @staticmethod
    def _is_constraint(field, value):
        return value and "(%s = '%s')" % (field, value)

    @staticmethod
    def _like_constraint(field, value):
        return value and "(%s LIKE '%%%s%%')" % (field, value)

    @staticmethod
    def _sz_constraint(field, bounds):
        if bounds is None or (bounds[0] is None and bounds[1] is None):
            return None
        constraints = []
        if bounds[0] is not None:
            constraints.append("%s >= %d" % (field, bounds[0]))
        if bounds[1] is not None:
            constraints.append("%s <= %d" % (field, bounds[1]))
        return " ( " + " AND ".join(constraints) + " ) "

    @staticmethod
    def _bool_constraint(field, value):
        if value is None:
            return None
        elif value:
            return "(%s = 1)" % field
        else:
            return "(%s = 0)" % field

    def search(
        self,
        matid=None,
        group=None,
        name=None,
        rowbounds=None,
        colbounds=None,
        nzbounds=None,
        dtype=None,
        is2d3d=None,
        isspd=None,
        kind=None,
        limit=10,
    ):

        querystring = "SELECT * FROM %s" % self.matrix_table

        mid_constraint = _SuiteSparseMatrixDataBase._is_constraint("id", matid)
        grp_constraint = _SuiteSparseMatrixDataBase._is_constraint("matrixgroup", group)
        nam_constraint = _SuiteSparseMatrixDataBase._like_constraint("name", name)
        row_constraint = _SuiteSparseMatrixDataBase._sz_constraint("rows", rowbounds)
        col_constraint = _SuiteSparseMatrixDataBase._sz_constraint("cols", colbounds)
        nnz_constraint = _SuiteSparseMatrixDataBase._sz_constraint("nnz", nzbounds)
        dty_constraint = _SuiteSparseMatrixDataBase._is_constraint("dtype", dtype)
        geo_constraint = _SuiteSparseMatrixDataBase._bool_constraint("is2d3d", is2d3d)
        spd_constraint = _SuiteSparseMatrixDataBase._bool_constraint("isspd", isspd)
        knd_constraint = _SuiteSparseMatrixDataBase._like_constraint("kind", kind)

        constraints = list(
            filter(
                lambda x: x is not None,
                (
                    mid_constraint,
                    grp_constraint,
                    nam_constraint,
                    row_constraint,
                    col_constraint,
                    nnz_constraint,
                    dty_constraint,
                    geo_constraint,
                    spd_constraint,
                    knd_constraint,
                ),
            )
        )

        if any(constraints):
            querystring += " WHERE " + " AND ".join(constraints)

        querystring += " LIMIT (%s)" % limit

        logger.debug(querystring)

        return _SuiteSparseMatrixList(
            _SuiteSparseMatrix(*x) for x in self.conn.execute(querystring).fetchall()
        )


# Parse ssstats.csv file and generate entries for each row in the SuiteSparseMatrix DB
def getdtype(real, logical):
    """Converts a (real, logical) pair into one of the three types:

    'real', 'complex' and 'binary'
    """
    return "binary" if logical else ("real" if real else "complex")


def gen_rows(csvrows):
    """Creates a generator that returns a single row in the matrix database."""
    reader = csv.reader(csvrows)


def csvindex_generate():
    with contextlib.closing(
        requests.get(SUITESPARSE_INDEX_URL, stream=True)
    ) as response:
        reader = csv.reader(response.iter_lines(), delimiter=",", quotechar='"')
        matid = 0
        for line in reader:
            matid += 1
            group = line[0]
            name = line[1]
            rows = int(line[2])
            cols = int(line[3])
            nnz = int(line[4])
            real = bool(int(line[5]))
            logical = bool(int(line[6]))
            is2d3d = bool(int(line[7]))
            isspd = bool(int(line[8]))
            psym = float(line[9])
            nsym = float(line[10])
            kind = line[11]
            yield matid, group, name, rows, cols, nnz, getdtype(
                real, logical
            ), is2d3d, isspd, psym, nsym, kind


# SuiteSparseMatrix Database Singleton
os.makedirs(SUITESPARSE_DIR, exist_ok=True)
suitesparse_db_instance = _SuiteSparseMatrixDataBase()

if suitesparse_db_instance.nrows == 0 or (
    datetime.datetime.utcnow() - suitesparse_db_instance.last_update
) > datetime.timedelta(days=90):
    logger.info("{Re}creating index from CSV file...")
    suitesparse_db_instance.refresh(csvindex_generate())
