"""SuiteSparse Database."""

import csv
import datetime
import logging
import os
import sqlite3
from typing import Iterator, Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)

from .._suitesparse import SUITESPARSE_DB, SUITESPARSE_DIR, SUITESPARSE_INDEX_URL
from ._matrix import _SuiteSparseMatrix, _SuiteSparseMatrixList

__all__ = ["suitesparse_db_instance"]


class _SuiteSparseMatrixDataBase:
    """SuiteSparse matrix database.

    Parameters
    ----------
    database :
        SQLite database location.
    matrix_table :
        Name of the table containing the matrices.
    """

    def __init__(self, database: str = SUITESPARSE_DB, matrix_table: str = "MATRICES"):
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
        # TODO: check for non-tuple
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
        name: Optional[str] = None,
        matid: Optional[int] = None,
        group: Optional[str] = None,
        rows: Optional[Union[Tuple[int, int], int]] = None,
        cols: Optional[Union[Tuple[int, int], int]] = None,
        nnz: Optional[Union[Tuple[int, int], int]] = None,
        dtype: Optional[str] = None,
        isspd: Optional[bool] = None,
        psym: Optional[Union[Tuple[float, float], float]] = None,
        nsym: Optional[Union[Tuple[float, float], float]] = None,
        is2d3d: Optional[bool] = None,
        kind: Optional[str] = None,
        limit=10,
    ):

        querystring = "SELECT * FROM %s" % self.matrix_table

        mid_constraint = _SuiteSparseMatrixDataBase._is_constraint("id", matid)
        grp_constraint = _SuiteSparseMatrixDataBase._is_constraint("matrixgroup", group)
        nam_constraint = _SuiteSparseMatrixDataBase._like_constraint("name", name)
        row_constraint = _SuiteSparseMatrixDataBase._sz_constraint("rows", rows)
        col_constraint = _SuiteSparseMatrixDataBase._sz_constraint("cols", cols)
        nnz_constraint = _SuiteSparseMatrixDataBase._sz_constraint("nnz", nnz)
        dty_constraint = _SuiteSparseMatrixDataBase._is_constraint("dtype", dtype)
        geo_constraint = _SuiteSparseMatrixDataBase._bool_constraint("is2d3d", is2d3d)
        spd_constraint = _SuiteSparseMatrixDataBase._bool_constraint("isspd", isspd)
        psym_constraint = _SuiteSparseMatrixDataBase._sz_constraint("psym", psym)
        nsym_constraint = _SuiteSparseMatrixDataBase._sz_constraint("nsym", nsym)
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
                    psym_constraint,
                    nsym_constraint,
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


def _from_timestamp(timestamp: str):
    if hasattr(datetime.datetime, "fromisoformat"):
        return datetime.datetime.fromisoformat(timestamp)
    return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


# Parse ssstats.csv file and generate entries for each row in the SuiteSparseMatrix DB
def getdtype(real, logical):
    """Converts a (real, logical) pair into one of the three types:

    'real', 'complex' and 'binary'
    """
    return "binary" if logical else ("real" if real else "complex")


def gen_rows(csvrows: Iterator[str]):
    """Creates a generator that returns a single row in the matrix database."""
    reader = csv.reader(csvrows)
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


def csvindex_generate():
    response = requests.get(SUITESPARSE_INDEX_URL)
    lines = response.iter_lines()

    # Read the number of entries
    logger.info(f"Number of entries in the CSV file: {next(lines)}")
    # Read the last modified date
    logger.info(f"Last modified date: {next(lines)}")

    return gen_rows(line.decode("utf-8") for line in lines)


# SuiteSparseMatrix Database Singleton
os.makedirs(SUITESPARSE_DIR, exist_ok=True)
suitesparse_db_instance = _SuiteSparseMatrixDataBase()

# Refresh the database index if the last update was > 90 days ago
if suitesparse_db_instance.nrows == 0 or (
    datetime.datetime.utcnow() - suitesparse_db_instance.last_update
) > datetime.timedelta(days=90):
    logger.info("{Re}creating index from CSV file...")
    suitesparse_db_instance.refresh(csvindex_generate())
