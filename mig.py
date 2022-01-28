#!/usr/bin/env python3
import logging
import textwrap
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
import tomlkit


def create_arg_parser():
    p = ArgumentParser(description='My index guru',
                       formatter_class=ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(title='subcommands',
                          description='valid subcommands',
                          help='additional help')
    p.set_defaults(func=dummy_cmd)

    p.add_argument('--log-level', default='INFO',
                   choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                   help='logger level')

    configure_recommend_arg_parser(sp.add_parser('recommend', help='Recommend the indexes to use'))

    return p


def configure_recommend_arg_parser(p):
    p.set_defaults(func=recommend_cmd)
    p.add_argument('table_def', type=FileType('r'), help='the file with the table definition')


def dummy_cmd(args):
    pass


class Sql:
    def generate_sql(self):
        raise NotImplementedError


class Operator(Sql):
    def flatten(self):
        raise NotImplementedError

    def get_column_restrictions(self):
        raise NotImplementedError


@dataclass
class ValueOperator:
    uid: str
    value: Any

    def flatten(self):
        return [self]

    def generate_sql(self):
        if isinstance(self.value, (int, float)):
            return str(self.value)
        if isinstance(self.value, str):
            return f'"{self.value}"'
        if isinstance(self.value, bool):
            return "1" if self.value else "0"
        if self.value is None:
            return "null"
        raise NotImplementedError


@dataclass
class BinaryOperator(Operator):
    operator: str
    column: Any
    value: Any

    def flatten(self):
        return [self]

    def get_column_restrictions(self):
        return {self.column: (self.get_column_restriction(), Counter([self.get_value_uid()]))}

    def get_column_restriction(self):
        raise NotImplementedError

    def get_value_uid(self):
        if not isinstance(self.value, ValueOperator):
            raise NotImplementedError
        return self.value.uid

    def generate_sql(self):
        return f"`{self.column}` {self.operator} {self.value.generate_sql()}"


class EqualOperator(BinaryOperator):
    def get_column_restriction(self):
        return ColumnRestriction.EQ


class RangeOperator(BinaryOperator):
    def get_column_restriction(self):
        return ColumnRestriction.RANGE


@dataclass
class VariableOperator(Operator):
    values: Any


@dataclass
class InOperator(VariableOperator):
    column: Any

    def flatten(self):
        values = [EqualOperator("=", self.column, x) for x in self.values]
        return OrOperator(values).flatten()

    def generate_sql(self):
        values = ','.join(x.generate_sql() for x in self.values)
        return f"`{self.column}` in ({values})"


class ColumnRestriction(Enum):
    NONE = 0
    EQ = 1
    RANGE = 2
    EQ_RANGE = 3

    def has_eq(self):
        return (self.value & ColumnRestriction.EQ.value) == ColumnRestriction.EQ.value

    def has_range(self):
        return (self.value & ColumnRestriction.RANGE.value) == ColumnRestriction.RANGE.value

    def merge(self, other):
        return ColumnRestriction(self.value | other.value)


class AndOperator(VariableOperator):
    def flatten(self):
        if not self.values:
            return [self]
        to_check = [self]
        values = []
        while to_check:
            for x in to_check.pop().values:
                if isinstance(x, AndOperator):
                    to_check.append(x)
                else:
                    values.append(x)
        flatten = [[x] for x in values[0].flatten()]
        for i in range(1, len(values)):
            tmp_flatten = []
            for x in values[i].flatten():
                for y in flatten:
                    tmp_flatten.append(y + [x])
            flatten = tmp_flatten
        return [AndOperator(x) for x in flatten]

    def get_column_restrictions(self):
        restr_by_column = {}
        for op in self.values:
            if not isinstance(op, BinaryOperator):
                raise NotImplementedError
            new_rest = op.get_column_restriction()
            value_uid = op.get_value_uid()
            if t := restr_by_column.get(op.column):
                t[1][value_uid] += 1
                restr_by_column[op.column] = (t[0].merge(new_rest), t[1])
            else:
                restr_by_column[op.column] = (new_rest, Counter([value_uid]))
        return restr_by_column

    def generate_sql(self):
        return ' AND '.join(f"({x.generate_sql()})" for x in self.values)


class OrOperator(VariableOperator):
    def flatten(self):
        to_check = [self]
        values = []
        while to_check:
            for x in to_check.pop().values:
                if isinstance(x, OrOperator):
                    to_check.append(x)
                else:
                    values.extend(x.flatten())
        return values

    def generate_sql(self):
        return ' OR '.join(f"({x.generate_sql()})" for x in self.values)


class Query(Sql):
    pass


class Logger:
    _id = 0

    class Level(Enum):
        WARNING = "WARNING"

    def __init__(self, messages=None):
        self._messages = messages or {}

    def warning(self, tmpl, *args, **kwargs):
        self._add_log(self.Level.WARNING, tmpl, args, kwargs)

    def _add_log(self, level, tmpl, args, kwargs):
        self._messages[Logger._id] = (level, tmpl.format(*args, **kwargs))
        Logger._id += 1

    def union(self, other):
        messages = {**self._messages, **other._messages}
        return Logger(messages)

    def messages(self):
        return [v for _, v in sorted(self._messages.items())]


@dataclass
class SelectQuery(Query):
    where: Optional[Operator]
    group_by: Optional[List[str]]
    order_by: Optional[List[Tuple[str, bool]]]
    limit: Optional[int]

    def generate_sql(self):
        parts = ["SELECT * FROM `whatever`"]
        if self.where is not None:
            parts.append(f"    WHERE {self.where.generate_sql()}")
        if self.group_by is not None:
            group_by = ', '.join(f"`{name}`" for name in self.group_by)
            parts.append(f"    GROUP BY {group_by}")
        if self.order_by is not None:
            order_by = ', '.join(f"`{name}` {'ASC' if asc else 'DESC'}"
                                 for name, asc in self.order_by)
            parts.append(f"    ORDER BY {order_by}")
        if self.limit is not None:
            parts.append(f"    LIMIT {self.limit}")
        return '\n'.join(parts)

    def generate_index(self, query_id):
        logger = Logger()

        order_by_columns = [name for name, _ in self.order_by or []]
        group_by_columns = set(x for x in self.group_by or order_by_columns)
        if not all(x in group_by_columns for x in order_by_columns):
            logger.warning("Isn't possible optimize both the `ORDER BY` and the `GROUP BY` in the query #{}, so only the `GROUP BY` will be optimized", query_id)
            order_by_columns = []

        if self.where is None:
            subindexes = self._generate_subindexes({},
                                                   order_by_columns,
                                                   group_by_columns)
            return Index(subindexes, queries=set([query_id]), logger=logger)

        flatten_operators = self.where.flatten()
        columns_restrictions = [op.get_column_restrictions() for op in flatten_operators]

        union_of_columns = {}
        for restrictions in columns_restrictions:
            for name, (restr, cnt) in restrictions.items():
                column = Column(name=name, restriction=restr, values=cnt)
                other_column = union_of_columns.get(name)
                if other_column is not None:
                    column = column.union(other_column)
                union_of_columns[name] = column

        if order_by_columns:
            # Avoid add the order by to the index if some comparation value
            # use a `IN` operator or a number of `OR`
            could_not_use_order_by = any(
                c.restriction.has_eq()
                and c.name not in group_by_columns
                and c.num_max_searchs() > 1
                for c in union_of_columns.values()
            )
            if could_not_use_order_by:
                logger.warning("Isn't possible optimize the `ORDER BY` in the query #{} because some of the `WHERE` filter a column with multiple values, probably with an `IN` or an `OR`. Remove the `ORDER BY` or execute multiple queries concatenating the output with an `UNION`", query_id)
                order_by_columns = []
                if not self.group_by:
                    group_by_columns = set()

        indexes = []
        for restrictions in columns_restrictions:
            columns = {name: Column(name=name, restriction=restr, values=cnt)
                       for name, (restr, cnt) in restrictions.items()}
            subindexes = self._generate_subindexes(columns,
                                                   order_by_columns,
                                                   group_by_columns)
            index = Index(subindexes, queries=set([query_id]), logger=logger)
            indexes.append(index)

        index = union_indexes(indexes)
        if index.len() != max(x.len() for x in indexes):
            #TODO deal with empty indexes
            index.logger.warning("Isn't possible optimize all the `OR` operator of the query #{}, try to refactor it to execute multiple queries concatenating the output with an `UNION`", query_id)
        return index

    def _generate_subindexes(self, columns, order_by_columns, group_by_columns):
        subindexes = []
        columns_added = set()

        subindex = self._generate_base_subindex(columns, group_by_columns, columns_added)
        if subindex.len() > 0:
            subindexes.append(subindex)

        if order_by_columns:
            subindex = self._generate_orderby_subindex(columns, order_by_columns, columns_added)
            if subindex.len() > 0:
                subindexes.append(subindex)

        if group_by_columns:
            subindex = self._generate_groupby_subindex(columns, group_by_columns, columns_added)
            if subindex.len() > 0:
                subindexes.append(subindex)
        elif not order_by_columns:
            subindex = self._generate_range_subindex(columns, columns_added)
            if subindex.len() > 0:
                subindexes.append(subindex)

        return subindexes

    def _generate_base_subindex(self, columns, group_by_columns, columns_added):
        subindex_columns = {}
        for c in columns.values():
            if c.restriction.has_eq() and c.name not in group_by_columns and c.name not in columns_added:
                subindex_columns[c.name] = c
                columns_added.add(c.name)
        return UnorderedSubindex(subindex_columns)

    def _generate_orderby_subindex(self, columns, order_by_columns, columns_added):
        subindex_columns = []
        for name in order_by_columns:
            if name not in columns_added:
                c = columns.get(name)
                if c is None:
                    c = Column(name=name)
                subindex_columns.append(c)
                columns_added.add(name)
        return OrderedSubindex(subindex_columns)

    def _generate_groupby_subindex(self, columns, group_by_columns, columns_added):
        subindex_columns = {}
        for name in group_by_columns:
            if name not in columns_added:
                c = columns.get(name)
                if c is None:
                    c = Column(name=name)
                subindex_columns[name] = c
                columns_added.add(name)
        return UnorderedSubindex(subindex_columns)

    def _generate_range_subindex(self, columns, columns_added):
        subindex_columns = {}
        for c in columns.values():
            if c.restriction.has_range() and c.name not in columns_added:
                subindex_columns[c.name] = c
                columns_added.add(c.name)
        return MultipleSubindex(subindex_columns)


@dataclass
class Column:
    name: str
    restriction: ColumnRestriction = ColumnRestriction.NONE
    values: Counter = field(default_factory=Counter)

    def union(self, other: "Column") -> "Column":
        restriction = self.restriction.merge(other.restriction)
        values = self.values + other.values
        return Column(name=self.name, restriction=restriction, values=values)

    def key(self):
        return f"{self.name}@{len(self.values)}"

    def priority(self, table):
        return (self.num_max_searchs(), -table.fields[self.name].size, self.name)

    def num_max_searchs(self):
        return len(self.values)


# Kind of indexes
# - Ordered
# - Unordered
# - Multiple
class Subindex:
    def unify(self):
        raise NotImplementedError

    def append_if_possible(self, other):
        raise NotImplementedError

    def explain(self):
        raise NotImplementedError

    def key(self):
        raise NotImplementedError

    def restriction_level(self):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError

    def columns_till(self, till):
        raise NotImplementedError

    def max_size(self, table):
        raise NotImplementedError

    def num_max_searchs(self, table):
        raise NotImplementedError

    def choose_recommendation(self, table):
        raise NotImplementedError

    def columns(self, idx):
        raise NotImplementedError

    def column(self, idx, name):
        raise NotImplementedError

    def union(self, other, length):
        raise NotImplementedError

    def disjoint(self, other):
        raise NotImplementedError


class OrderedSubindex:
    def __init__(self, columns):
        self._columns = columns

    def unify(self):
        return self

    def append_if_possible(self, other):
        if isinstance(other, OrderedSubindex):
            return OrderedSubindex(self._columns + other._columns)
        return None

    def explain(self, start):
        lines = ["Ordered subindex with the columns"]
        for i, c in enumerate(self._columns):
            lines.append(f"  {start+i}. `{c.name}`")
        return '\n'.join(lines)

    def key(self):
        columns = ["ordered"]
        columns.extend(x.key() for x in self._columns)
        return ':'.join(columns)

    def restriction_level(self):
        return 0

    def __repr__(self):
        return f"OrderedSubindex(columns: {self._columns})"

    def len(self):
        return len(self._columns)

    def columns_till(self, till):
        return self._columns[:till]

    def max_size(self, table):
        return sum(table.fields[c.name].size for c in self._columns)

    def num_max_searchs(self, table):
        r = 1
        rr = 0
        for c in self._columns:
            r *= c.num_max_searchs()
            rr += r
        return rr

    def choose_recommendation(self, table):
        return self._columns

    def columns(self, idx):
        return [self._columns[idx]]

    def column(self, idx, name):
        return c if (c := self._columns[idx]).name == name else None

    def union(self, other, length):
        new_columns = []
        for i in range(length):
            c = self._columns[i]
            oc = other.column(i, c.name)
            if oc is None:
                break
            new_c = c.union(oc)
            new_columns.append(new_c)
        return OrderedSubindex(new_columns)

    def disjoint(self, other):
        return OrderedSubindex(self._columns[other.len():])


class UnorderedSubindex:
    def __init__(self, columns):
        self._columns = columns

    def unify(self):
        return self if len(self._columns) > 1 else OrderedSubindex(list(self._columns.values()))

    def append_if_possible(self, other):
        return None

    def explain(self, start):
        lines = ["Unordered subindex with the columns"]
        for name in sorted(self._columns.keys()):
            lines.append(f"  {start}-{start+len(self._columns)-1}. `{name}`")
        return '\n'.join(lines)

    def key(self):
        columns = ["unordered"]
        columns.extend(sorted(x.key() for x in self._columns.values()))
        return ':'.join(columns)

    def restriction_level(self):
        return 2

    def __repr__(self):
        return f"UnorderedSubindex(columns: {self._columns})"

    def len(self):
        return len(self._columns)

    def columns_till(self, till):
        return self._columns.values()

    def max_size(self, table):
        return sum(table.fields[c.name].size for c in self._columns.values())

    def num_max_searchs(self, table):
        r = 1
        rr = 0
        for c in sorted(self._columns.values(), key=lambda x: x.priority(table)):
            r *= c.num_max_searchs()
            rr += r
        return rr

    def choose_recommendation(self, table):
        return list(sorted(self._columns.values(), key=lambda x: x.priority(table)))

    def columns(self, idx):
        return self._columns.values()

    def column(self, idx, name):
        return self._columns.get(name)

    def union(self, other, length):
        new_columns = {}
        for oc in other.columns_till(length):
            c = self._columns.get(oc.name)
            if c is not None:
                new_columns[c.name] = c.union(oc)
        return UnorderedSubindex(new_columns)

    def disjoint(self, other):
        new_columns = dict(self._columns)
        for c in other.columns_till(other.len()):
            new_columns.pop(c.name, None)
        return UnorderedSubindex(new_columns)


class MultipleSubindex:
    def __init__(self, columns):
        self._columns = columns

    def unify(self):
        return self if len(self._columns) > 1 else OrderedSubindex(list(self._columns.values()))

    def append_if_possible(self, other):
        return None

    def explain(self, start):
        lines = ["Multiple subindex with any of the columns"]
        for name in sorted(self._columns.keys()):
            lines.append(f"  {start}. `{name}`")
        return '\n'.join(lines)

    def key(self):
        columns = ["multiple"]
        columns.extend(sorted(x.key() for x in self._columns.values()))
        return ':'.join(columns)

    def restriction_level(self):
        return 1

    def __repr__(self):
        return f"MultipleSubindex(columns: {self._columns})"

    def len(self):
        return 1 if self._columns else 0

    def columns_till(self, till):
        return self._columns.values()

    def max_size(self, table):
        return max(table.fields[c.name].size for c in self._columns.values())

    def num_max_searchs(self, table):
        c = min(self._columns.values(), key=lambda x: x.priority(table))
        return c.num_max_searchs()

    def choose_recommendation(self, table):
        return [min(self._columns.values(), key=lambda x: x.priority(table))]

    def columns(self, idx):
        return self._columns.values()

    def column(self, idx, name):
        return self._columns.get(name)

    def union(self, other, length):
        new_columns = {}
        for oc in other.columns_till(length):
            c = self._columns.get(oc.name)
            if c is not None:
                new_columns[c.name] = c.union(oc)
        return MultipleSubindex(new_columns)

    def disjoint(self, other):
        new_columns = dict(self._columns)
        for c in other.columns_till(other.len()):
            new_columns.pop(c.name, None)
        return MultipleSubindex(new_columns)


def union_indexes(indexes):
    queries = set()
    logger = Logger()
    for index in indexes:
        queries.update(index.queries)
        logger = logger.union(index.logger)

    tmp_index = TmpIndex([])
    tmp_indexes = [TmpIndex(list(reversed(x._subindexes))) for x in indexes]
    while tmp_indexes:
        subindex = _get_prefix(tmp_indexes)
        tmp_index = tmp_index.append(subindex)
        if subindex.len() < tmp_indexes[-1].subindexes[-1].len():
            break
        tmp_indexes = _remove_prefix(tmp_indexes, subindex)
    return tmp_index.to_index(queries, logger)


@dataclass
class TmpIndex:
    subindexes: List[Subindex]

    def append(self, subindex):
        subindexes = list(self.subindexes)
        subindex = subindex.unify()
        if subindex.len() > 0 and self.subindexes:
            new_subindex = self.subindexes[-1].append_if_possible(subindex)
            if new_subindex is not None:
                subindexes[-1] = new_subindex
                return TmpIndex(subindexes)
        subindexes.append(subindex)
        return TmpIndex(subindexes)

    def to_index(self, queries, logger):
        return Index(self.subindexes, queries, logger)

    def clone(self):
        return TmpIndex(list(self.subindexes))


def _get_prefix(tmp_indexes):
    idx, _ = min(enumerate(tmp_indexes), key=lambda x: x[1].subindexes[-1].len())
    tmp_indexes[-1], tmp_indexes[idx] = tmp_indexes[idx], tmp_indexes[-1]
    subindex = tmp_indexes[-1].subindexes[-1]
    for i in range(len(tmp_indexes)-1):
        osubindex = tmp_indexes[i].subindexes[-1]
        if subindex.restriction_level() < osubindex.restriction_level():
            subindex = subindex.union(osubindex, subindex.len())
        else:
            subindex = osubindex.union(subindex, subindex.len())
    return subindex


def _remove_prefix(tmp_indexes, subindex):
    new_tmp_indexes = []
    for tmp_index in tmp_indexes:
        tmp_index = tmp_index.clone()
        tmp_index.subindexes[-1] = tmp_index.subindexes[-1].disjoint(subindex)
        if tmp_index.subindexes[-1].len() == 0:
            tmp_index.subindexes.pop()
        if tmp_index.subindexes:
            new_tmp_indexes.append(tmp_index)
    return new_tmp_indexes


class Index:
    def __init__(self, subindexes: List[Subindex], queries: Set[int], logger: Logger):
        self._subindexes = subindexes
        self.queries = queries
        self.logger = logger
        self._subindexes_mapping = self._make_subindexes_mapping()

    def __repr__(self):
        return f"Index(queries: {self.queries}, subindexes: {self._subindexes})"

    def peiority(self):
        return len(self._subindexes_mapping)

    def explain(self):
        start = 0
        lines = []
        for subindex in self._subindexes:
            desc = subindex.explain(start)
            lines.append(desc)
            start += subindex.len()
        return '\n'.join(lines)

    def key(self):
        return '|'.join(x.key() for x in self._subindexes)

    def _make_subindexes_mapping(self):
        result = []
        for i, subindex in enumerate(self._subindexes):
            for j in range(subindex.len()):
                result.append((i, j))
        return result

    def max_size(self, table):
        return sum(x.max_size(table) for x in self._subindexes)

    def num_max_searchs(self, table):
        r = 1
        rr = 0
        for subindex in self._subindexes:
            r *= subindex.num_max_searchs(table)
            rr += r
        return rr

    def choose_recommendation(self, table):
        example = []
        for x in self._subindexes:
            example.extend(x.choose_recommendation(table))
        return example

    def columns_till(self, till):
        columns = []
        for subindex in self._subindexes:
            subindex_size = subindex.len()
            size = min(subindex_size, till)
            columns.extend(subindex.columns_till(size))
            till -= subindex_size
            if till <= 0:
                break
        return columns

    def columns(self, idx):
        i, j = self._subindexes_mapping[idx]
        return self._subindexes[i].columns(j)

    def column(self, idx, name):
        i, j = self._subindexes_mapping[idx]
        return self._subindexes[i].column(j, name)

    def len(self):
        return len(self._subindexes_mapping)

    def is_prefix_of(self, other):
        if self.len() > other.len():
            return False
        for i in range(self.len()):
            if all(other.column(i, c.name) is None for c in self.columns(i)):
                return False
        return True


@dataclass
class Field:
    name: str
    size: int


@dataclass
class Table:
    fields: Dict[str, Field]
    unique_indexes: List[Set[str]]
    queries: List[Query]


class Parser:
    def __init__(self):
        super().__init__()
        self._value_id = 0

    def parse(self, table_def):
        fields = {}
        for x in table_def.get('fields', []):
            field = Field(x["name"], x["size"])
            fields[field.name] = field

        unique_indexes = []
        for x in table_def.get('unique_indexes', []):
            unique_indexes.append(set(x['columns']))

        queries = []
        for x in table_def.get('queries'):
            where = None
            order_by = None
            if sql := x.get('sql'):
                expression_tree = sqlglot.parse_one(sql)
                x = self._parse_sql(expression_tree)
            if value := x.get('where'):
                where = self._parse_where(value)
            if value := x.get('order_by'):
                order_by = self._parse_order_by(value)
            group_by = x.get('group_by')
            limit = x.get('limit')
            query = SelectQuery(where=where, group_by=group_by, order_by=order_by, limit=limit)
            queries.append(query)

        return Table(fields, unique_indexes, queries)

    def _parse_sql(self, tree):
        if isinstance(tree, sqlglot.expressions.Select):
            query = {}
            if value := tree.args['where']:
                query['where'] = self._parse_sql(value)
            if value := tree.args['group']:
                query['group_by'] = self._parse_sql(value)
            if value := tree.args['order']:
                query['order_by'] = self._parse_sql(value)
            if value := tree.args['limit']:
                query['limit'] = self._parse_sql(value)
            return query
        if isinstance(tree, sqlglot.expressions.Where):
            return self._parse_sql(tree.this)
        if isinstance(tree, sqlglot.expressions.Group):
            return [self._parse_sql(x) for x in tree.args['expressions']]
        if isinstance(tree, sqlglot.expressions.Order):
            return [self._parse_sql(x) for x in tree.args['expressions']]
        if isinstance(tree, sqlglot.expressions.Ordered):
            col = self._parse_sql(tree.this)
            desc = tree.args['desc']
            return [col, False] if desc else col
        if isinstance(tree, sqlglot.expressions.Limit):
            return self._parse_sql(tree.this)
        if isinstance(tree, sqlglot.expressions.Column):
            return self._parse_sql(tree.this)

        if isinstance(tree, sqlglot.expressions.Paren):
            return self._parse_sql(tree.this)
        if isinstance(tree, sqlglot.expressions.And):
            v1 = self._parse_sql(tree.this)
            v2 = self._parse_sql(tree.args['expression'])
            return {"operator": "and", "values": [v1, v2]}
        if isinstance(tree, sqlglot.expressions.Or):
            v1 = self._parse_sql(tree.this)
            v2 = self._parse_sql(tree.args['expression'])
            return {"operator": "or", "values": [v1, v2]}

        if isinstance(tree, sqlglot.expressions.LT):
            col = self._parse_sql(tree.this)
            val = self._parse_sql(tree.args['expression'])
            return {"operator": "<", "column": col, "value": val}
        if isinstance(tree, sqlglot.expressions.LTE):
            col = self._parse_sql(tree.this)
            val = self._parse_sql(tree.args['expression'])
            return {"operator": "<=", "column": col, "value": val}
        if isinstance(tree, sqlglot.expressions.GT):
            col = self._parse_sql(tree.this)
            val = self._parse_sql(tree.args['expression'])
            return {"operator": ">", "column": col, "value": val}
        if isinstance(tree, sqlglot.expressions.GTE):
            col = self._parse_sql(tree.this)
            val = self._parse_sql(tree.args['expression'])
            return {"operator": ">=", "column": col, "value": val}
        if isinstance(tree, sqlglot.expressions.EQ):
            col = self._parse_sql(tree.this)
            val = self._parse_sql(tree.args['expression'])
            return {"operator": "=", "column": col, "value": val}
        if isinstance(tree, sqlglot.expressions.In):
            col = self._parse_sql(tree.this)
            vals = [self._parse_sql(x) for x in tree.args['expressions']]
            return {"operator": "in", "column": col, "values": vals}

        if isinstance(tree, sqlglot.expressions.Identifier):
            #TODO Need to remove inner quotes (?)
            return tree.this.strip('`')
        if isinstance(tree, sqlglot.expressions.Literal):
            return tree.this if tree.args['is_string'] else int(tree.this)

        raise NotImplementedError

    def _parse_where(self, where):
        if not isinstance(where, dict):
            uid = self._make_value_uid()
            return ValueOperator(uid, where)

        operator = where['operator']
        if operator == "value":
            uid = where.get('id')
            uid = self._make_value_uid() if uid is None else f"manual:{uid}"
            return ValueOperator(uid, where['value'])
        elif operator == "and":
            values = [self._parse_where(x) for x in where['values']]
            return AndOperator(values)
        elif operator == "or":
            values = [self._parse_where(x) for x in where['values']]
            return OrOperator(values)
        elif operator == "in":
            values = [self._parse_where(x) for x in where['values']]
            return InOperator(values, where['column'])
        elif operator in ("=", "is"):
            return EqualOperator(operator, where['column'],
                                 self._parse_where(where.get('value')))
        elif operator in ("<", ">", "<=", ">="):
            return RangeOperator(operator, where['column'],
                                 self._parse_where(where.get('value')))
        raise NotImplementedError

    def _make_value_uid(self):
        uid = f"auto:{self._value_id}"
        self._value_id += 1
        return uid

    def _parse_order_by(self, order_by):
        result = []
        for x in order_by:
            if isinstance(x, str):
                result.append((x, True))
            else:
                result.append((x[0], x[1]))
        return result


@dataclass
class SearchNode:
    size: int
    num_max_searchs: int
    num_queries_covered: int
    num_queries_duplicated: int
    queries_covered: List[bool]
    indexes: List[Index]

    @staticmethod
    def new(num_queries):
        queries_covered = [False] * num_queries
        return SearchNode(0, 0, 0, 0, queries_covered, [])

    def key(self):
        return tuple(self.queries_covered)

    def priority(self):
        return (self.size, self.num_max_searchs, self.num_queries_duplicated)

    def successors(self, table, indexes):
        for index in indexes:
            if all(self.queries_covered[i] for i in index.queries):
                continue

            queries_covered = list(self.queries_covered)
            num_queries_covered = self.num_queries_covered
            num_queries_duplicated = self.num_queries_duplicated
            for i in index.queries:
                if queries_covered[i]:
                    num_queries_duplicated += 1
                else:
                    num_queries_covered += 1
                queries_covered[i] = True
            indexes = list(self.indexes)
            indexes.append(index)
            size = self.size + index.max_size(table)
            num_max_searchs = self.num_max_searchs + index.num_max_searchs(table)
            yield SearchNode(size=size, num_max_searchs=num_max_searchs,
                             num_queries_covered=num_queries_covered,
                             num_queries_duplicated=num_queries_duplicated,
                             queries_covered=queries_covered,
                             indexes=indexes)


class IndexesSelector:
    def __init__(self, table, indexes, num_queries_covered):
        super().__init__()
        self._table = table
        self._indexes = indexes
        self._all_indexes = []
        self._max_size_by_query = []
        self._num_queries_covered = num_queries_covered

    def execute(self):
        self._prepare_all_indexes()
        return self._search_indexes()

    def _prepare_all_indexes(self):
        all_indexes = {x.key(): x for x in self._indexes}
        step_indexes = dict(all_indexes)
        while step_indexes:
            new_step_indexes = {}
            for x in step_indexes.values():
                for i, y in enumerate(self._indexes):
                    if i not in x.queries:
                        new_index = union_indexes([x, y])
                        if new_index.len() == max(x.len(), y.len()):
                            key = new_index.key()
                            step_index = new_step_indexes.get(key)
                            if step_index is None:
                                new_step_indexes[key] = new_index
                            else:
                                step_index.queries.update(new_index.queries)
            step_indexes = {}
            for key, index in new_step_indexes.items():
                other_index = all_indexes.get(key)
                if other_index is None:
                    step_indexes[key] = index
                    all_indexes[key] = index
                elif index.queries != other_index.queries:
                    other_index.queries.update(index.queries)
                    step_indexes[key] = other_index

        # The indexes are sorted to make the search reproducible
        self._all_indexes = [v for _, v in sorted(all_indexes.items())]

    def _search_indexes(self):
        sn = SearchNode.new(len(self._table.queries))
        heap = [(sn.priority(), 0, sn)]
        idx = 1  # Use a counter to make the search reproducible
        sn_seen = {}
        while heap:
            _, _, sn = heappop(heap)
            if sn.num_queries_covered == self._num_queries_covered:
                return sn.indexes
            for sn in sn.successors(self._table, self._all_indexes):
                key = sn.key()
                priority = sn.priority()
                seen_priority = sn_seen.get(key, priority)
                if seen_priority >= priority:
                    sn_seen[key] = priority
                    heappush(heap, (priority, idx, sn))
                    idx += 1

        return None


def recommend_cmd(args):
    table_def = tomlkit.parse(args.table_def.read())

    parser = Parser()

    table = parser.parse(table_def)

    print("Fields")
    for i, field in enumerate(table.fields.values()):
        print(f"  `{field.name}` SIZE({field.size})")
    print()

    if table.unique_indexes:
        print("Unique indexes")
        for unique_index in table.unique_indexes:
            s = ', '.join(f"`{x}`" for x in sorted(unique_index))
            print(f"  {s}")
        print()

    indexes = []
    for i, query in enumerate(table.queries):
        print(f"Query #{i}")
        print(textwrap.indent(query.generate_sql(), "    "))

        index = query.generate_index(i)
        if index.len() > 0:
            indexes.append(index)

    num_queries_covered = len(indexes)

    unique_indexes = []
    for uk_col_names in table.unique_indexes:
        queries_covered = set()
        uk_columns = {}
        for index in indexes:
            num_uk_columns_used = 0
            columns = index.columns_till(index.len())
            for c in columns:
                if c.restriction == ColumnRestriction.EQ:
                    if c.name in uk_col_names:
                        num_uk_columns_used += 1
            if num_uk_columns_used == len(uk_col_names):
                queries_covered.update(index.queries)
                for c in columns:
                    if c.name in uk_col_names:
                        cc = uk_columns.get(c.name)
                        if cc is not None:
                            c = c.union(cc)
                        uk_columns[c.name] = c
        if queries_covered:
            subindex = UnorderedSubindex(uk_columns)
            #TODO Deal with the indexes logger
            uk = Index([subindex], queries_covered, logger=Logger())
            unique_indexes.append(uk)

    indexes.extend(unique_indexes)

    indexes_selector = IndexesSelector(table, indexes, num_queries_covered)
    indexes = indexes_selector.execute() or []

    print()
    print("Some optimal indexes patterns")
    print("-" * 80)
    for i, index in enumerate(indexes):
        print(f"Index #{i}")
        print(textwrap.indent(index.explain(), "    "))
        columns = index.choose_recommendation(table)
        columns_names = ', '.join(f"`{x.name}`" for x in columns)
        to_use_by = ', '.join(f"#{x}" for x in sorted(index.queries))
        print(f"    To use by query: {to_use_by}")
        print(f"    Max size: {index.max_size(table)}")
        print(f"    Recommendation: {columns_names}")
        if messages := index.logger.messages():
            print("    Log messages:")
            for level, msg in messages:
                print(f"      - {level.value}: {msg}")


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()
