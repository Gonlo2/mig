![](/assets/logo.readme.png?raw=true "my index guru logo")

# Introduction

Your own MySQL guru to deal with indexes in a friendly way.

## Usage

First you need [pipenv](https://pipenv.pypa.io/en/latest/#) to install the dependencies. Once installed it is possible to obtain an index recommendation with the command:

```shell
$ pipenv run ./mig.py recommend examples/recommend/ex001.toml 
Fields
  `aaa` SIZE(4)
  `bbb` SIZE(4)
  `ccc` SIZE(255)
  `xxx` SIZE(255)

Unique indexes
  `id`

Query #0
    SELECT * FROM `whatever`
        WHERE `aaa` = 123
Query #1
    SELECT * FROM `whatever`
        WHERE (`aaa` >= 22) AND (`aaa` <= 44)
Query #2
    SELECT * FROM `whatever`
        WHERE (`aaa` >= 22) AND (`bbb` = 1)
Query #3
    SELECT * FROM `whatever`
        WHERE (`aaa` >= 22) AND (`ccc` > "xyz")
Query #4
    SELECT * FROM `whatever`
        WHERE `aaa` >= 22
        ORDER BY `aaa` ASC
Query #5
    SELECT * FROM `whatever`
        ORDER BY `aaa` DESC
Query #6
    SELECT * FROM `whatever`
        WHERE (`aaa` = 123) AND (`bbb` = 1)
        GROUP BY `ccc`
Query #7
    SELECT * FROM `whatever`
        WHERE `aaa` = 123
        GROUP BY `ccc`
        ORDER BY `ddd` ASC
Query #8
    SELECT * FROM `whatever`
        WHERE `aaa` = 123
        GROUP BY `ccc`
        ORDER BY `ccc` ASC
Query #9
    SELECT * FROM `whatever`
        WHERE `ccc` >= 432
        ORDER BY `ccc` ASC
        LIMIT 10
Query #10
    SELECT * FROM `whatever`
        WHERE `ccc` in (1,2,3)
        ORDER BY `aaa` ASC
        LIMIT 10
Query #11
    SELECT * FROM `whatever`
        WHERE ((`aaa` = 123) AND (`bbb` = 1)) OR ((`ccc` = 22) AND (`aaa` = 123))

Some optimal indexes patterns
--------------------------------------------------------------------------------
Index #0
    Ordered subindex with the columns
      0. `ccc`
    To use by query: #9, #10
    Max size: 255
    Recommendation: `ccc`
    Log messages:
      - WARNING: Isn't possible optimize the `ORDER BY` in the query #10 because some of the `WHERE` filter a column with multiple values, probably with an `IN` or an `OR`. Remove the `ORDER BY` or execute multiple queries concatenating the output with an `UNION`
Index #1
    Ordered subindex with the columns
      0. `aaa`
      1. `ccc`
    To use by query: #0, #1, #3, #4, #5, #7, #8, #11
    Max size: 259
    Recommendation: `aaa`, `ccc`
    Log messages:
      - WARNING: Isn't possible optimize both the `ORDER BY` and the `GROUP BY` in the query #7, so only the `GROUP BY` will be optimized
      - WARNING: Isn't possible optimize all the `OR` operator of the query #11, try to refactor it to execute multiple queries concatenating the output with an `UNION`
Index #2
    Ordered subindex with the columns
      0. `bbb`
      1. `aaa`
      2. `ccc`
    To use by query: #2, #6
    Max size: 263
    Recommendation: `bbb`, `aaa`, `ccc`
```

## Table definition file format

At the moment only the btree indexes work with the operators `AND`, `OR`, `=`, `IS`, `<=`, `>=`, `>`, `<` and the query options `ORDER BY`, `GROUP BY` and `LIMIT`.

You could see some examples [here](./examples/recommend/).

## Credits

Created and maintained by [@Gonlo2](https://github.com/Gonlo2/).

## License

This project is licensed under the GPL-2.0 - see the [LICENSE](LICENSE) file for details
