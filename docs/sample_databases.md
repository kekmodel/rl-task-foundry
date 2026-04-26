# Sample Databases

This project expects live PostgreSQL sample databases for integration tests and real-db trials.

## Pagila

`rl_task_foundry.yaml` now points at Pagila, the PostgreSQL port of Sakila:

```text
postgresql://pagila:pagila@127.0.0.1:5433/pagila
```

Bootstrap it with Docker:

```bash
scripts/bootstrap_sample_dbs.sh pagila
```

The script starts a `postgres:17` container, downloads `pagila-schema.sql` and
`pagila-data.sql` from `devrimgunduz/pagila`, loads them, and grants `rlvr_reader`.

## Postgres Air

`rl_task_foundry.postgres_air.yaml` points at:

```text
postgresql://postgres_air:postgres_air@127.0.0.1:5434/postgres_air
```

Postgres Air dumps are large. By default, the script downloads the latest
`postgres_air_2024.sql.zip` dump from the official Google Drive folder and restores it:

```bash
scripts/bootstrap_sample_dbs.sh postgres_air
```

Use another official edition/format if needed:

```bash
POSTGRES_AIR_EDITION=2023 POSTGRES_AIR_FORMAT=backup scripts/bootstrap_sample_dbs.sh postgres_air
```

To restore an already downloaded file:

```bash
POSTGRES_AIR_DUMP=/path/to/postgres_air_2024.sql.zip scripts/bootstrap_sample_dbs.sh postgres_air
```

You can also provide a direct URL:

```bash
POSTGRES_AIR_URL=https://example.com/postgres_air_2024.sql.zip scripts/bootstrap_sample_dbs.sh postgres_air
```

Supported restore formats are `.sql`, `.sql.zip`, `.backup`, and `.dump`.

## Both

```bash
scripts/bootstrap_sample_dbs.sh all
```

The command uses plain `docker run` and `docker exec`; it does not require
`docker compose` or host-installed `psql`/`pg_restore`.

## MIMIC-IV

MIMIC-IV is optional because the full dataset requires PhysioNet access. A demo
config and full config are included:

```bash
scripts/bootstrap_sample_dbs.sh mimiciv_demo
scripts/bootstrap_sample_dbs.sh mimiciv
```

The demo path uses the public PhysioNet MIMIC-IV demo release. The full path
uses `PHYSIONET_USERNAME`/`PHYSIONET_PASSWORD`, `~/.netrc`, or an existing
`MIMIC_IV_DATA_DIR`.
