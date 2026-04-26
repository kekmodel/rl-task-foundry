#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${DB_CACHE_DIR:-$ROOT_DIR/artifacts/sample_dbs}"
PG_IMAGE="${PG_IMAGE:-postgres:17}"

PAGILA_CONTAINER="${PAGILA_CONTAINER:-rl-task-foundry-pagila}"
PAGILA_VOLUME="${PAGILA_VOLUME:-rl-task-foundry-pagila-data}"
PAGILA_PORT="${PAGILA_PORT:-5433}"
PAGILA_DB="${PAGILA_DB:-pagila}"
PAGILA_USER="${PAGILA_USER:-pagila}"
PAGILA_PASSWORD="${PAGILA_PASSWORD:-pagila}"

POSTGRES_AIR_CONTAINER="${POSTGRES_AIR_CONTAINER:-rl-task-foundry-postgres-air}"
POSTGRES_AIR_VOLUME="${POSTGRES_AIR_VOLUME:-rl-task-foundry-postgres-air-data}"
POSTGRES_AIR_PORT="${POSTGRES_AIR_PORT:-5434}"
POSTGRES_AIR_DB="${POSTGRES_AIR_DB:-postgres_air}"
POSTGRES_AIR_USER="${POSTGRES_AIR_USER:-postgres_air}"
POSTGRES_AIR_PASSWORD="${POSTGRES_AIR_PASSWORD:-postgres_air}"

PAGILA_SCHEMA_URL="${PAGILA_SCHEMA_URL:-https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-schema.sql}"
PAGILA_DATA_URL="${PAGILA_DATA_URL:-https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-data.sql}"
POSTGRES_AIR_SOURCE_URL="https://github.com/hettie-d/postgres_air"
POSTGRES_AIR_DRIVE_URL="https://drive.google.com/drive/folders/13F7M80Kf_somnjb-mTYAnh1hW1Y_g4kJ"
POSTGRES_AIR_EDITION="${POSTGRES_AIR_EDITION:-2024}"
POSTGRES_AIR_FORMAT="${POSTGRES_AIR_FORMAT:-sql.zip}"

MIMIC_IV_CONTAINER="${MIMIC_IV_CONTAINER:-rl-task-foundry-mimiciv}"
MIMIC_IV_VOLUME="${MIMIC_IV_VOLUME:-rl-task-foundry-mimiciv-data}"
MIMIC_IV_PORT="${MIMIC_IV_PORT:-5435}"
MIMIC_IV_DB="${MIMIC_IV_DB:-mimiciv}"
MIMIC_IV_USER="${MIMIC_IV_USER:-mimiciv}"
MIMIC_IV_PASSWORD="${MIMIC_IV_PASSWORD:-mimiciv}"
MIMIC_IV_VERSION="${MIMIC_IV_VERSION:-3.1}"
MIMIC_IV_BASE_URL="${MIMIC_IV_BASE_URL:-https://physionet.org/files/mimiciv}"
MIMIC_IV_CACHE_NAME="${MIMIC_IV_CACHE_NAME:-mimiciv}"
MIMIC_IV_REQUIRES_CREDENTIALS="${MIMIC_IV_REQUIRES_CREDENTIALS:-1}"
MIMIC_CODE_REPO="${MIMIC_CODE_REPO:-https://github.com/MIT-LCP/mimic-code.git}"
MIMIC_CODE_REF="${MIMIC_CODE_REF:-}"
MIMIC_CODE_DIR="${MIMIC_CODE_DIR:-}"

target="${1:-all}"

log() {
  printf '[sample-dbs] %s\n' "$*" >&2
}

die() {
  printf '[sample-dbs] ERROR: %s\n' "$*" >&2
  exit 1
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker CLI is required"
  docker info >/dev/null 2>&1 || die "Docker daemon is not running"
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

ensure_postgres_container() {
  local name="$1"
  local volume="$2"
  local port="$3"
  local db="$4"
  local user="$5"
  local password="$6"

  if container_exists "$name"; then
    log "starting existing container $name"
    docker start "$name" >/dev/null
    return
  fi

  log "creating container $name on localhost:$port"
  docker run \
    --name "$name" \
    --detach \
    --publish "127.0.0.1:${port}:5432" \
    --env "POSTGRES_DB=$db" \
    --env "POSTGRES_USER=$user" \
    --env "POSTGRES_PASSWORD=$password" \
    --volume "${volume}:/var/lib/postgresql/data" \
  "$PG_IMAGE" >/dev/null
}

mimiciv_host_data_dir() {
  if [[ -n "${MIMIC_IV_DATA_DIR:-}" ]]; then
    printf '%s\n' "$MIMIC_IV_DATA_DIR"
  else
    printf '%s\n' "$CACHE_DIR/$MIMIC_IV_CACHE_NAME/$MIMIC_IV_VERSION"
  fi
}

ensure_mimiciv_container() {
  local host_data_dir
  host_data_dir="$(mimiciv_host_data_dir)"
  mkdir -p "$host_data_dir"

  if container_exists "$MIMIC_IV_CONTAINER"; then
    log "starting existing container $MIMIC_IV_CONTAINER"
    docker start "$MIMIC_IV_CONTAINER" >/dev/null
    return
  fi

  log "creating container $MIMIC_IV_CONTAINER on localhost:$MIMIC_IV_PORT"
  docker run \
    --name "$MIMIC_IV_CONTAINER" \
    --detach \
    --publish "127.0.0.1:${MIMIC_IV_PORT}:5432" \
    --env "POSTGRES_DB=$MIMIC_IV_DB" \
    --env "POSTGRES_USER=$MIMIC_IV_USER" \
    --env "POSTGRES_PASSWORD=$MIMIC_IV_PASSWORD" \
    --volume "${MIMIC_IV_VOLUME}:/var/lib/postgresql/data" \
    --volume "$host_data_dir:/mimiciv/$MIMIC_IV_VERSION:ro" \
    "$PG_IMAGE" >/dev/null
}

wait_for_postgres() {
  local name="$1"
  local db="$2"
  local user="$3"

  log "waiting for $name"
  for _ in {1..90}; do
    if docker exec "$name" pg_isready -U "$user" -d "$db" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  die "$name did not become ready"
}

psql_exec() {
  local name="$1"
  local db="$2"
  local user="$3"
  docker exec -i "$name" psql -U "$user" -d "$db" -v ON_ERROR_STOP=1
}

table_exists() {
  local name="$1"
  local db="$2"
  local user="$3"
  local regclass="$4"

  local result
  result="$(
    docker exec "$name" psql -U "$user" -d "$db" -Atqc \
      "SELECT to_regclass('$regclass') IS NOT NULL" 2>/dev/null || true
  )"
  [[ "$result" == "t" ]]
}

table_has_rows() {
  local name="$1"
  local db="$2"
  local user="$3"
  local relation="$4"

  local result
  result="$(
    docker exec "$name" psql -U "$user" -d "$db" -Atqc \
      "SELECT EXISTS (SELECT 1 FROM $relation LIMIT 1)" 2>/dev/null || true
  )"
  [[ "$result" == "t" ]]
}

grant_readonly_role() {
  local name="$1"
  local db="$2"
  local user="$3"
  local schema="$4"

  psql_exec "$name" "$db" "$user" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rlvr_reader') THEN
    CREATE ROLE rlvr_reader;
  END IF;
END
\$\$;
GRANT CONNECT ON DATABASE "$db" TO rlvr_reader;
GRANT USAGE ON SCHEMA "$schema" TO rlvr_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA "$schema" TO rlvr_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA "$schema" TO rlvr_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA "$schema" GRANT SELECT ON TABLES TO rlvr_reader;
GRANT rlvr_reader TO "$user";
SQL
}

grant_readonly_role_many_schemas() {
  local name="$1"
  local db="$2"
  local user="$3"
  shift 3

  psql_exec "$name" "$db" "$user" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rlvr_reader') THEN
    CREATE ROLE rlvr_reader;
  END IF;
END
\$\$;
GRANT CONNECT ON DATABASE "$db" TO rlvr_reader;
SQL

  local schema
  for schema in "$@"; do
    grant_readonly_role "$name" "$db" "$user" "$schema"
  done
}

ensure_role_exists() {
  local name="$1"
  local db="$2"
  local user="$3"
  local role="$4"

  psql_exec "$name" "$db" "$user" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$role') THEN
    CREATE ROLE "$role";
  END IF;
END
\$\$;
SQL
}

download_if_missing() {
  local url="$1"
  local path="$2"

  if [[ -s "$path" ]]; then
    log "using cached $(basename "$path")"
    return
  fi
  log "downloading $url"
  mkdir -p "$(dirname "$path")"
  curl --fail --location --progress-bar "$url" --output "$path"
}

postgres_air_google_drive_file() {
  case "$POSTGRES_AIR_EDITION:$POSTGRES_AIR_FORMAT" in
    2024:sql.zip)
      printf '%s %s\n' "1C7PVxeYvLDr6n_7qjdA2k0vahv__jMEo" "postgres_air_2024.sql.zip"
      ;;
    2024:backup)
      printf '%s %s\n' "1nNXcTjhJK6EzJdzaZvdoJDrYj_V61rq8" "postgres_air_2024.backup"
      ;;
    2023:sql.zip)
      printf '%s %s\n' "1kyR_8waFniYsxWHeJKOvR5k1UMmLKqFn" "postgres_air_2023.sql.zip"
      ;;
    2023:backup)
      printf '%s %s\n' "1sXIYJNkDj1uozkIMd8JlYh25wM0G5F11" "postgres_air_2023.backup"
      ;;
    initial:sql.zip)
      printf '%s %s\n' "1OWw1uP_vp017zlelMKzpO0kkM5VRraRi" "postgres_air.sql.zip"
      ;;
    initial:backup)
      printf '%s %s\n' "1lvn-10AI6__UX2--xlozz961EVt9GYbH" "postgres_air.backup"
      ;;
    *)
      die "unsupported POSTGRES_AIR_EDITION/POSTGRES_AIR_FORMAT: $POSTGRES_AIR_EDITION/$POSTGRES_AIR_FORMAT"
      ;;
  esac
}

download_postgres_air_from_google_drive() {
  local air_dir="$1"
  local file_id
  local filename
  read -r file_id filename < <(postgres_air_google_drive_file)

  local path="$air_dir/$filename"
  local url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
  download_if_missing "$url" "$path"
  printf '%s\n' "$path"
}

bootstrap_pagila() {
  require_docker
  ensure_postgres_container \
    "$PAGILA_CONTAINER" "$PAGILA_VOLUME" "$PAGILA_PORT" \
    "$PAGILA_DB" "$PAGILA_USER" "$PAGILA_PASSWORD"
  wait_for_postgres "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER"

  local pagila_dir="$CACHE_DIR/pagila"
  local schema="$pagila_dir/pagila-schema.sql"
  local data="$pagila_dir/pagila-data.sql"
  download_if_missing "$PAGILA_SCHEMA_URL" "$schema"
  download_if_missing "$PAGILA_DATA_URL" "$data"
  ensure_role_exists "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER" "postgres"

  if table_exists "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER" "public.customer"; then
    log "pagila is already loaded"
  else
    log "loading pagila schema"
    psql_exec "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER" < "$schema"
    log "loading pagila data"
    psql_exec "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER" < "$data"
  fi

  grant_readonly_role "$PAGILA_CONTAINER" "$PAGILA_DB" "$PAGILA_USER" "public"
  log "pagila ready: postgresql://$PAGILA_USER:$PAGILA_PASSWORD@127.0.0.1:$PAGILA_PORT/$PAGILA_DB"
}

resolve_postgres_air_dump() {
  if [[ -n "${POSTGRES_AIR_DUMP:-}" ]]; then
    [[ -f "$POSTGRES_AIR_DUMP" ]] || die "POSTGRES_AIR_DUMP does not exist: $POSTGRES_AIR_DUMP"
    printf '%s\n' "$POSTGRES_AIR_DUMP"
    return
  fi

  local air_dir="$CACHE_DIR/postgres_air"
  mkdir -p "$air_dir"

  if [[ -n "${POSTGRES_AIR_URL:-}" ]]; then
    local file="$air_dir/$(basename "${POSTGRES_AIR_URL%%\?*}")"
    download_if_missing "$POSTGRES_AIR_URL" "$file"
    printf '%s\n' "$file"
    return
  fi

  local candidate
  candidate="$(find "$air_dir" -maxdepth 1 -type f \( \
    -name 'postgres_air*.sql' -o \
    -name 'postgres_air*.sql.zip' -o \
    -name 'postgres_air*.backup' -o \
    -name 'postgres_air*.dump' \
  \) | sort -r | head -n 1)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  if [[ "${POSTGRES_AIR_AUTO_DOWNLOAD:-1}" != "0" ]]; then
    download_postgres_air_from_google_drive "$air_dir"
    return
  fi

  cat >&2 <<MSG
[sample-dbs] postgres_air dump not found.
[sample-dbs] Download one dump from:
[sample-dbs]   $POSTGRES_AIR_DRIVE_URL
[sample-dbs] Source repo:
[sample-dbs]   $POSTGRES_AIR_SOURCE_URL
[sample-dbs]
[sample-dbs] Then rerun one of:
[sample-dbs]   POSTGRES_AIR_DUMP=/path/to/postgres_air_2024.sql.zip scripts/bootstrap_sample_dbs.sh postgres_air
[sample-dbs]   POSTGRES_AIR_URL=https://example.com/postgres_air_2024.sql.zip scripts/bootstrap_sample_dbs.sh postgres_air
[sample-dbs] Or allow the built-in Google Drive downloader:
[sample-dbs]   POSTGRES_AIR_AUTO_DOWNLOAD=1 scripts/bootstrap_sample_dbs.sh postgres_air
MSG
  return 1
}

restore_postgres_air_dump() {
  local dump="$1"
  local lower
  lower="$(printf '%s' "$dump" | tr '[:upper:]' '[:lower:]')"

  case "$lower" in
    *.sql)
      log "loading postgres_air SQL dump"
      psql_exec "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER" < "$dump"
      ;;
    *.sql.zip|*.zip)
      command -v unzip >/dev/null 2>&1 || die "unzip is required for $dump"
      log "loading postgres_air zipped SQL dump"
      unzip -p "$dump" '*.sql' | psql_exec "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER"
      ;;
    *.backup|*.dump)
      log "restoring postgres_air custom-format dump"
      docker exec -i "$POSTGRES_AIR_CONTAINER" pg_restore \
        -U "$POSTGRES_AIR_USER" \
        -d "$POSTGRES_AIR_DB" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --verbose < "$dump"
      ;;
    *)
      die "unsupported postgres_air dump format: $dump"
      ;;
  esac
}

bootstrap_postgres_air() {
  require_docker
  ensure_postgres_container \
    "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_VOLUME" "$POSTGRES_AIR_PORT" \
    "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER" "$POSTGRES_AIR_PASSWORD"
  wait_for_postgres "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER"

  if table_exists "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER" "postgres_air.airport"; then
    log "postgres_air is already loaded"
  else
    ensure_role_exists "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER" "postgres"
    local dump
    dump="$(resolve_postgres_air_dump)"
    restore_postgres_air_dump "$dump"
  fi

  grant_readonly_role "$POSTGRES_AIR_CONTAINER" "$POSTGRES_AIR_DB" "$POSTGRES_AIR_USER" "postgres_air"
  log "postgres_air ready: postgresql://$POSTGRES_AIR_USER:$POSTGRES_AIR_PASSWORD@127.0.0.1:$POSTGRES_AIR_PORT/$POSTGRES_AIR_DB"
}

resolve_mimic_code_dir() {
  local ref="${MIMIC_CODE_REF:-main}"
  local code_dir="${MIMIC_CODE_DIR:-$CACHE_DIR/mimic-code-$ref}"

  if [[ -d "$code_dir/.git" ]]; then
    log "using cached mimic-code repo"
    printf '%s\n' "$code_dir"
    return
  fi

  mkdir -p "$(dirname "$code_dir")"
  log "cloning mimic-code repo"
  local clone_args=(--depth 1 --branch "$ref")
  git clone "${clone_args[@]}" "$MIMIC_CODE_REPO" "$code_dir" >/dev/null
  printf '%s\n' "$code_dir"
}

download_mimiciv_from_physionet() {
  local data_root="$CACHE_DIR/$MIMIC_IV_CACHE_NAME"
  local data_dir="$data_root/$MIMIC_IV_VERSION"

  if [[ -d "$data_dir/hosp" && -d "$data_dir/icu" ]]; then
    log "using cached MIMIC-IV data at $data_dir"
    printf '%s\n' "$data_dir"
    return
  fi

  if [[ "$MIMIC_IV_REQUIRES_CREDENTIALS" != "0" && ( -z "${PHYSIONET_USERNAME:-}" || -z "${PHYSIONET_PASSWORD:-}" ) ]]; then
    if [[ ! -f "$HOME/.netrc" ]]; then
      die "MIMIC-IV requires PhysioNet credentials. Set PHYSIONET_USERNAME/PHYSIONET_PASSWORD or add a physionet.org entry to ~/.netrc."
    fi
  fi

  log "downloading MIMIC-IV $MIMIC_IV_VERSION from PhysioNet"
  mkdir -p "$data_root"
  "${PYTHON:-python3}" - "$MIMIC_IV_BASE_URL" "$MIMIC_IV_VERSION" "$data_root" <<'PY'
from __future__ import annotations

import os
import sys
from html.parser import HTMLParser
from netrc import netrc
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    build_opener,
)

base_url = sys.argv[1].rstrip("/") + "/"
version = sys.argv[2].strip("/")
data_root = Path(sys.argv[3])
start_url = urljoin(base_url, version + "/")
dest_root = data_root / version


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def credentials() -> tuple[str, str] | None:
    username = os.environ.get("PHYSIONET_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD")
    if username and password:
        return username, password
    try:
        auth = netrc().authenticators("physionet.org")
    except FileNotFoundError:
        return None
    if auth is None:
        return None
    login, _, passwd = auth
    return login, passwd


creds = credentials()
password_mgr = HTTPPasswordMgrWithDefaultRealm()
if creds:
    password_mgr.add_password(None, "https://physionet.org", creds[0], creds[1])
opener = build_opener(HTTPBasicAuthHandler(password_mgr))

seen: set[str] = set()
queue = [start_url]
files: list[str] = []

while queue:
    url = queue.pop(0)
    if url in seen:
        continue
    seen.add(url)
    try:
        with opener.open(url) as response:
            content_type = response.headers.get("content-type", "")
            body = response.read()
    except HTTPError as exc:
        raise SystemExit(
            f"failed to read {url}: HTTP {exc.code}. Check PhysioNet access for MIMIC-IV {version}."
        ) from exc
    if "text/html" not in content_type:
        continue

    parser = LinkParser()
    parser.feed(body.decode("utf-8", errors="ignore"))
    for href in parser.hrefs:
        if href.startswith("?") or href.startswith("#") or href in {"../", "/"}:
            continue
        absolute = urljoin(url, href)
        parsed = urlparse(absolute)
        if parsed.netloc != "physionet.org":
            continue
        if not absolute.startswith(start_url):
            continue
        relative = absolute[len(start_url):]
        if not (relative.startswith("hosp/") or relative.startswith("icu/")):
            continue
        if absolute.endswith("/"):
            queue.append(absolute)
        elif absolute.endswith(".csv.gz"):
            files.append(absolute)

if not files:
    raise SystemExit(
        "no MIMIC-IV CSV files were discovered. Check credentials and version."
    )

for file_url in sorted(set(files)):
    relative = file_url[len(start_url):]
    path = dest_root / relative
    if path.exists() and path.stat().st_size > 0:
        print(f"[sample-dbs] cached {relative}", file=sys.stderr)
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"[sample-dbs] downloading {relative}", file=sys.stderr)
    with opener.open(file_url) as response, tmp_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    tmp_path.replace(path)

print(dest_root)
PY
}

resolve_mimiciv_data_dir() {
  if [[ -n "${MIMIC_IV_DATA_DIR:-}" ]]; then
    [[ -d "$MIMIC_IV_DATA_DIR/hosp" ]] || die "MIMIC_IV_DATA_DIR is missing hosp/: $MIMIC_IV_DATA_DIR"
    [[ -d "$MIMIC_IV_DATA_DIR/icu" ]] || die "MIMIC_IV_DATA_DIR is missing icu/: $MIMIC_IV_DATA_DIR"
    printf '%s\n' "$MIMIC_IV_DATA_DIR"
    return
  fi

  local data_dir="$CACHE_DIR/$MIMIC_IV_CACHE_NAME/$MIMIC_IV_VERSION"
  if [[ -d "$data_dir/hosp" && -d "$data_dir/icu" ]]; then
    printf '%s\n' "$data_dir"
    return
  fi

  if [[ "${MIMIC_IV_AUTO_DOWNLOAD:-1}" != "0" ]]; then
    download_mimiciv_from_physionet
    return
  fi

  die "MIMIC-IV data not found. Set MIMIC_IV_DATA_DIR=/path/to/mimiciv/$MIMIC_IV_VERSION or enable MIMIC_IV_AUTO_DOWNLOAD=1 with PhysioNet credentials."
}

psql_file_with_vars() {
  local name="$1"
  local db="$2"
  local user="$3"
  local file="$4"
  shift 4
  docker exec -i "$name" psql -U "$user" -d "$db" -v ON_ERROR_STOP=1 "$@" < "$file"
}

bootstrap_mimiciv() {
  require_docker
  local data_dir
  data_dir="$(resolve_mimiciv_data_dir)"
  local code_dir
  code_dir="$(resolve_mimic_code_dir)"
  local pg_build_dir="$code_dir/mimic-iv/buildmimic/postgres"

  [[ -f "$pg_build_dir/create.sql" ]] || die "missing MIMIC create.sql in $pg_build_dir"
  [[ -f "$pg_build_dir/load_gz.sql" ]] || die "missing MIMIC load_gz.sql in $pg_build_dir"

  ensure_mimiciv_container
  wait_for_postgres "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER"

  if table_has_rows "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" "mimiciv_hosp.patients"; then
    log "MIMIC-IV is already loaded"
  else
    ensure_role_exists "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" "postgres"
    log "creating MIMIC-IV schemas"
    psql_file_with_vars "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" "$pg_build_dir/create.sql"
    log "loading MIMIC-IV data; large event tables can take hours"
    psql_file_with_vars \
      "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" \
      "$pg_build_dir/load_gz.sql" \
      -v "mimic_data_dir=/mimiciv/$MIMIC_IV_VERSION"
    log "applying MIMIC-IV constraints"
    psql_file_with_vars \
      "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" \
      "$pg_build_dir/constraint.sql" \
      -v "mimic_data_dir=/mimiciv/$MIMIC_IV_VERSION"
    log "creating MIMIC-IV indexes"
    psql_file_with_vars \
      "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" \
      "$pg_build_dir/index.sql" \
      -v "mimic_data_dir=/mimiciv/$MIMIC_IV_VERSION"
  fi

  grant_readonly_role_many_schemas \
    "$MIMIC_IV_CONTAINER" "$MIMIC_IV_DB" "$MIMIC_IV_USER" \
    "mimiciv_hosp" "mimiciv_icu"
  log "MIMIC-IV ready: postgresql://$MIMIC_IV_USER:$MIMIC_IV_PASSWORD@127.0.0.1:$MIMIC_IV_PORT/$MIMIC_IV_DB"
}

bootstrap_mimiciv_demo() {
  MIMIC_IV_CONTAINER="${MIMIC_IV_DEMO_CONTAINER:-rl-task-foundry-mimiciv-demo}"
  MIMIC_IV_VOLUME="${MIMIC_IV_DEMO_VOLUME:-rl-task-foundry-mimiciv-demo-data}"
  MIMIC_IV_PORT="${MIMIC_IV_DEMO_PORT:-5436}"
  MIMIC_IV_DB="${MIMIC_IV_DEMO_DB:-mimiciv_demo}"
  MIMIC_IV_USER="${MIMIC_IV_DEMO_USER:-mimiciv_demo}"
  MIMIC_IV_PASSWORD="${MIMIC_IV_DEMO_PASSWORD:-mimiciv_demo}"
  MIMIC_IV_VERSION="${MIMIC_IV_DEMO_VERSION:-2.2}"
  MIMIC_IV_BASE_URL="${MIMIC_IV_DEMO_BASE_URL:-https://physionet.org/files/mimic-iv-demo}"
  MIMIC_IV_CACHE_NAME="${MIMIC_IV_DEMO_CACHE_NAME:-mimiciv_demo}"
  MIMIC_IV_REQUIRES_CREDENTIALS=0
  MIMIC_CODE_REF="${MIMIC_CODE_REF:-v2.4.0}"
  MIMIC_CODE_DIR="${MIMIC_CODE_DIR:-$CACHE_DIR/mimic-code-$MIMIC_CODE_REF}"
  bootstrap_mimiciv
}

case "$target" in
  pagila)
    bootstrap_pagila
    ;;
  postgres_air)
    bootstrap_postgres_air
    ;;
  mimiciv|mimic_iv)
    bootstrap_mimiciv
    ;;
  mimiciv_demo|mimic_iv_demo)
    bootstrap_mimiciv_demo
    ;;
  all)
    bootstrap_pagila
    bootstrap_postgres_air
    ;;
  *)
    die "usage: scripts/bootstrap_sample_dbs.sh [pagila|postgres_air|mimiciv|mimiciv_demo|all]"
    ;;
esac
