#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
NS3_VERSION="3.47"
NS3_ROOT="/opt/tnsm-ns3"
ARCHIVE="ns-3.${NS3_VERSION#3.}.tar.bz2"
SOURCE_DIR="${NS3_ROOT}/ns-${NS3_VERSION}"
DOWNLOAD_URL="https://www.nsnam.org/releases/${ARCHIVE}"
EXPECTED_SHA1="897c3f2db2c6de6a8291e46b3020dbeb1af75269"

echo "[1/6] Updating Ubuntu package metadata"
apt-get update

echo "[2/6] Installing the documented ns-3 build prerequisites"
apt-get install -y --no-install-recommends \
    build-essential python3 cmake ninja-build git ca-certificates wget bzip2 pkg-config

mkdir -p "${NS3_ROOT}"
cd "${NS3_ROOT}"

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[3/6] Downloading ns-${NS3_VERSION}"
    wget --https-only --secure-protocol=TLSv1_2 -O "${ARCHIVE}" "${DOWNLOAD_URL}"
else
    echo "[3/6] Reusing the existing archive ${ARCHIVE}"
fi

echo "[4/6] Verifying the official SHA-1 checksum"
echo "${EXPECTED_SHA1}  ${ARCHIVE}" | sha1sum --check --strict

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "[5/6] Extracting the verified source archive"
    tar -xjf "${ARCHIVE}"
else
    echo "[5/6] Reusing the existing source directory ${SOURCE_DIR}"
fi

cd "${SOURCE_DIR}"
echo "[6/6] Configuring and building the required ns-3 modules"
if ! id -u tnsmsim >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash tnsmsim
fi
chown -R tnsmsim:tnsmsim "${NS3_ROOT}"
runuser -u tnsmsim -- ./ns3 configure \
    --build-profile=optimized \
    --disable-examples \
    --disable-tests \
    --enable-modules=core,network,mobility,propagation,wifi,internet,applications,flow-monitor
runuser -u tnsmsim -- ./ns3 build

echo "NS3_INSTALL_COMPLETE=1"
echo "NS3_VERSION=${NS3_VERSION}"
echo "NS3_SOURCE_DIR=${SOURCE_DIR}"
