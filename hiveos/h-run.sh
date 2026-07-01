#!/usr/bin/env bash
# Launches vulkminer with the generated config. HiveOS captures stdout/stderr
# into the miner log.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf

# (Re)generate the config from the current flight sheet.
[[ -e h-config.sh ]] && bash h-config.sh

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

CONF="$(cat "$CUSTOM_CONFIG_FILENAME" 2>/dev/null)"

# RUST_LOG keeps share/accept lines at info level.
export RUST_LOG="${RUST_LOG:-info}"

# shellcheck disable=SC2086
exec ./vulkminer $CONF
