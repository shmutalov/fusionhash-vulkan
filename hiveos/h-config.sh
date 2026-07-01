#!/usr/bin/env bash
# Generates the vulkminer command line from the HiveOS flight sheet.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf

# Pool URL: normalise to a ws:// / wss:// WebSocket URL.
URL="$CUSTOM_URL"
URL="${URL#stratum+tcp://}"
URL="${URL#stratum+ssl://}"
case "$URL" in
    ws://*|wss://*) ;;
    ssl://*)  URL="wss://${URL#ssl://}" ;;
    *)        URL="ws://$URL" ;;
esac

WALLET="$CUSTOM_TEMPLATE"
PASS="${CUSTOM_PASS:-x}"

CONF="--pool $URL --user $WALLET --pass $PASS --stats-file ${CUSTOM_LOG_BASENAME}.stats.json"

# Freeform extra args from the flight sheet "Extra config arguments" box,
# e.g. --intensity 0.5 -d 0
[[ -n "$CUSTOM_USER_CONFIG" ]] && CONF="$CONF $CUSTOM_USER_CONFIG"

echo "$CONF" > "$CUSTOM_CONFIG_FILENAME"
