#!/usr/bin/env bash
# Reports miner stats to the HiveOS agent. Sourced by the agent, which reads the
# `khs` (total kH/s) and `stats` (JSON) variables. Reads the miner's own
# stats.json and merges temp/fan from HiveOS, matched by PCI bus.
cd "$(dirname "$0")" 2>/dev/null || true
. h-manifest.conf

khs=0
stats=""

STATS_FILE="${CUSTOM_LOG_BASENAME}.stats.json"
data=$(cat "$STATS_FILE" 2>/dev/null)

if [[ -n "$data" ]] && echo "$data" | jq -e . >/dev/null 2>&1; then
    khs=$(echo "$data" | jq -r '.khs // 0')
    acc=$(echo "$data" | jq -r '.accepted // 0')
    rej=$(echo "$data" | jq -r '.rejected // 0')
    uptime=$(echo "$data" | jq -r '.uptime // 0')
    hs=$(echo "$data" | jq -c '[.gpus[].khs]')
    buses=$(echo "$data" | jq -c '[.gpus[].bus]')

    # temp/fan from HiveOS, matched to our GPUs by PCI bus.
    temp="[]"
    fan="[]"
    GS=/run/hive/gpu-stats.json
    if [[ -f "$GS" ]]; then
        mapfile -t HB < <(jq -r '.busids[]?' "$GS" 2>/dev/null)
        mapfile -t HT < <(jq -r '.temp[]?'   "$GS" 2>/dev/null)
        mapfile -t HF < <(jq -r '.fan[]?'    "$GS" 2>/dev/null)
        mapfile -t MYBUS < <(echo "$data" | jq -r '.gpus[].bus')
        temps=()
        fans=()
        for b in "${MYBUS[@]}"; do
            hex=$(printf "%02x" "$b" 2>/dev/null)
            t=0; f=0
            for i in "${!HB[@]}"; do
                pref="${HB[$i]%%:*}"
                if [[ "${pref,,}" == "$hex" ]]; then
                    t="${HT[$i]:-0}"; f="${HF[$i]:-0}"; break
                fi
            done
            temps+=("$t"); fans+=("$f")
        done
        temp=$(printf '%s\n' "${temps[@]}" | jq -R 'tonumber? // 0' | jq -sc .)
        fan=$(printf '%s\n' "${fans[@]}"  | jq -R 'tonumber? // 0' | jq -sc .)
    fi

    stats=$(jq -nc \
        --argjson hs "$hs" \
        --argjson temp "$temp" \
        --argjson fan "$fan" \
        --argjson bus "$buses" \
        --arg uptime "$uptime" \
        --argjson acc "$acc" \
        --argjson rej "$rej" \
        '{hs:$hs, hs_units:"khs", temp:$temp, fan:$fan, uptime:($uptime|tonumber),
          ar:[$acc,$rej], algo:"cn/gpu", bus_numbers:$bus}')
fi

[[ -z "$khs" || "$khs" == "null" ]] && khs=0
