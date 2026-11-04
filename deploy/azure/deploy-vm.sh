#!/usr/bin/env bash
# Create an Azure VM and run zone az (the HUB) on it.
#
#   SPOKE_CIDRS="203.0.113.7/32 198.51.100.0/28" ./deploy/azure/deploy-vm.sh
#
# SPOKE_CIDRS: the egress IPs of the spokes (zone b's proxy, zone cn's NAT). Only they may reach the hub
#              port 9090. Default "Internet", which is acceptable only with HTTPS in front of the hub.
# Env: RG (agentzones-rg), LOCATION (westeurope), VM (agentzones-hub), SIZE (Standard_D4s_v5),
#      ADMIN_CIDR (ssh; default: your IP), ZONE_B_TOKEN / ZONE_CN_TOKEN (default: generated and printed).
set -euo pipefail
RG=${RG:-agentzones-rg}; LOCATION=${LOCATION:-westeurope}; VM=${VM:-agentzones-hub}
SIZE=${SIZE:-Standard_D4s_v5}; SPOKE_CIDRS=${SPOKE_CIDRS:-Internet}
ADMIN_CIDR=${ADMIN_CIDR:-$(curl -fsS https://api.ipify.org)/32}
ZONE_B_TOKEN=${ZONE_B_TOKEN:-$(openssl rand -base64 32 | tr -d '/+=')}
ZONE_CN_TOKEN=${ZONE_CN_TOKEN:-$(openssl rand -base64 32 | tr -d '/+=')}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)

az group create -n "$RG" -l "$LOCATION" -o none
az vm create -g "$RG" -n "$VM" --image Ubuntu2204 --size "$SIZE" --admin-username azureuser \
  --generate-ssh-keys --custom-data "$HERE/cloud-init.yaml" --nsg "$VM-nsg" --nsg-rule NONE \
  --public-ip-sku Standard -o none
az network nsg rule create -g "$RG" --nsg-name "$VM-nsg" -n ssh --priority 100 \
  --source-address-prefixes "$ADMIN_CIDR" --destination-port-ranges 22 --access Allow --protocol Tcp -o none
# shellcheck disable=SC2086
az network nsg rule create -g "$RG" --nsg-name "$VM-nsg" -n agentzones-hub --priority 110 \
  --source-address-prefixes $SPOKE_CIDRS --destination-port-ranges 9090 --access Allow --protocol Tcp -o none

PUB=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
ssh -o StrictHostKeyChecking=accept-new azureuser@"$PUB" 'cloud-init status --wait >/dev/null; docker --version'
rsync -az --exclude .git --exclude data "$ROOT/" azureuser@"$PUB":/opt/agentzones/
ssh azureuser@"$PUB" "cd /opt/agentzones && umask 077 && cat > zones/az/.env <<ENV
GATEWAY_BIND=0.0.0.0
ZONE_B_TOKEN=$ZONE_B_TOKEN
ZONE_CN_TOKEN=$ZONE_CN_TOKEN
ENV
sudo scripts/zone.sh az"

cat <<MSG
Hub running on $PUB (port 9090, open to: $SPOKE_CIDRS).

Zone b:   export HUB_URL=http://$PUB:9090 HUB_PROXY=http://<proxy>:<port> HUB_TOKEN=$ZONE_B_TOKEN
Zone cn:  export HUB_URL=http://$PUB:9090 HUB_TOKEN=$ZONE_CN_TOKEN

Local API of az via SSH:  ssh -L 8080:127.0.0.1:8080 azureuser@$PUB   then   curl localhost:8080/status
MSG
