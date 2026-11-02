#!/usr/bin/env bash
# Create an Azure VM and run an Agent Zone on it.
#
#   ZONE_NAME=azure-a ZONE_SECRET=... ZONE_SEED=http://<other-zone>:7700 \
#   PEER_CIDR=10.0.0.0/8 ./deploy/azure/deploy-vm.sh
#
# Env: RG (agentzones-rg), LOCATION (westeurope), VM (=$ZONE_NAME), SIZE (Standard_D4s_v5;
#      GPU: Standard_NC4as_T4_v3), PEER_CIDR (who may reach port 7700), ADMIN_CIDR (ssh; default: your IP),
#      VNET/SUBNET (optional: join an existing network, recommended for private peering).
set -euo pipefail
: "${ZONE_NAME:?set ZONE_NAME}" "${ZONE_SECRET:?set ZONE_SECRET}"
RG=${RG:-agentzones-rg}; LOCATION=${LOCATION:-westeurope}; VM=${VM:-$ZONE_NAME}
SIZE=${SIZE:-Standard_D4s_v5}; PEER_CIDR=${PEER_CIDR:-10.0.0.0/8}
ADMIN_CIDR=${ADMIN_CIDR:-$(curl -fsS https://api.ipify.org)/32}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)

az group create -n "$RG" -l "$LOCATION" -o none
NET_ARGS=()
[[ -n "${VNET:-}" ]] && NET_ARGS=(--vnet-name "$VNET" --subnet "${SUBNET:-default}")
az vm create -g "$RG" -n "$VM" --image Ubuntu2204 --size "$SIZE" --admin-username azureuser \
  --generate-ssh-keys --custom-data "$HERE/cloud-init.yaml" --nsg "$VM-nsg" --nsg-rule NONE \
  --public-ip-sku Standard "${NET_ARGS[@]}" -o none
az network nsg rule create -g "$RG" --nsg-name "$VM-nsg" -n ssh --priority 100 \
  --source-address-prefixes "$ADMIN_CIDR" --destination-port-ranges 22 --access Allow --protocol Tcp -o none
az network nsg rule create -g "$RG" --nsg-name "$VM-nsg" -n agentzone --priority 110 \
  --source-address-prefixes "$PEER_CIDR" --destination-port-ranges 7700 --access Allow --protocol Tcp -o none

PUB=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
PRIV=$(az vm show -d -g "$RG" -n "$VM" --query privateIps -o tsv)
ZONE_URL=${ZONE_URL:-http://$PRIV:7700}
echo "VM $VM: public $PUB, private $PRIV; zone URL $ZONE_URL"

ssh -o StrictHostKeyChecking=accept-new azureuser@"$PUB" 'cloud-init status --wait >/dev/null; docker --version'
rsync -az --exclude .git --exclude .zone-data --exclude data "$ROOT/" azureuser@"$PUB":/opt/agentzones/
ssh azureuser@"$PUB" "cd /opt/agentzones && cat > .env <<ENV
ZONE_NAME=$ZONE_NAME
ZONE_URL=$ZONE_URL
ZONE_SEED=${ZONE_SEED:-}
ZONE_SECRET=$ZONE_SECRET
ZONE_PROFILE=${ZONE_PROFILE:-uniform}
MODEL_NAME=${MODEL_NAME:-qwen2.5:1.5b}
HTTPS_PROXY=${HTTPS_PROXY:-}
HTTP_PROXY=${HTTP_PROXY:-}
ENV
sudo docker compose up -d --build"
echo "Zone running. From a machine in $PEER_CIDR: curl $ZONE_URL/health"
echo "Dashboard via SSH tunnel: ssh -L 7700:localhost:7700 azureuser@$PUB  ->  http://localhost:7700"
